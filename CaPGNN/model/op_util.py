import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
import dgl
import gevent
import numpy as np
import torch
from typing import Dict, Tuple
from functools import wraps
from typing import Tuple

from numba import njit
from torch import Tensor, autocast
from torch import jit

from ..helper import BitType
from ..communicator import Basic_Buffer_Type
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine
from ..util.caches import CACHEALG, create_cache
from ..util.timer import TimerRecord

'''
*************************************************
****************** decorators *******************
*************************************************
'''

def wrap_stream(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if engine.ctx.use_parallel:
            with torch.cuda.stream(engine.ctx.marginal_stream):
                result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

def wrap_stream_force(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if engine.ctx.our_cache is True or engine.ctx.our_partition is True:
            # engine.ctx._comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(engine.ctx._comm_stream):
                result = func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        return result
    return wrapper

'''
*************************************************
********** message exchange functions ***********
*************************************************
'''

def msg_all2all_GLOO(send_messages: Tensor,
                     name: str,
                     is_train: bool = True) -> Tensor:
    '''
    perform messages exchange between all workers.
    '''
    # get necessary params
    msg_dim = send_messages.shape[-1]
    msg_dtype = send_messages.dtype
    num_remote = engine.ctx.num_remove
    send_idx = engine.ctx.send_idx
    recv_idx = engine.ctx.recv_idx
    total_send_idx = engine.ctx.total_send_idx
    return fp_msg_transfer_process(send_messages, total_send_idx, send_idx, recv_idx, msg_dim, msg_dtype, num_remote, name, is_train)


# @wrap_stream_force
def fp_msg_transfer_process(send_messages: Tensor,
                            total_send_idx,
                            send_idx: Dict[int, Tuple[int, int]],
                            recv_idx: Basic_Buffer_Type,
                            msg_dim: int,
                            msg_dtype: torch.dtype,
                            num_remote: int,
                            name: str,
                            is_train: bool) -> Tensor:
    # get communication buffer
    layer = int(name[-1])
    if not is_train:
        # xxx_cpu is pinned memory
        recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu = comm.ctx.comm_buffer.get_test_buffer(layer)
    else:
        recv_buffer_cpu, recv_buffer_gpu, send_buffer_cpu = comm.ctx.comm_buffer.get_train_buffer(name)
    # communication
    # with torch.profiler.record_function('msg_communication'):
    # with torch.cuda.stream(engine.ctx.cache_stream[0]):
    if not engine.ctx.use_pipeline: engine.ctx.timer.start(f'msg_comm')
    comm.ctx.fp_msg_exchange(recv_buffer_cpu=recv_buffer_cpu, recv_buffer_gpu=recv_buffer_gpu,
                             send_buffer_cpu=send_buffer_cpu, total_send_idx=total_send_idx,
                             send_idx=send_idx, recv_idx=recv_idx, send_messages=send_messages,
                             storage_server=engine.ctx.storage_server, g_info=engine.ctx.g_info,
                             gpb=engine.ctx.gpb, timer=engine.ctx.timer,
                             _comm_stream=engine.ctx._comm_stream, _corr_stream=engine.ctx._corr_stream,
                             use_cache=engine.ctx.our_cache, use_pipeline=engine.ctx.use_pipeline, name=name)
    if not engine.ctx.use_pipeline:  engine.ctx.timer.stop(f'msg_comm')
    if engine.ctx.our_cache and name.startswith('forward'):
        # engine.ctx.timer.start(f'pick_cache')
        with engine.ctx.timer.record('pick_cache'):
            remote_messages = pick_cache(recv_idx, recv_buffer_cpu, recv_buffer_gpu, msg_dim, msg_dtype, num_remote, name, engine.ctx.storage_server)
        # engine.ctx.timer.stop(f'pick_cache')
    else:
        with engine.ctx.timer.record('merge_msg'):
            remote_messages = torch.zeros(num_remote, msg_dim, dtype=msg_dtype, device=comm.ctx.device)
            # 将接收到的数据装填到remote_messages
            for pid, idx in recv_idx.items():
                if remote_messages[idx].dtype != recv_buffer_gpu[0][pid].dtype:
                    remote_messages[idx] = recv_buffer_gpu[0][pid].half()  # Convert the target tensor to half type
                else:
                    remote_messages[idx] = recv_buffer_gpu[0][pid]

    engine.ctx._f_cuda_event[layer].record(engine.ctx._comm_stream)
    engine.ctx._f_cpu_event[layer].set()
    return remote_messages

@wrap_stream_force
def pick_cache(recv_idx, recv_buffer_cpu, recv_buffer_gpu,
               msg_dim, msg_dtype, num_remote, name,
               storage_server) -> Tensor:
    # 2. When receiving, the cache will be fetched directly;
    remote_messages = torch.zeros(num_remote, msg_dim, dtype=msg_dtype, device=comm.ctx.device)
    non_cache_mask_all = torch.ones(remote_messages.size(0), dtype=torch.bool, device=comm.ctx.device)  #
    cached_num_l = 0
    cached_num_g = 0
    # Extract features in the cache
    # Select the remote node on the current processing partition
    for pid, ids in recv_idx.items():
        sub_remote_messages = remote_messages[ids]
        # cache_info = recv_buffer_cpu[1][pid]
        # valid_cache_indices = cache_info[1:cache_info[0]+1]
        # valid_recv_values_num = sub_remote_messages.size(0)-valid_cache_indices.size(0)
        # recv_non_cache_values = recv_buffer_gpu[0][pid][:valid_recv_values_num]
        # all_global_ids = engine.ctx.g_info.node_feats[dgl.NID][engine.ctx.g_info.num_inner + ids]
        # cache_info = recv_buffer_cpu[1][pid]
        # valid_cache_indices = cache_info[1:cache_info[0]+1]
        # valid_recv_values_num = sub_remote_messages.size(0)-valid_cache_indices.size(0)
        # -------------------------------- #
        recv_non_cache_values = recv_buffer_gpu[0][pid]
        all_global_ids = engine.ctx.g_info.node_feats[dgl.NID][engine.ctx.g_info.num_inner + ids]
        # Create a Boolean mask to indicate where the missed cache is located
        # non_cache_mask = torch.ones(sub_remote_messages.size(0), dtype=torch.bool, device=sub_remote_messages.device) #
        non_cache_mask = non_cache_mask_all[:sub_remote_messages.size(0)]
        non_cache_mask.fill_(True)
        cached_local_feature, cached_local_indices = storage_server.cache_server.get_halo_feature_from_local(name=name, feature_ids=all_global_ids)
        if cached_local_indices is not None: non_cache_mask[cached_local_indices] = False
        cached_global_feature, cached_global_indices = storage_server.cache_server.get_halo_feature_from_global2(name=name, feature_ids=all_global_ids[non_cache_mask], device=comm.ctx.device, add_local=True)
        if cached_global_indices is not None: non_cache_mask[cached_global_indices] = False
        non_cache_num = torch.count_nonzero(non_cache_mask)
        # Add uncached gpu to gpu
        if non_cache_num>0: sub_remote_messages[non_cache_mask].copy_(recv_non_cache_values[:non_cache_num], non_blocking=True)
        # Add the hit cache to gpu
        if cached_global_indices is not None:
            sub_remote_messages[cached_global_indices[:cached_global_feature.shape[0]]].copy_(cached_global_feature, non_blocking=True)
            cached_num_g += len(cached_global_indices)
        if cached_local_indices is not None:
            sub_remote_messages[cached_local_indices[:cached_local_feature.shape[0]]].copy_(cached_local_feature, non_blocking=True)
            cached_num_l += len(cached_local_indices)
        # -------------------------------- #
    # print(f"[{int(time.time())}] {comm.ctx.get_rank()}->{pid} [{name}]: {cached_num_l+cached_num_g}/{num_remote} [l:{cached_num_l}, g:{cached_num_g}]")
    # storage_server.cache_server.cache_streams[0].synchronize()
    # storage_server.cache_server.cache_streams[1].synchronize()
    return remote_messages

