import asyncio
import os
import logging
import threading
import time
from multiprocessing.pool import ThreadPool
import dgl
import gevent
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from typing import Dict, List, Any, Tuple
from queue import Queue
from .buffer import CommBuffer, Basic_Buffer_Type
from ..helper import MessageType
from ..util.timer import TimerRecord
logger = logging.getLogger('trainer')

class Communicator(object):
    '''
    the communicator class for distributed training. Communicator is a wrapper of torch.distributed, and managers all the communication buffers and operations.
    '''
    def __init__(self, backend: str ='gloo', init_method: str ='env://', server_num=1):
        self._init(backend, init_method)
        self.comm_buffer: CommBuffer = None
        # set ctx (do not share across processes)
        self.server_num = server_num
        Communicator.ctx = self
        self._create_groups()


    def _init(self, backend: str, init_method: str="env://"):
        '''
        initialize the communicator.
        '''
        # Ensure support for NCCL and Gloo backends
        if backend not in ['gloo', 'nccl']: raise NotImplementedError('Supported backends are gloo and nccl.')
        # When using NCCL, you must use a GPU device
        if backend == 'nccl' and not torch.cuda.is_available(): raise RuntimeError('NCCL backend requires CUDA.')

        dist.init_process_group(backend, init_method=init_method)
        assert dist.is_initialized()
        self._backend = backend
        self._init_method = init_method
        self._local_rank = int(os.environ.get('LOCAL_RANK', None))
        self._device = torch.device(f'cuda:{self._local_rank}')
        # torch.cuda.set_device(self._device)
        self.queue_stream = [torch.cuda.Stream() for i in range(6)]

    def _create_groups(self):
        parts_env = os.environ.get('NUM_PARTS_LIST', None)
        if not parts_env: raise RuntimeError('NUM_PARTS_LIST not set. Example: "2,4,4" for 3 nodes.')
        self.parts = list(map(int, parts_env.split(',')))
        if len(self.parts) != self.server_num: raise AssertionError('NUM_PARTS_LIST长度需等于节点数 (server_num)。')

        local_world_size = self.parts[self.server_id]
        self.global_start_rank = sum(self.parts[:self.server_id])
        self.global_end_rank = self.global_start_rank + local_world_size
        self.local_ranks = list(range(local_world_size))                                    # local_ranks 应该从 0 开始连续编号
        self.global_ranks = list(range(self.global_start_rank, self.global_end_rank))       # global_ranks 在全局中唯一，跨节点连续编号
        self.all_global_ranks = list(range(sum(self.parts)))
        self.global_leaders_rank = [sum(self.parts[:i]) for i in range(self.server_num)]    # leaders的全局rank. 第一个rank当做leader
        # patch
        self.all_node_groups = []
        start = 0
        for i in range(self.server_num):
            end = start + self.parts[i]
            self.all_node_groups.append(dist.new_group(ranks=list(range(start, end))))
            start = end
        self.local_group = self.all_node_groups[self.server_id]
        # self.local_group = dist.new_group(ranks=self.global_ranks)
        # print(f">> [{self._local_rank}/{dist.get_rank()}] local_ranks={self.local_ranks}, local_group={self.local_group}")
        # self.global_group = dist.new_group(ranks=self.global_leaders_rank)
        dist.barrier()
        self.all_global_group = dist.new_group(ranks=self.all_global_ranks)
        # print(f">> [{self._local_rank}/{dist.get_rank()}] all_global_ranks={self.all_global_ranks}, all_global_group={self.all_global_group}")
        # print(f">> [{self._local_rank}/{dist.get_rank()}] global_ranks={self.global_ranks}, global_group={self.global_group}")

        # local_size = self.get_local_world_size()
        # node_rank = self.server_id
        # start_rank = node_rank * local_size
        # local_ranks = list(range(start_rank, start_rank + local_size))
        # self.local_group = dist.new_group(ranks=local_ranks)
        # global_ranks = [i * local_size for i in range(self.server_num)]
        # self.global_group = dist.new_group(ranks=global_ranks)


    def __repr__(self):
        return f'<Communicator(rank: {self.get_rank()}, backend: {self.backend}, world_size: {self.get_world_size()}, local_rank: {self.local_rank}, device: {self.device})>'

    '''
    *************************************************
    ***************** getter methods ****************
    *************************************************
    '''

    @property
    def server_id(self):
        return int(os.environ['NODE_RANK'])

    @property
    def local_rank(self):
        return self._local_rank
    
    @property
    def device(self):
        return self._device
    
    @property
    def init_method(self):
        return self._init_method
    
    @property
    def backend(self):
        return self._backend

    @staticmethod
    def get_local_world_size():
        return int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    @staticmethod
    def get_rank(group=None):
        return dist.get_rank(group=group)

    @staticmethod
    def get_world_size(group=None):
        return dist.get_world_size(group=group)

    @staticmethod
    def get_backend(group=None):
        return dist.get_backend(group=group)

    @staticmethod
    def _destroy(group=None):
        dist.destroy_process_group(group=group)

    def __del__(self):
        self._destroy()

    @staticmethod
    def barrier(group=None):
        dist.barrier(group=group)

    '''
    *************************************************
    ************* collective primitives *************
    *************************************************
    '''

    def hierarchical_allreduce(self, tensor, average: bool = True):
        '''
        Example of manual hierarchical allreduce:
          1. First intra-node
          2. Then inter-node
        '''
        self.all_reduce_sum(tensor, group=self.all_global_group)

        # world_size = dist.get_world_size()
        # local_size = dist.get_world_size(group=self.local_group)
        # # print(f"[{self.local_rank}] world_size={world_size}, local_size={local_size}, is_in_local_group={dist.get_rank(group=self.local_group)}, is_in_global_group={dist.get_rank(group=self.global_group)}")
        # # 1. 节点内聚合（本机所有进程都要调用）
        # if local_size > 1: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.local_group)
        # # 2. 节点间聚合（只有 leader 参与，因为只有 leader 在 global_group 里）
        # if self.local_rank == 0: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.global_group)
        # # 3. 节点内广播（本机所有进程都要调用；src 必须是本机 leader 的“全局 rank”）
        # if local_size > 1: dist.broadcast(tensor, src=self.global_start_rank, group=self.local_group)
        # if average: tensor.div_(world_size)



    def hierarchical_barrier(self):
        torch.cuda.synchronize()
        self.barrier(group=self.local_group)
        # self.barrier(group=self.all_global_group)


    @staticmethod
    def get_global_rank(rank, group):
        if group is None: return rank
        return dist.get_global_rank(group, rank)

    @staticmethod
    def all_reduce_max(tensor: Tensor, group=None):
        '''
        all reduce the tensor with max operation.
        '''
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    
    @staticmethod
    def all_reduce_sum(tensor: Tensor, group=None):
        '''
        all reduce the tensor with sum operation.
        '''
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

    @staticmethod
    def all_gather_any(obj_list: List[Any], obj: Any, group=None):
        '''
        all gather any objects across all the workers. (only support by gloo backend)
        '''
        dist.all_gather_object(obj_list, obj, group=group)
    
    @staticmethod
    def broadcast_any(obj_list: List[Any], src: int = 0, group=None):
        '''
        broadcast objects to all the workers
        '''
        dist.broadcast_object_list(obj_list, Communicator.get_global_rank(src, group), group=group)

    @staticmethod
    def scatter_any(output_list: List[Any], input_list: List[Any], src: int = 0, group=None):
        '''
        scatter objects from input_list to output_list. (only support by gloo backend)
        '''
        dist.scatter_object_list(output_list, input_list, Communicator.get_global_rank(src, group), group=group)
    
    @staticmethod
    def gather_any(obj, obj_list: List[Any], dst: int = 0, group=None):
        dist.gather_object(obj, obj_list, Communicator.get_global_rank(dst, group), group=group)

    # p2p primitives
    @staticmethod
    def sync_send(tensor: Tensor, dst: int, tag: MessageType, group=None):
        '''
        send tensor to dst synchronously.
        '''
        return dist.send(tensor, Communicator.get_global_rank(dst, group), tag=tag.value, group=group)

    @staticmethod
    def sync_recv(tensor: Tensor, src: int, tag: MessageType, group=None):
        '''
        receive tensor from src synchronously.
        '''
        return dist.recv(tensor, Communicator.get_global_rank(src, group), tag=tag.value, group=group)
    
    @staticmethod
    def async_send(tensor: Tensor, dst: int, tag: MessageType, group=None):
        '''
        send tensor to dst asynchronously.
        '''
        return dist.isend(tensor, Communicator.get_global_rank(dst, group), tag=tag.value, group=group)
    
    @staticmethod
    def async_recv(tensor: Tensor, src=None, tag: MessageType=MessageType.NONE, group=None):
        '''
        receive tensor from src asynchronously.
        '''
        return dist.irecv(tensor, Communicator.get_global_rank(src, group), tag=tag.value, group=group)

    '''
    *************************************************
    *********** messages exchange methods ***********
    *************************************************
    '''
    def process_request(self, r, left, recv_buffer_gpu, recv_buffer_cpu):
        r.wait()
        with torch.cuda.stream(self.queue_stream[left]):
            recv_buffer_gpu[0][left].copy_(recv_buffer_cpu[0][left], non_blocking=True)

    def fp_msg_exchange(self,
                        recv_buffer_cpu: Basic_Buffer_Type,
                        recv_buffer_gpu: Basic_Buffer_Type,
                        send_buffer_cpu: Basic_Buffer_Type,
                        total_send_idx,                         # The specific ID index of send_idx on all remote partitions. Such as [xx,xx,xx,...,xx]
                        send_idx: Dict[int, Tuple[int, int]],   # On each remote partition i, the start and end position of send_idx and the specific ID. For example {0: (0, 17660), 2: (17660, 36595)}
                        recv_idx: Basic_Buffer_Type,            # On the local partition, the HALO node belonging to the remote partition i is offset from the inner point ID. Such as {0: (0, 3, 6, 16), 2: (8, 9)}
                        send_messages: Tensor,
                        g_info, gpb,  timer, storage_server,
                        _comm_stream, _corr_stream,
                        use_cache=True, use_pipeline=False, name='',
                        ):
        '''
        all-to-all full-precision message exchange across all the worker
        '''
        # rank, world_size = self.get_rank(), self.get_world_size()
        rank, world_size = self.local_rank, self.get_local_world_size()
        # Store all request objects for sending/receiving operations
        req_send, req_recv = [], Queue()

        # Send and receive data to all other nodes
        _comm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(_comm_stream):
            for i in range(1, world_size):
                # The rank of the node where the current node receives data in the current loop
                left = (rank - i + world_size) % world_size
                # The rank of the node where the current node sends data in the current loop
                right = (rank + i) % world_size
                # Get the location range of data to be sent to the right node in send_messages
                retrieve_idx = send_idx[right]
                # Asynchronously receive data from the left node into recv_buffer_cpu[left]
                r2_0 = self.async_recv(recv_buffer_cpu[0][left], left, MessageType.DATA, group=self.local_group)
                ## r2_1 = self.async_recv(recv_buffer_cpu[1][left], left, MessageType.PARAMs)
                req_recv.put((r2_0, left))  # (r2_0, r2_1, left)

                if use_cache and name.startswith('forward'):
                    all_data = send_messages[retrieve_idx[0]:retrieve_idx[1]]
                    # ----------------------------------------------- #
                    #   Parse out the global ID of the node to be sent for searching from the cache
                    global_send_idx = gpb.partid2nids(rank)[0].item() + total_send_idx[retrieve_idx[0]:retrieve_idx[1]]
                    # 1. When sending, no sending in the cache;
                    # Find the key that exists in the cache
                    # timer.start(f'check_cache')
                    with timer.record('check_cache'):
                        cache_mask = storage_server.is_feature_in_cache(name, global_send_idx, features=all_data, target_rank=right)
                    # timer.stop(f'check_cache')
                    noncache_mask = ~cache_mask
                    # ----------------------------------------------- #
                    # Copy non-cache data
                    # Get non_cache_mask data (this data needs to be sent)
                    non_cache_data = all_data[noncache_mask]
                    send_values = send_buffer_cpu[0][right][:non_cache_data.size(0)]
                    # timer.start(f'copy_non_cache_data')
                    send_values.copy_(non_cache_data, non_blocking=True)
                    # timer.stop(f'copy_non_cache_data')

                    # send_cache_indices = send_buffer_cpu[1][right]
                    # cache_indices = torch.nonzero(cache_mask).squeeze()
                    # send_cache_indices[0] = min(send_cache_indices.numel()-1, cache_indices.numel())
                    # # Copy cache index, end_buffer_cpu is pinned memory
                    # send_cache_indices[1:send_cache_indices[0]+1].copy_(cache_indices[:send_cache_indices[0]], non_blocking=True)
                    # # Send graph structure instead of feature to reduce
                else:
                    # Send data asynchronously. Copy the data into send_buffer_cpu[right]
                    send_values = send_buffer_cpu[0][right]
                    # send_cache_indices = send_buffer_cpu[1][right]
                    send_values.copy_(send_messages[retrieve_idx[0]:retrieve_idx[1]], non_blocking=True)

                # Use the async_send function to send data asynchronously to the right node
                req_send.append(self.async_send(send_values, right, MessageType.DATA, group=self.local_group))
                # req_send.append(self.async_send(send_cache_indices, right, MessageType.PARAMs))
            # if not use_pipeline: timer.start(f'msg_comm')
            while not req_recv.empty():
                # Get a rank from the queue for receiving the request object and corresponding sending node
                r_0, left = req_recv.get(block=False)  # , r_1
                # Wait for the reception operation to complete
                r_0.wait()
                # r_1.wait()
                # Copy the received data from recv_buffer_cpu[left] to recv_buffer_gpu[left]
                recv_buffer_gpu[0][left].copy_(recv_buffer_cpu[0][left], non_blocking=True)
                # recv_buffer_gpu[1][left].copy_(recv_buffer_cpu[1][left], non_blocking=True)
            # Wait for the sending operation to complete
            for r in req_send: r.wait()
            # if not use_pipeline: timer.stop(f'msg_comm')


    '''
    *************************************************
    *********** buffer management methods ***********
    *************************************************
    '''

    def init_buffer(self, *args, **kwargs):
        '''
        wrapper to initialize the communication buffer
        '''
        self.comm_buffer = CommBuffer(*args, **kwargs, device=self.device)
    
    def update_buffer(self, *args, **kwargs):
        '''
        wrapper to update the communication buffer
        '''
        assert self.comm_buffer is not None, 'please initialize the communication buffer first'
        self.comm_buffer._update(*args, **kwargs)
    
    def delete_buffer(self, *args, **kwargs):
        '''
        wrapper to delete the communication buffer
        '''
        assert self.comm_buffer is not None, 'please initialize the communication buffer first'
        self.comm_buffer._delete(*args, **kwargs)

