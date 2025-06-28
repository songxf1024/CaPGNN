import multiprocessing
import os

import gpu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pickle
import asyncio
import copy
import ctypes
import queue
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple
import gevent
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from torch import nn, Tensor
# from torch.backends.quantized import engine
from ..manager import GraphEngine as engine
from tqdm import tqdm
from numba import jit, njit, prange
from loguru import logger
from CaPGNN.communicator import Communicator
from CaPGNN.helper import DistGNNType, MessageType
from CaPGNN.manager.processing import _get_agg_scores
from CaPGNN.util.timer import TimerRecord, Timer
from CaPGNN.util.caches import CACHEALG, create_cache
import dgl
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager, Array, RawArray, shared_memory
import cupy as cp
# from dgl import graphbolt as gb
from functools import lru_cache


DEBUG = False
def debug(msg):
    if DEBUG: print(msg)


@njit
def isin_numba(a, b):
    result = np.zeros(a.size, dtype=np.bool_)
    b_set = set(b)
    for i in range(a.size): result[i] = a[i] in b_set
    return result

def isin_numpy(a, b):
    return np.isin(convert_CNP(b, 'numpy'), convert_CNP(b, 'numpy'), assume_unique=True, kind='table')

isin_kernel = cp.RawKernel(r'''
extern "C" __global__
void isin(const int* a, const int* b, bool* mask, int a_size, int b_size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < a_size) {
        int val = a[tid];
        bool found = false;
        for (int i = 0; i < b_size; ++i) {
            if (val == b[i]) {
                found = true;
                break;
            }
        }
        mask[tid] = found;
    }
}
''', 'isin')

@njit
def filter_values(k, v, val):
    mask = k != val
    return k[mask], v[mask]

def isin_cupy(a, b, ret='numpy', stream=None):
    # return np.isin(convert_CNP(b, 'numpy'), convert_CNP(b, 'numpy'), assume_unique=True, kind='table')

    a = convert_CNP(a, 'cupy')
    b = convert_CNP(b, 'cupy')
    # 检查空数组情况
    if a.size == 0 or b.size == 0: return np.zeros(a.shape, dtype=np.bool_)
    return convert_CNP(cp.isin(a, b, assume_unique=True), ret)

    # if b.size == 0: return np.full_like(a, False, dtype='bool')
    # a = convert_CNP(a, 'cupy')
    # b = convert_CNP(b, 'cupy')
    # if stream is None: return convert_CNP(cp.isin(a, b, assume_unique=True), ret)
    # with stream: res = cp.isin(a, b, assume_unique=True)
    # stream.synchronize()
    # return convert_CNP(res, ret)

@njit
def unique_numba(arr):
    unique_set = set()
    unique_list = []
    for x in arr:
        if x not in unique_set:
            unique_set.add(x)
            unique_list.append(x)
    return np.array(unique_list), np.array([unique_list.index(x) for x in arr])

@njit
def find_valid_indices(feature_ids, all_keys_np, all_indices_np):
    # 初始化匹配掩码
    matching_mask = np.zeros(feature_ids.size, dtype=np.bool_)
    # 创建一个用于快速查找的集合
    all_keys_set = set(all_keys_np)
    # 查找 feature_ids 中的每个元素是否存在于 all_keys_np 中
    for i in range(feature_ids.size):
        if feature_ids[i] in all_keys_set:
            matching_mask[i] = True
    # 获取与 all_keys_np 匹配的 feature_ids
    matched_feature_ids = feature_ids[matching_mask]
    # 初始化结果掩码
    keys_mask = np.zeros(all_keys_np.size, dtype=np.bool_)
    # 创建一个用于快速查找的集合
    matched_feature_ids_set = set(matched_feature_ids)
    # 查找 all_keys_np 中的每个元素是否存在于匹配的 feature_ids 中
    for i in range(all_keys_np.size):
        if all_keys_np[i] in matched_feature_ids_set:
            keys_mask[i] = True
    # 使用 keys_mask 过滤 all_indices_np
    valid_indices_np = all_indices_np[keys_mask]
    return valid_indices_np

def convert_CNP(input_array, target_type, device=None):
    """
    将输入的数据类型转换为目标类型，支持 CuPy、NumPy 和 PyTorch Tensor 之间的转换。
    对于 CuPy 到 PyTorch 的转换，使用 DLPack 进行高效转换。

    参数:
    input_array: 输入的数组，可以是 CuPy 数组、NumPy 数组或 PyTorch Tensor
    target_type: 目标类型，可以是 'cupy', 'numpy', 或 'torch'

    返回:
    转换后的目标类型的数据
    """
    if target_type == None: return input_array
    if isinstance(input_array, cp.ndarray):
        # 从 CuPy 转换
        if target_type == 'numpy':
            return input_array.get()  # 或 cp.asnumpy(input_array)
        elif target_type == 'torch':
            # CuPy 到 PyTorch 使用 DLPack
            res = torch.utils.dlpack.from_dlpack(input_array.toDlpack())
            # return torch.from_numpy(input_array.get())
            # 如果目标设备是 None，则返回默认设备上的 tensor；否则转移到指定的设备
            if device: res = res.to(device)
            return res
        elif target_type == 'cupy':
            return input_array
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for CuPy array.")
    elif isinstance(input_array, np.ndarray):
        # 从 NumPy 转换
        if target_type == 'cupy':
            if device:
                with cp.cuda.Device(device[-1]):
                    res = cp.asarray(input_array)
            else:
                res = cp.asarray(input_array)
            return res
        elif target_type == 'torch':
            res = torch.from_numpy(input_array)
            # 如果目标设备是 None，则返回默认设备上的 tensor；否则转移到指定的设备
            if device: res = res.to(device)
            return res
        elif target_type == 'numpy':
            return input_array
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for NumPy array.")
    elif isinstance(input_array, torch.Tensor):
        # 从 PyTorch Tensor 转换
        if target_type == 'cupy':
            # 确保 tensor 在 GPU 上
            if input_array.device.type != 'cuda': raise ValueError("PyTorch tensor must be on GPU.")
            if device: input_array = input_array.to(device)
            return cp.from_dlpack(torch.utils.dlpack.to_dlpack(input_array))
            # return cp.asarray(input_array.cpu().numpy())
        elif target_type == 'numpy':
            return input_array.cpu().numpy()
        elif target_type == 'torch':
            if device: input_array = input_array.to(device)
            return input_array
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for PyTorch Tensor.")
    else:
        raise TypeError("Input must be a CuPy array, NumPy array, or PyTorch Tensor.")


# 替换节点时候是否使用基于节点重要性的优先级排序
use_weight_update = True
# 还有op_util.py的pick_cache2、comm.py的fp_msg_exchange
use_real_cache = True
# 节点重叠率使用降序排序，即优先选择重叠率高的
use_descending = True


class CacheServer:
    def __init__(self, gpus_num, dims, barrier, feat_type, gsize, cache_alg=CACHEALG.JACA):
        self.gsize = gsize
        self.lsize = None
        self.g_feature_ids_map_lock = None
        self.g_feature_ids_map_write_lock = None
        self.gpus_num = gpus_num
        self.dims = dims
        self.cache_alg = cache_alg
        self.rank = 0
        self.g_feature_pool = None
        self.g_feature_ids_map = None
        self.barrier = barrier
        self.l_feature_pool = None
        self.l_feature_ids_map = None
        self.l_feature_ids_map_keys = None
        self.l_feature_ids_map_values = None
        self.device = None
        self.cache_queue = None
        self.cache_queue_exit = False
        self.cache_process = None
        self.pool = None
        self.pool_size = 1
        self.g_mgr_queue = None
        self.cache_comm_queue = None
        self.g_feature_ids_map_keys = None
        self.g_feature_ids_map_values = None
        self.g_feature_ids_map_cache = {}
        self.feat_type = feat_type
        self.vl_pm_pool = None
        self.vl_pm_size = 900000
        self.vl_pm_stream = None
        self.thread_pool = None
        self.cache_streams = None
        self.cupy_stream = None
        self.cache_g = None
        self.cache_l = None

    def init_local_sources(self, g_feature_pool, g_feature_ids_map, g_feature_ids_map_keys, g_feature_ids_map_values,
                           lsize, l_feature_pool, l_feature_ids_map, global_id_counter,
                           g_feature_ids_map_lock, g_feature_ids_map_write_lock, cache_comm_queue, g_mgr_queue, device, rank):
        self.g_feature_ids_map_lock = g_feature_ids_map_lock
        self.g_feature_ids_map_write_lock = g_feature_ids_map_write_lock
        self.g_feature_ids_map_keys = g_feature_ids_map_keys
        self.g_feature_ids_map_values = g_feature_ids_map_values
        self.g_feature_ids_map = g_feature_ids_map
        self.g_feature_ids_map_cache = copy.deepcopy(g_feature_ids_map)
        self.lsize = lsize
        self.l_feature_pool = l_feature_pool
        self.l_feature_ids_map = l_feature_ids_map
        self.g_feature_pool = g_feature_pool
        self.device = device
        self.rank = rank
        self.l_feature_ids_map_keys = {}
        self.l_feature_ids_map_values = {}
        for name in self.l_feature_ids_map.keys():
            self.l_feature_ids_map_keys[name] = np.array([], dtype=np.int32)
            self.l_feature_ids_map_values[name] = np.array([], dtype=np.int32)
        self.vl_pm_pool = None
        self.vl_pm_stream = torch.cuda.Stream(device=self.device)
        self.vl_pm_streams = [torch.cuda.Stream(device=self.device) for _ in range(4)]
        self.cupy_stream = cp.cuda.stream.Stream()
        self.thread_pool = ThreadPool(processes=4)
        self.global_id_counter = global_id_counter
        self.g_mgr_queue = g_mgr_queue
        self.cache_comm_queue = cache_comm_queue
        self.cache_queue = queue.Queue()
        self.cache_streams = [torch.cuda.Stream(device=self.device) for _ in range(2)]
        threading.Thread(target=self.cache_worker, daemon=True).start()
        # threading.Thread(target=self.prefetch_worker, daemon=True).start()
        # ThreadPool(processes=2).apply_async(self.prefetch_worker)
        # if self.rank==0: ThreadPool(processes=2).apply_async(self.gcache_worker)
        if self.rank==self.gpus_num-1: threading.Thread(target=self.gcache_worker, daemon=True).start()
        if self.cache_alg != CACHEALG.JACA:
            self.cache_g = create_cache(self.cache_alg, self.gsize, singleton=False)
            self.cache_l = create_cache(self.cache_alg, self.lsize, singleton=False)

    def init_vl_pool(self, vl_pm_size=None):
        self.vl_pm_pool = {
            "forward0": torch.zeros(size=(vl_pm_size or self.vl_pm_size, self.dims[0]), dtype=self.feat_type).pin_memory(),
            "forward1": torch.zeros(size=(vl_pm_size or self.vl_pm_size, self.dims[1]), dtype=self.feat_type).pin_memory(),
            "forward2": torch.zeros(size=(vl_pm_size or self.vl_pm_size, self.dims[2]), dtype=self.feat_type).pin_memory(),
            "backward2": torch.zeros(size=(vl_pm_size or self.vl_pm_size, self.dims[2]), dtype=self.feat_type).pin_memory(),
            "backward1": torch.zeros(size=(vl_pm_size or self.vl_pm_size, self.dims[1]), dtype=self.feat_type).pin_memory(),
        }

    def init_local_cache(self):
        feature_ids, all_indices_np = self.get_g_feature_ids_map_kv(name='forward0')
        features = self.g_feature_pool["forward0"][1][all_indices_np]
        cap = self.l_feature_pool["forward0"].size(0)
        self.add_halo_features_to_local_ext_worker('forward0', feature_ids[:cap], features[:cap])

    def get_g_feature_ids_map_kv(self, name):
        # with self.g_feature_ids_map_lock:
        #     k, v = self.g_feature_ids_map_keys[name], self.g_feature_ids_map_values[name]
        # k, v = self.g_feature_ids_map_keys[name], self.g_feature_ids_map_values[name]
        # --------------- #
        k, v = np.frombuffer(self.g_feature_ids_map_keys[name], dtype=np.int32), np.frombuffer(self.g_feature_ids_map_values[name], dtype=np.int32)
        # k = filter_values(k, -1)
        # k = k[k != -1]
        mask = k != -1
        k, v = k[mask], v[mask]
        return k, v

    def update_g_feature_ids_map(self, name, add_keys_np, start_index, end_index):
        values = np.arange(start_index, end_index, dtype=np.int32)
        np.frombuffer(self.g_feature_ids_map_keys[name], dtype=np.int32)[values] = add_keys_np
        np.frombuffer(self.g_feature_ids_map_values[name], dtype=np.int32)[values] = values

    def prefetch_worker(self):
        logger.info('prefetch worker working...')
        cq = self.cache_comm_queue[self.rank]
        while not self.cache_queue_exit:
            cq_name, cq_feature_ids, cq_device = cq.get()
            if len(cq_feature_ids) == 0: continue
            self.get_halo_feature_from_global2(cq_name, cq_feature_ids, device=cq_device, add_local=True)

    def cache_worker(self):
        logger.info('cache worker working...')
        # try:
        while not self.cache_queue_exit:
            task = self.cache_queue.get()
            dir, name, feature_ids, features, device = task
            feature_len = 0
            if len(feature_ids) == 0: continue
            if dir == 'to_l':
                if device: features = features.to(device, non_blocking=True)
                feature_len = self.add_halo_features_to_local_ext_worker(name, feature_ids, features)
            elif dir == 'prefetch':
                self.cache_comm_queue[device.index].put_nowait((name, feature_ids, device))
                feature_len = len(feature_ids)
            elif dir == 'to_g_ext':
                feature_len = self.add_halo_features_to_global_ext_worker(name, feature_ids, features)
            if feature_len: debug(f'[{self.rank}->{dir}] cache worker add {name} / {feature_len} done')
        # except Exception as e:
        #     logger.error(e)

    def gcache_worker(self):
        while not self.cache_queue_exit:
            if self.g_mgr_queue is not None: break
            time.sleep(0.1)
        logger.info('gcache worker working...')
        while not self.cache_queue_exit:
            task = self.g_mgr_queue.get()
            dir, name, feature_ids, features, device = task
            if len(feature_ids) == 0: continue
            if dir == 'to_g_ext':
                feature_len = self.add_halo_features_to_global_ext_worker(name, feature_ids, features)
                # if feature_len: logger.info(f'[{self.rank}->{dir}] gcache worker add {name} / {feature_len} done')

    def find_idle_stream(self):
        idel = self.vl_pm_streams[0]
        for stream in self.vl_pm_streams:
            if stream.query(): return stream
        return idel

    def enqueue_cache_task(self, dir, name, feature_ids, features, device=None, to_global=False):
        try:
            if to_global:
                self.vl_pm_pool_p = self.vl_pm_pool[name][:features.shape[0]]
                # stream = self.find_idle_stream()
                # with torch.cuda.stream(stream):
                self.vl_pm_pool_p.copy_(features, non_blocking=True)
                # stream.synchronize()
                # torch.cuda.current_stream().wait_stream(stream)
                self.g_mgr_queue.put_nowait((dir, name, feature_ids, self.vl_pm_pool_p, device))
            else:
                self.cache_queue.put_nowait((dir, name, feature_ids, features, device))
        except queue.Full:
            print(f"Failed to enqueue cache task for {name}, queue is full")
        except Exception as e:
            logger.error(f"Failed to enqueue cache task for {name}, {e}")

    def shutdown(self):
        self.cache_queue_exit = True
        while not self.cache_queue.empty(): self.cache_queue.get_nowait()
        torch.cuda.empty_cache()

    def barrier_wait(self):
        self.barrier.wait()

    def chunkify(self, lst, n):
        """将 NumPy 数组 lst 均匀分割为 n 个子数组"""
        avg = len(lst) // n
        remainder = len(lst) % n
        indices = np.arange(n) < remainder
        splits = np.cumsum(indices + avg)
        return np.split(lst, splits[:-1])

    def sync_all_procs(self):
        """
        同步所有进程的全局缓存和局部缓存，实现有界过时策略的固定epoch同步
        """
        debug(f"[Rank {self.rank}] 开始同步所有进程的缓存...")
        # 使用barrier等待所有进程到达同步点
        self.barrier_wait()

        # 1. 收集所有进程需要同步到全局的更新
        all_updates = {}
        for name in self.l_feature_pool.keys():
            # 获取当前进程的局部缓存键值对
            local_keys = self.l_feature_ids_map_keys[name]
            local_values = self.l_feature_ids_map_values[name]
            if len(local_keys) == 0: continue
            # 转换为全局可访问的格式
            local_features = self.l_feature_pool[name][local_values]
            local_features = local_features.cpu()  # 转移到CPU以便共享
            # 构建更新字典
            updates = {local_keys[i]: local_features[i] for i in range(len(local_keys))}
            all_updates[name] = updates

        # 2. 将各进程的更新同步到全局缓存
        debug(f"[Rank {self.rank}] 同步局部缓存到全局缓存...")
        for name, updates in all_updates.items():
            self.add_halo_features_to_global(name, updates)

        # 等待所有进程完成全局更新
        self.barrier_wait()

        # 3. 从全局缓存拉取最新数据到各进程的局部缓存
        debug(f"[Rank {self.rank}] 从全局缓存拉取最新数据...")
        for name in self.l_feature_pool.keys():
            # 获取全局缓存中的所有键
            global_keys, global_indices = self.get_g_feature_ids_map_kv(name)
            if len(global_keys) == 0: continue
            global_keys_tensor = torch.from_numpy(global_keys).to(self.device)
            global_features = self.g_feature_pool[name][0][global_indices]
            global_features = global_features.to(self.device)
            # 更新局部缓存
            global_features_dict = {global_keys[i]: global_features[i] for i in range(len(global_keys))}
            self.add_halo_features_to_local(name, global_features_dict)

        debug(f"[Rank {self.rank}] 缓存同步完成")
        self.barrier_wait()  # 确保所有进程完成同步

    # 根据id查找是否在cache中，无需返回feature
    def is_feature_in_local_cache(self, name, feature_ids):
        if self.cache_alg != CACHEALG.JACA:
            return self.cache_l.is_keys_exist(feature_ids.numpy(), namespace=name)
        return isin_cupy(feature_ids.numpy(), self.l_feature_ids_map_keys[name], stream=self.cupy_stream)

    # 根据id查找是否在cache中，无需返回feature
    def is_feature_in_cache(self, name, feature_ids, features=None, target_rank=None):
        local_keys_np = self.l_feature_ids_map_keys[name]
        global_keys_np, _ = self.get_g_feature_ids_map_kv(name)
        feature_ids_np = feature_ids.numpy()

        if use_real_cache:
            global_found_mask = []
            if self.cache_alg == CACHEALG.JACA:
                local_found_mask = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
                # 如果局部缓存有缺失，再检查全局缓存
                if np.count_nonzero(~local_found_mask):
                    global_ids_np = feature_ids_np[~local_found_mask]
                    # feature really in cpu
                    global_found_mask = isin_cupy(global_ids_np, global_keys_np, stream=self.cupy_stream)
            else:
                local_found_mask = self.cache_l.is_keys_exist(feature_ids_np, namespace=name)
                if np.count_nonzero(~local_found_mask):
                    global_ids_np = feature_ids_np[~local_found_mask]
                    global_found_mask = self.cache_g.is_keys_exist(global_ids_np, namespace=name)
            # prefetch, cpu to gpu
            # if np.count_nonzero(global_found_mask): self.enqueue_cache_task('prefetch', name, global_ids_np[global_found_mask], features=None, device=torch.device('cuda', target_rank))
            local_found_mask[~local_found_mask] = global_found_mask
        else:
            # -------------------------------- #
            # 检查全局缓存
            global_ids_np = feature_ids_np
            # feature really in cpu
            global_found_mask = isin_cupy(global_ids_np, global_keys_np, stream=self.cupy_stream)
            # prefetch, cpu to gpu
            # if np.count_nonzero(global_found_mask): self.enqueue_cache_task('prefetch', name, global_ids_np[global_found_mask], features=None, device=torch.device('cuda', target_rank))
            local_found_mask = global_found_mask
            # -------------------------------- #

        # add to global cache
        if np.count_nonzero(~local_found_mask):
            self.thread_pool.apply_async(self.enqueue_cache_task, args=('to_g_ext', name, feature_ids_np[~local_found_mask], features[~local_found_mask], None, True))
        return local_found_mask

    @DeprecationWarning
    def get_halo_feature(self, name, feature_ids, device, out=None):
        # if feature_ids.numel() == 0: return torch.empty((0, self.dims[int(name[-1])]), device=device)
        if feature_ids.numel() == 0: return None
        l_feature_pool = self.l_feature_pool[name]
        local_keys_np = self.l_feature_ids_map_keys[name]
        local_ids_np = self.l_feature_ids_map_values[name]
        feature_ids_np = np.unique(feature_ids.cpu().numpy())
        with torch.cuda.stream(self.cache_streams[0]):
            result = torch.empty((feature_ids_np.size, self.dims[int(name[-1])]), device=device, dtype=self.feat_type)

        # 找到哪些 feature_ids 存在于 local_keys 中
        found_local_mask_np = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
        found_local_indices_np = np.where(found_local_mask_np)[0]
        missing_feature_ids_np = feature_ids_np[~found_local_mask_np]

        # 提取局部缓存中的特征
        if found_local_indices_np.size > 0:
            valid_local_indices = local_ids_np[isin_cupy(local_keys_np, feature_ids_np[found_local_indices_np], stream=self.cupy_stream)]
            with torch.cuda.stream(self.cache_streams[1]):
                local_features = l_feature_pool[valid_local_indices]
                # gpu to gpu
                result[found_local_indices_np].copy_(local_features, non_blocking=True)

        # 如果局部缓存未找到特征，从全局缓存中获取
        if missing_feature_ids_np.size > 0:
            # engine.ctx.timer.start(f'cccc')
            valid_global_ids, valid_global_indices = self.get_halo_feature_from_global(name, missing_feature_ids_np)
            # engine.ctx.timer.stop(f'cccc')
            if valid_global_indices is not None:
                # cpu pinned memory to gpu
                with torch.cuda.stream(self.cache_streams[0]):
                    # engine.ctx.timer.start(f'aaaa1')
                    global_features = self.g_feature_pool[name][0].index_select(0, valid_global_indices)
                    # engine.ctx.timer.stop(f'aaaa1')
                    # engine.ctx.timer.start(f'aaaa2')
                    # global_features = self.g_feature_pool[name][1][valid_global_indices]  # .contiguous().pin_memory()
                    # engine.ctx.timer.stop(f'aaaa2')
                self.add_halo_features_to_local_ext(name, valid_global_ids, global_features, device)
                if found_local_indices_np.size == 0:
                    with torch.cuda.stream(self.cache_streams[1]):
                        return global_features.to(device=device, non_blocking=True)
                # engine.ctx.timer.start(f'ffff')
                result[~found_local_mask_np].copy_(global_features, non_blocking=True)
                # engine.ctx.timer.stop(f'ffff')
        return result

    def get_halo_feature_from_local(self, name, feature_ids):
        if feature_ids.numel() == 0: return None, None
        l_feature_pool = self.l_feature_pool[name]
        local_keys_np = self.l_feature_ids_map_keys[name]
        local_ids_np = self.l_feature_ids_map_values[name]
        feature_ids_np = np.unique(feature_ids.cpu().numpy())
        # 找到哪些 feature_ids 存在于 local_keys 中
        if self.cache_alg == CACHEALG.JACA:
            found_local_mask_np = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
            found_local_indices_np = np.where(found_local_mask_np)[0]
            # 提取局部缓存中的特征
            if found_local_indices_np.size == 0: return None, None
            valid_local_indices = local_ids_np[isin_cupy(local_keys_np, feature_ids_np[found_local_indices_np], stream=self.cupy_stream)]
            return l_feature_pool[valid_local_indices], found_local_indices_np
        else:
            return self.cache_l.get_batch(feature_ids_np, namespace=name)

    
    def get_halo_feature_from_global2(self, name, feature_ids, device='cpu', add_local=False):
        if isinstance(feature_ids, torch.Tensor):
            if feature_ids.numel() == 0: return None, None
        elif isinstance(feature_ids, np.ndarray):
            if feature_ids.size == 0: return None, None
        if self.cache_alg == CACHEALG.JACA:
            # 获取特征池和对应的索引字典
            all_keys_np, all_indices_np = self.get_g_feature_ids_map_kv(name)
            # 使用 numpy 的 isin 来找到哪些 feature_ids 存在于 all_keys 中
            matching_mask = isin_cupy(feature_ids, all_keys_np, stream=self.cupy_stream)
            match_feature_ids = feature_ids[matching_mask]
            found_global_indices_np = np.where(matching_mask)[0]
            # 查找这些匹配的 feature_ids 在 all_keys 中的位置
            keys_mask = isin_cupy(all_keys_np, match_feature_ids, stream=self.cupy_stream)
            valid_indices_np = all_indices_np[keys_mask]
            if valid_indices_np.size == 0: return None, None
            with torch.cuda.stream(self.cache_streams[0]):
                global_features = self.g_feature_pool[name][0].index_select(0, torch.from_numpy(valid_indices_np))
            global_feature_ids = all_keys_np[valid_indices_np]
            # if match_feature_ids.shape[0] != global_features.shape[0]:
            #     print(global_features.shape, feature_ids.shape, match_feature_ids.shape)
        else:
            global_features, found_global_indices_np = self.cache_g.get_batch(feature_ids, namespace=name)
            global_feature_ids = feature_ids[found_global_indices_np]
        if add_local and global_features is not None: self.add_halo_features_to_local_ext(name, global_feature_ids, global_features, device)
        return global_features, found_global_indices_np

    @DeprecationWarning
    def get_halo_feature_from_global(self, name, feature_ids, device='cpu'):
        # 获取特征池和对应的索引字典
        all_keys_np, all_indices_np = self.get_g_feature_ids_map_kv(name)
        # 使用 numpy 的 isin 来找到哪些 feature_ids 存在于 all_keys 中
        matching_mask = isin_cupy(feature_ids, all_keys_np, stream=self.cupy_stream)
        # 查找这些匹配的 feature_ids 在 all_keys 中的位置
        keys_mask = isin_cupy(all_keys_np, feature_ids[matching_mask], stream=self.cupy_stream)
        valid_indices_np = all_indices_np[keys_mask]
        # valid_indices_np = find_valid_indices(feature_ids, all_keys_np, all_indices_np)
        if valid_indices_np.size == 0: return None, None
        return feature_ids[matching_mask], torch.from_numpy(valid_indices_np).to(device)

    def add_halo_features_to_local(self, name, node_features):
        # 将字典的键转换为 numpy 数组
        add_keys_np = np.fromiter(node_features.keys(), dtype=np.int32)
        if self.cache_alg != CACHEALG.JACA:
            feats = torch.stack(list(node_features.values()), dim=0)
            self.cache_l.put_batch(add_keys_np, feats, namespace=name)
            return
        ## else:
        if name not in self.l_feature_pool: raise ValueError(f"Local feature pool {name} does not exist")
        # 获取局部特征池和对应的索引字典
        feature_pool = self.l_feature_pool[name]
        l_feature_ids_map = self.l_feature_ids_map[name]
        existing_keys_np = self.l_feature_ids_map_keys[name]
        # 查找哪些待添加的键已经存在
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)


        existing_mask = torch.from_numpy(existing_mask_np)
        # 筛选出不在 existing_keys 中的新键
        add_keys_np = add_keys_np[~existing_mask_np]
        # 如果没有需要添加的新特征
        if add_keys_np.size == 0: return

        # 筛选需要添加的特征
        features_to_add = torch.stack(list(node_features.values()), dim=0)[~existing_mask]

        # 查找空闲位置的起始位置
        empty_rows_mask = torch.all(feature_pool == 0, dim=1)
        empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
        if empty_rows_indices.size(0) == 0: return
        start_index = empty_rows_indices[0].item()

        # 如果剩余空间不够，只缓存一部分
        available_space = feature_pool.size(0) - start_index
        if available_space < len(features_to_add):
            debug(f"Local feature pool {name} does not have enough space to add all features, "
                  f"only caching {available_space} out of {len(features_to_add)} features")
            features_to_add = features_to_add[:available_space]
            add_keys_np = add_keys_np[:available_space]

        # 使用切片进行批量传输
        end_index = start_index + len(features_to_add)
        feature_pool[start_index:end_index].copy_(features_to_add.cpu(), non_blocking=True)

        # 更新索引，确保每个name有自己的索引
        l_feature_ids_map.update(zip(add_keys_np, np.arange(start_index, end_index)))
        self.l_feature_ids_map_keys[name] = np.fromiter(l_feature_ids_map.keys(), dtype=np.int32) 
        self.l_feature_ids_map_values[name] = np.fromiter(l_feature_ids_map.values(), dtype=np.int32)

    def add_halo_features_to_local_ext(self, name, feature_ids, features, device=None):
        self.enqueue_cache_task('to_l', name, feature_ids, features, device, to_global=False)

    def add_halo_features_to_local_ext_worker(self, name, feature_ids, features):
        feature_ids, unique_indices = torch.unique(feature_ids, return_inverse=True) if isinstance(feature_ids, torch.Tensor) else np.unique(feature_ids, return_index=True)
        features = features[unique_indices]
        add_keys_np = feature_ids.cpu().numpy() if isinstance(feature_ids, torch.Tensor) else feature_ids

        if self.cache_alg != CACHEALG.JACA:
            self.cache_l.put_batch(add_keys_np, features, namespace=name)
            return
        ## else:
        feature_len = 0
        l_feature_pool = self.l_feature_pool[name]
        l_feature_ids_map = self.l_feature_ids_map[name]
        existing_keys_np = self.l_feature_ids_map_keys[name]

        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)
        existing_indices = np.nonzero(existing_mask_np)[0]
        update_keys_np = add_keys_np[existing_indices]
        add_keys_np = add_keys_np[~existing_mask_np]

        # 更新已有的 feature_idsures
        if update_keys_np.size > 0:
            existing_keys_mask = isin_cupy(existing_keys_np, update_keys_np)
            existing_keys_indices = np.nonzero(existing_keys_mask)[0]
            l_feature_pool[existing_keys_indices].copy_(features[existing_indices], non_blocking=True)
            feature_len += existing_indices.size

        # 根据权重排序，插入到可用空间中
        # with TimerRecord(prefix='aaa'):
        if use_weight_update:
            node_weight_keys = self.global_id_counter[0].numpy()
            node_weight_values = self.global_id_counter[1].numpy()
            add_keys_weight_keys_mask = isin_cupy(node_weight_keys, add_keys_np)
            add_keys_weight_keys_indices = np.nonzero(add_keys_weight_keys_mask)[0]
            add_keys_weight_values = node_weight_values[add_keys_weight_keys_indices]
            sorted_add_keys_weight_values_indices = np.argsort(add_keys_weight_values)
            # 降序排序
            if use_descending: sorted_add_keys_weight_values_indices = sorted_add_keys_weight_values_indices[::-1]
            add_keys_np = add_keys_np[sorted_add_keys_weight_values_indices]
            # features = features[sorted_add_keys_weight_values_indices]

        # 新增新的 features
        if add_keys_np.size > 0:
            features = features[~existing_mask_np]
            if self.cache_alg==CACHEALG.JACA:
                empty_rows_mask = torch.all(l_feature_pool == 0, dim=1)
                empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
                if empty_rows_indices.size(0) == 0: return 0
                start_index = empty_rows_indices[0].item()
                available_space = l_feature_pool.size(0) - start_index
                if available_space < features.size(0):
                    features = features[:available_space]
                    add_keys_np = add_keys_np[:available_space]
                end_index = start_index + features.size(0)
            # gpu to gpu
            l_feature_pool[start_index:end_index].copy_(features, non_blocking=True)
            l_feature_ids_map.update(zip(add_keys_np, np.arange(start_index, end_index)))
            self.l_feature_ids_map_keys[name] = np.fromiter(l_feature_ids_map.keys(), dtype=np.int32)
            self.l_feature_ids_map_values[name] = np.fromiter(l_feature_ids_map.values(), dtype=np.int32)
            feature_len += features.size(0)
        return feature_len

    def add_halo_features_to_global(self, name, node_features):
        # 构建待添加键的 numpy 数组
        add_keys_np = np.fromiter(node_features.keys(), dtype=np.int32)
        if self.cache_alg != CACHEALG.JACA:
            feats = torch.stack(list(node_features.values()), dim=0)
            self.cache_g.put_batch(add_keys_np, feats, namespace=name)
            return
        ## else:
        # 获取特征池和索引
        g_feature_pool = self.g_feature_pool[name][0]
        existing_keys_np, _ = self.get_g_feature_ids_map_kv(name)

        # 查找已存在的特征
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)
        existing_mask = torch.from_numpy(existing_mask_np)
        add_keys_np = add_keys_np[~existing_mask_np]
        if add_keys_np.size == 0: return
        # 筛选需要添加的特征
        features_to_add = torch.stack(list(node_features.values()), dim=0)[~existing_mask]
        # 查找空闲位置的起始位置
        start_index = torch.nonzero(g_feature_pool[:, 0] == 0).view(-1).min().item()
        # 如果剩余空间不够，只缓存一部分
        available_space = g_feature_pool.size(0) - start_index
        if available_space < len(features_to_add):
            debug(f"Feature pool {name} only caching {available_space} out of {len(features_to_add)} features")
            features_to_add = features_to_add[:available_space]
            add_keys_np = add_keys_np[:available_space]
        end_index = start_index + features_to_add.size(0)
        # gpu to pin
        self.g_feature_pool[name][1][start_index:end_index].copy_(features_to_add, non_blocking=True)
        # pin to share
        self.g_feature_pool[name][0][start_index:end_index].copy_(self.g_feature_pool[name][1][start_index:end_index], non_blocking=True)
        self.update_g_feature_ids_map(name, add_keys_np, start_index, end_index)
        return features_to_add.size(0)

    def add_halo_features_to_global_ext(self, name, feature_ids, features):
        self.enqueue_cache_task('to_g_ext', name, feature_ids, features, None, to_global=False)

    def add_halo_features_to_global_ext_worker(self, name, feature_ids, features):
        add_keys_np = feature_ids.numpy() if isinstance(feature_ids, torch.Tensor) else feature_ids
        add_keys_np, unique_indices = np.unique(add_keys_np, return_index=True)
        try:
            features = features[unique_indices]
        except Exception as e:
            print(e)
            return 0

        if self.cache_alg != CACHEALG.JACA:
            self.cache_g.put_batch(add_keys_np, features, namespace=name)
            return len(add_keys_np)

        feature_len = 0
        g_feature_pool = self.g_feature_pool[name][0]
        existing_keys_np, _ = self.get_g_feature_ids_map_kv(name)
        # 查找已存在和新增的 keys
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)
        existing_indices = np.nonzero(existing_mask_np)[0]
        update_keys_np = add_keys_np[existing_indices]
        add_keys_np = add_keys_np[~existing_mask_np]

        # 更新已有的 features
        if update_keys_np.size > 0:
            existing_keys_mask = isin_cupy(existing_keys_np, update_keys_np)
            existing_keys_indices = np.nonzero(existing_keys_mask)[0]
            self.g_feature_pool[name][1][:existing_keys_indices.size].copy_(features[existing_indices], non_blocking=True)
            self.g_feature_pool[name][0][existing_keys_indices].copy_(self.g_feature_pool[name][1][:existing_keys_indices.size], non_blocking=True)
            feature_len += existing_indices.size

        # 新增新的 features
        if add_keys_np.size > 0:
            features = features[~existing_mask_np]
            empty_rows_mask = ~torch.any(g_feature_pool != 0, dim=1)
            empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
            if empty_rows_indices.size(0) == 0: return feature_len
            start_index = empty_rows_indices[0].item()
            available_space = g_feature_pool.size(0) - start_index
    
            # 根据权重排序，插入到可用空间中
            if use_weight_update:
                node_weight_keys = self.global_id_counter[0].numpy()
                node_weight_values = self.global_id_counter[1].numpy()
                add_keys_weight_keys_mask = isin_cupy(node_weight_keys, add_keys_np)
                add_keys_weight_keys_indices = np.nonzero(add_keys_weight_keys_mask)[0]
                add_keys_weight_values = node_weight_values[add_keys_weight_keys_indices]
                sorted_add_keys_weight_values_indices = np.argsort(add_keys_weight_values)
                # 降序排序
                if use_descending: sorted_add_keys_weight_values_indices = sorted_add_keys_weight_values_indices[::-1]
                add_keys_np = add_keys_np[sorted_add_keys_weight_values_indices]

            if available_space < features.size(0):
                # 缓存替换：根据节点权重，用权重高的替换现有的得分低的。
                debug(f"Feature pool {name} only caching {available_space} out of {features.size(0)} features")
                features = features[:available_space]
                add_keys_np = add_keys_np[:available_space]

            end_index = start_index + features.size(0)
            # gpu to cpu pin
            self.update_g_feature_ids_map(name, add_keys_np, start_index, end_index)
            # self.barrier_wait()
            self.g_feature_pool[name][1][start_index:end_index].copy_(features, non_blocking=False)
            # cpu pin to cpu shared
            self.g_feature_pool[name][0][start_index:end_index].copy_(self.g_feature_pool[name][1][start_index:end_index], non_blocking=True)

            feature_len += features.size(0)


        
        # if add_keys_np.size > 0:
        #     features = features[~existing_mask_np]
        #     # if update_keys_np.size>0: features = features[~existing_keys_mask]
        #     empty_rows_mask = ~torch.any(g_feature_pool != 0, dim=1)
        #     empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
        #     if empty_rows_indices.size(0) == 0: return feature_len
        #     start_index = empty_rows_indices[0].item()
        #
        #     available_space = g_feature_pool.size(0) - start_index
        #     if available_space < features.size(0):
        #         # 缓存替换：根据节点权重，用权重高的替换现有的得分低的。
        #         debug(f"Feature pool {name} only caching {available_space} out of {features.size(0)} features")
        #         features = features[:available_space]
        #         add_keys_np = add_keys_np[:available_space]
        #
        #     end_index = start_index + features.size(0)
        #     # gpu to cpu pin
        #     self.g_feature_pool[name][1][start_index:end_index].copy_(features, non_blocking=True)
        #     g_feature_ids_map.update(zip(add_keys_np, np.arange(start_index, end_index)))
        #     self.g_feature_ids_map_keys[name] = np.fromiter(g_feature_ids_map.keys(), dtype=np.int32)
        #     self.g_feature_ids_map_values[name] = np.fromiter(g_feature_ids_map.values(), dtype=np.int32)
        #     # cpu pin to cpu shared
        #     self.g_feature_pool[name][0][start_index:end_index].copy_(self.g_feature_pool[name][1][start_index:end_index], non_blocking=True)
        #     feature_len += features.size(0)
        return feature_len




class StorageServer:
    def __init__(self, manager, gpus_num, gsize, dims, cache_alg=CACHEALG.JACA):
        self.feat_type = torch.float32
        self.rank = None
        # 创建一个Barrier以确保所有进程同步
        self.cache_barrier = manager.Barrier(gpus_num)
        self.lsize = None
        self.gsize = gsize
        self.dims = dims
        self.cache_alg = cache_alg
        self.cache_comm_queue = [manager.Queue() for _ in range(gpus_num)]
        
        # 初始化全局特征池
        self.g_feature_pool = {
            "forward0": [torch.zeros(size=(gsize, dims[0]), dtype=self.feat_type).share_memory_(), None],
            "forward1": [torch.zeros(size=(gsize, dims[1]), dtype=self.feat_type).share_memory_(), None],
            "forward2": [torch.zeros(size=(gsize, dims[2]), dtype=self.feat_type).share_memory_(), None],
            "backward2": [torch.zeros(size=(gsize, dims[2]), dtype=self.feat_type).share_memory_(), None],
            "backward1": [torch.zeros(size=(gsize, dims[1]), dtype=self.feat_type).share_memory_(), None],
        }
        # 节点全局id <=> 全局特征池位置索引
        self.g_feature_ids_map = {
            "forward0": manager.dict(),
            "forward1": manager.dict(),
            "forward2": manager.dict(),
            "backward2": manager.dict(),
            "backward1": manager.dict(),
        }
        self.g_feature_ids_map_keys = {}
        self.g_feature_ids_map_values = {}
        for name in self.g_feature_ids_map.keys():
            # self.g_feature_ids_map_keys[name] = np.zeros(gsize, dtype=np.int32)
            # self.g_feature_ids_map_values[name] = np.zeros(gsize dtype=np.int32)
            # --------------- #
            self.g_feature_ids_map_keys[name] = RawArray('i', gsize)
            self.g_feature_ids_map_values[name] = RawArray('i', gsize)
            np.frombuffer(self.g_feature_ids_map_keys[name], dtype=np.int32)[:] = -1
            np.frombuffer(self.g_feature_ids_map_values[name], dtype=np.int32)[:] = -1

        self.l_feature_pool = None
        self.l_feature_ids_map = None
        self.g_feature_ids_map_lock = manager.RLock()
        self.g_feature_ids_map_write_lock = manager.Lock()
        # 全局ID的计数器
        self.global_id_counter = manager.dict()
        self.g_mgr_queue = manager.Queue()
        # self.g_mgr_queue = multiprocessing.Queue()
        self.cache_server = CacheServer(gpus_num, self.dims, self.cache_barrier, self.feat_type, self.gsize, self.cache_alg)

    def init_local_sources(self, lsize, device, rank):
        self.rank = rank
        self.lsize = lsize
        for key in self.g_feature_pool.keys():
            self.g_feature_pool[key][1] = self.g_feature_pool[key][0].pin_memory()
        self.l_feature_pool = {
            "forward0": torch.zeros(size=(self.lsize, self.dims[0]), device=device, dtype=self.feat_type),
            "forward1": torch.zeros(size=(self.lsize, self.dims[1]), device=device, dtype=self.feat_type),
            "forward2": torch.zeros(size=(self.lsize, self.dims[2]), device=device, dtype=self.feat_type),
            "backward2": torch.zeros(size=(self.lsize, self.dims[2]), device=device, dtype=self.feat_type),
            "backward1": torch.zeros(size=(self.lsize, self.dims[1]), device=device, dtype=self.feat_type),
        }
        self.l_feature_ids_map = {
            "forward0": {},
            "forward1": {},
            "forward2": {},
            "backward2": {},
            "backward1": {},
        }
        self.cache_server.init_local_sources(self.g_feature_pool, self.g_feature_ids_map,
                                             self.g_feature_ids_map_keys, self.g_feature_ids_map_values,
                                             self.lsize, self.l_feature_pool, self.l_feature_ids_map, self.global_id_counter,
                                             self.g_feature_ids_map_lock, self.g_feature_ids_map_write_lock,
                                             self.cache_comm_queue, self.g_mgr_queue, device=device, rank=self.rank)

    def clear_cache(self, rank):
        for layer in ['forward0', 'forward1', 'forward2', 'backward2', 'backward1']:
            self.l_feature_pool[layer].zero_()
            self.l_feature_ids_map[layer].clear()
            if rank == 0:
                self.g_feature_pool[layer][0].zero_()
                self.g_feature_ids_map[layer].clear()
                if self.g_feature_pool[layer][1] is not None: self.g_feature_pool[layer][1].zero_()
                self.g_feature_ids_map_keys[layer] = np.array([], dtype=np.int32)
                self.g_feature_ids_map_values[layer] = np.array([], dtype=np.int32)

    def shutdown(self):
        self.cache_server.shutdown()

    def is_feature_in_cache(self, name, feature_ids, features=None, target_rank=None):
        return self.cache_server.is_feature_in_cache(name, feature_ids, features=features, target_rank=target_rank)

    def remain_local_cache_size(self, name):
        return self.lsize - self.cache_server.l_feature_ids_map_keys[name].size

    def get_halo_feature(self, name, feature_ids, device, out=None):
        return self.cache_server.get_halo_feature(name, feature_ids, device, out)

    def get_halo_feature_from_global(self, name, feature_ids):
        return self.cache_server.get_halo_feature_from_global(name, feature_ids)

    def add_halo_features_to_local(self, name, node_features):
        self.cache_server.add_halo_features_to_local(name, node_features)

    def add_halo_features_to_local_ext(self, name, feature_ids, features):
        self.cache_server.add_halo_features_to_local_ext(name, feature_ids, features)

    def add_halo_features_to_global_ext(self, name, feature_ids, features):
        self.cache_server.add_halo_features_to_global_ext(name, feature_ids, features)

    def add_halo_features_to_global(self, name, node_features):
        self.cache_server.add_halo_features_to_global(name, node_features)


    def extract_max_halo_features(self, part_size, dataset, part_dir='data/part_data', k=-1, our_partition=False):
        all_part_nodes_feats = []
        halo_node_features = {}
        for part_id in range(part_size):
            part_config = f'/home/sxf/Desktop/gnn/dist_gnn_fullbatch/data/part_data/flick/{part_size}part/flick.json'
            g, part_nodes_feats, efeats, _, _, node_type, _ = dgl.distributed.load_partition(part_config, part_id)
            node_type = node_type[0]
            part_nodes_feats[dgl.NID] = g.ndata[dgl.NID]
            part_nodes_feats['part_id'] = g.ndata['part_id']
            part_nodes_feats['inner_node'] = g.ndata['inner_node'].bool()
            part_nodes_feats['feat'] = part_nodes_feats[node_type + '/feat']
            all_part_nodes_feats.append(part_nodes_feats)

        for part_id in range(part_size):
            nodes_feats = all_part_nodes_feats[part_id]
            for i in range(part_size):
                if i == part_id: continue
                halo_local_masks = (nodes_feats['part_id'] == i)
                halo_local_idx = torch.nonzero(halo_local_masks).view(-1)
                if halo_local_idx.numel() == 0: continue
                halo_global_ids = nodes_feats[dgl.NID][halo_local_idx]
                part_i_nodes_feats = all_part_nodes_feats[i]
                halo_remote_mask = torch.isin(part_i_nodes_feats[dgl.NID], halo_global_ids)
                halo_remote_idx = torch.nonzero(halo_remote_mask).view(-1)
                features = part_i_nodes_feats['feat'][halo_remote_idx]
                temp = {}
                for global_id, feature in zip(halo_global_ids, features):
                    if global_id.item() in halo_node_features.keys(): continue
                    temp[global_id.item()] = feature
                halo_node_features.update(temp)
                logger.info(f"[{part_id}-{i}]本次添加{len(temp.keys())}/{len(halo_global_ids)}个halo节点进来")
        logger.info(f"最终有效的节点数：{len(halo_node_features.keys())}")
        return halo_node_features

    def get_halo_count(self, part_size, dataset, gpus_list, part_dir='data/part_data', our_partition=False):
        # /home/sxf/Desktop/gnn/dist_gnn_fullbatch
        partition_file = f'{part_dir}/{dataset}/{part_size}part/{dataset}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl'
        with open(partition_file, 'rb') as f: assig_graphs_gpus = pickle.load(f)
        g_infos = list(assig_graphs_gpus.values())
        temp = Counter()
        for part_id in range(part_size):
            g_info = g_infos[part_id]
            g = g_info.graph
            node_feats = g_info.node_feats
            halo_nodes = g.nodes()[~node_feats['inner_node'].bool()].numpy()
            halo_global_ids = node_feats[dgl.NID][halo_nodes].numpy()
            temp.update(halo_global_ids)
        self.global_id_counter[0] = torch.asarray(list(temp.keys()))
        self.global_id_counter[1] = torch.asarray(list(temp.values()), dtype=torch.float32)
        return self.global_id_counter

    def extract_halo_features_by_score(self, part_size, dataset, gpus_list, part_dir='data/part_data', k=-1, our_partition=False):
        max_subg_size = 0
        all_part_nodes_feats = []
        halo_node_features = {}
        model_type = DistGNNType.DistGCN
        # 加载pkl文件  /home/sxf/Desktop/gnn/dist_gnn_fullbatch
        partition_file = f'{part_dir}/{dataset}/{part_size}part/{dataset}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl'
        with open(partition_file, 'rb') as f: assig_graphs_gpus = pickle.load(f)

        g_infos = list(assig_graphs_gpus.values())
        for part_id in range(part_size):
            g_info = g_infos[part_id]
            g = g_info.graph
            part_nodes_feats = g_info.node_feats
            all_part_nodes_feats.append((g, part_nodes_feats))

        for part_id in range(part_size):
            g, nodes_feats = all_part_nodes_feats[part_id]
            for i in range(part_size):
                if i == part_id: continue
                halo_local_masks = (nodes_feats['part_id'] == i)
                halo_local_idx = torch.nonzero(halo_local_masks).view(-1)
                if halo_local_idx.numel() == 0: continue
                halo_global_ids = nodes_feats[dgl.NID][halo_local_idx]


                # method 1:
                # agg_scores = _get_agg_scores(g, halo_local_masks, nodes_feats, model_type)
                # _, ori_idxs = torch.sort(agg_scores[0] + agg_scores[1], descending=True)

                # method 2:
                indices = torch.nonzero(torch.isin(torch.tensor(self.global_id_counter[0]), halo_global_ids)).view(-1)
                _, ori_idxs = torch.sort(torch.tensor(self.global_id_counter[1])[indices], descending=use_descending)  # 降序排序

                # method 3:
                # agg_scores = _get_agg_scores(g, halo_local_masks, nodes_feats, model_type)
                # indices = np.nonzero(np.isin(self.global_id_counter[0].numpy(), halo_global_ids.numpy()))[0]
                # scores = (agg_scores[0] + agg_scores[1]) * self.global_id_counter[1][indices]
                # _, ori_idxs = torch.sort(scores, descending=True)
                # self.global_id_counter[1][indices] = scores


                # select top k
                top_k_ids = halo_global_ids[ori_idxs[:k] if k > 0 else ori_idxs]
                # 从分区i中批量提取halo节点的特征
                remote_g, part_i_nodes_feats = all_part_nodes_feats[i]
                halo_remote_mask = torch.isin(part_i_nodes_feats[dgl.NID], top_k_ids)
                halo_remote_idx = torch.nonzero(halo_remote_mask).view(-1)
                features = part_i_nodes_feats['feat'][halo_remote_idx]
                temp = {}
                for global_id, feature in zip(top_k_ids, features):
                    if global_id.item() in halo_node_features.keys(): continue
                    temp[global_id.item()] = feature
                halo_node_features.update(temp)
                max_subg_size = max(max_subg_size, len(halo_global_ids))
                logger.info(f"[{part_id}-{i}]本次添加{len(temp.keys())}/{len(halo_global_ids)}个halo节点进来")
        logger.info(f"最终有效的节点数：{len(halo_node_features.keys())}")
        return halo_node_features, max_subg_size

    @staticmethod
    def cal_capacity(part_size, dataset, gpus_list, f_dims, part_dir='data/part_data', k=-1, our_partition=False):
        # reserved_mem是手动设定的值，用于排除训练过程中需要占用的内存，比如梯度等。比如10MB。
        reserved_mem_gpu = 0  # 10 * 1024
        reserved_mem_cpu = 0  # 10 * 1024
        # cpu_mem是手动设定的值，表示CPU的可用内存。比如256GB。
        cpu_mem = 786 * 1024 * 1024
        # 每个GPU实际的缓存容量
        gpus_capacity = {}
        # CPU上最大的缓存容量
        max_capacity_CPU = 0
        all_part_nodes_feats = []
        partition_file = f'{part_dir}/{dataset}/{part_size}part/{dataset}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl'
        with open(partition_file, 'rb') as f: assig_graphs_gpus = pickle.load(f)

        # assig_graphs_gpus的格式：{gpu1: subgraph1, gpu2: subgraph2, ...}
        g_infos = list(assig_graphs_gpus.values())
        gpu_ids = list(assig_graphs_gpus.keys())
        for part_id in range(part_size):
            g_info = g_infos[part_id]
            g = g_info.graph
            part_nodes_feats = g_info.node_feats
            all_part_nodes_feats.append((g, part_nodes_feats))

        for part_id in range(part_size):
            g, nodes_feats = all_part_nodes_feats[part_id]
            halo_node_ids = set()
            for i in range(part_size):
                if i == part_id: continue
                # 这里是获取当前分区part_id中，有哪些halo节点是属于分区i的。表示为halo_global_ids
                halo_local_masks = (nodes_feats['part_id'] == i)
                halo_local_idx = torch.nonzero(halo_local_masks).view(-1)
                if halo_local_idx.numel() == 0: continue
                halo_global_ids = nodes_feats[dgl.NID][halo_local_idx]
                for global_id in halo_global_ids:
                    if global_id.item() in halo_node_ids: continue
                    halo_node_ids.add(global_id.item())
            # 当前分区part_id中，最大需要的缓存节点数
            max_capacity = len(halo_node_ids)
            gpu_id = gpu_ids[part_id]
            # 根据最大缓存节点数、gpu的可用内存、顶点维度等信息，来获取当前GPU实际可以承受的缓存容量
            gpu_memory = gpu.gpu_capability[gpu_id][0]  # MB
            temp = (gpu_memory - reserved_mem_gpu) * 1024 * 1024  / sum(f_dims) / 4
            gpus_capacity[part_id] = min(temp, max_capacity)
        temp = (cpu_mem - reserved_mem_cpu) * 1024 * 1024  / sum(f_dims) / 4
        cpu_capacity = min(temp, sum(gpus_capacity.values()))
        return cpu_capacity, gpus_capacity

def warmup(device):
    # 定义一个网络
    model = nn.Sequential(
        nn.Linear(8, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    inputs = torch.rand(32, 8)
    model.to(device)
    inputs = inputs.to(device)
    for _ in tqdm(range(300), desc='warmup...'):
        out = model(inputs)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()


def test_worker(rank, server):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    cp.cuda.Device(rank).use()
    device_name = torch.cuda.get_device_name(rank)
    print(f"GPU {rank}: {device_name}")
    warmup(device)
    # 初始化局部缓存
    server.init_local_sources(device)

    if rank == 0:
        keys = list(server.g_feature_ids_map['forward0'].keys())
        values = list(server.g_feature_ids_map['forward0'].values())
        node_feats = {k: v for k, v in zip(keys, server.g_feature_pool['forward0'][values])}
        server.add_halo_features_to_global('forward0', node_feats)
        # server.add_halo_features_to_local('forward0', node_feats)
        ids = np.random.choice(server.g_feature_ids_map['forward0'].keys(), size=10000, replace=False)
        ids = torch.tensor(ids, device=device)
        torch.cuda.synchronize()

        with TimerRecord(prefix=f'{rank} is_feature_in_cache'):
            server.is_feature_in_cache(name='forward0', feature_ids=ids)

        with TimerRecord(prefix=f'{rank} get {len(ids)} feature'):
            a = server.get_halo_feature(name='forward0', feature_ids=ids, device=device)

        print('done')
        time.sleep(5)

        torch.cuda.synchronize()
        with TimerRecord(prefix=f'{rank} is_feature_in_cache'):
            server.is_feature_in_cache(name='forward0', feature_ids=ids)

        with TimerRecord(prefix=f'{rank} get 10000 feature2'):
            a = server.get_halo_feature(name='forward0', feature_ids=ids, device=device)

        time.sleep(5)
        torch.cuda.synchronize()
        with TimerRecord(prefix=f'{rank} is_feature_in_cache'):
            server.is_feature_in_cache(name='forward0', feature_ids=ids)

        with TimerRecord(prefix=f'{rank} get 10000 feature3'):
            a = server.get_halo_feature(name='forward0', feature_ids=ids, device=device)

    server.clear_cache(rank)
    server.shutdown()
    return


    a = torch.rand(10000, 500).cpu().pin_memory()
    torch.cuda.synchronize()
    non_blocking = True
    for i in range(2):
        with TimerRecord(prefix=f'{i} [{rank}] cpu to gpu({rank})'):
            b = a.to(torch.device(f'cuda:{rank}'), non_blocking=non_blocking)
            torch.cuda.synchronize()

        with TimerRecord(prefix=f'{i} [{rank}] gpu({rank}) to cpu'):
            c = b.to(torch.device('cpu'), non_blocking=non_blocking)
            torch.cuda.synchronize()

        with TimerRecord(prefix=f'{i} [{rank}] cpu to gpu({0 if rank == 1 else 1})'):
            e = a.to(torch.device(f'cuda:{0 if rank == 1 else 1}'), non_blocking=non_blocking)
            torch.cuda.synchronize()

        with TimerRecord(prefix=f'{i} [{rank}] gpu({rank}) to gpu({0 if rank==1 else 1})'):
            f = b.to(torch.device(f'cuda:{0 if rank==1 else 1}'), non_blocking=non_blocking)
            torch.cuda.synchronize()


if __name__ == '__main__':
    from cspart import GraphInfo
    gpus = '5,3,1'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    gpus_num = len(gpus.split(','))
    manager = Manager()
    storage_server = StorageServer(manager, gpus_num=gpus_num, gsize=50000, lsize=5000, dims=[500, 256, 16])
    # halo_node_feats = storage_server.extract_halo_features(part_size=gpus_num, dataset='flick')
    halo_node_feats = storage_server.extract_halo_features_by_score(part_size=gpus_num, dataset='flick', k=2000)
    storage_server.add_halo_features_to_global('forward0', halo_node_feats)
    mp.spawn(test_worker, (storage_server, ), gpus_num, join=True, daemon=False)

