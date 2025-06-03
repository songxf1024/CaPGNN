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
    # Check the empty array
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
    # Initialize the matching mask
    matching_mask = np.zeros(feature_ids.size, dtype=np.bool_)
    # Create a collection for quick searches
    all_keys_set = set(all_keys_np)
    # Find out if each element in feature_ids exists in all_keys_np
    for i in range(feature_ids.size):
        if feature_ids[i] in all_keys_set:
            matching_mask[i] = True
    # Get feature_ids matching all_keys_np
    matched_feature_ids = feature_ids[matching_mask]
    # Initialization result mask
    keys_mask = np.zeros(all_keys_np.size, dtype=np.bool_)
    # Create a collection for quick searches
    matched_feature_ids_set = set(matched_feature_ids)
    # Find out if each element in all_keys_np exists in matching feature_ids
    for i in range(all_keys_np.size):
        if all_keys_np[i] in matched_feature_ids_set:
            keys_mask[i] = True
    # Use keys_mask to filter all_indices_np
    valid_indices_np = all_indices_np[keys_mask]
    return valid_indices_np

def convert_CNP(input_array, target_type, device=None):
    """
    Converts the input data type to the target type, supporting conversion between CuPy, NumPy, and PyTorch Tensor. For CuPy to PyTorch conversion, use DLPack for efficient conversion. 
    parameter:
        input_array: The input array can be a CuPy array, a NumPy array, or PyTorch Tensor 
        target_type: The target type can be 'cupy', 'numpy', or 'torch' 
    Return:
        Converted target type data
    """
    if target_type == None: return input_array
    if isinstance(input_array, cp.ndarray):
        # Convert from CuPy
        if target_type == 'numpy':
            return input_array.get()  # or cp.asnumpy(input_array)
        elif target_type == 'torch':
            # CuPy to PyTorch Using DLPack
            res = torch.utils.dlpack.from_dlpack(input_array.toDlpack())
            # return torch.from_numpy(input_array.get())
            # If the target device is None, the tensor on the default device is returned; otherwise, it will be transferred to the specified device
            if device: res = res.to(device)
            return res
        elif target_type == 'cupy':
            return input_array
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for CuPy array.")
    elif isinstance(input_array, np.ndarray):
        # Convert from NumPy
        if target_type == 'cupy':
            if device:
                with cp.cuda.Device(device[-1]):
                    res = cp.asarray(input_array)
            else:
                res = cp.asarray(input_array)
            return res
        elif target_type == 'torch':
            res = torch.from_numpy(input_array)
            # If the target device is None, the tensor on the default device is returned; otherwise, it will be transferred to the specified device
            if device: res = res.to(device)
            return res
        elif target_type == 'numpy':
            return input_array
        else:
            raise ValueError(f"Unsupported target type '{target_type}' for NumPy array.")
    elif isinstance(input_array, torch.Tensor):
        # Convert from PyTorch Tensor
        if target_type == 'cupy':
            # Make sure the tensor is on the GPU
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


# Whether to use priority sort based on node importance when replacing nodes
use_weight_update = True
# The overlap rate of nodes is sorted in descending order, which is preferred to select those with high overlap rate.
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
        """Evenly split NumPy array lst into n subarrays"""
        avg = len(lst) // n
        remainder = len(lst) % n
        indices = np.arange(n) < remainder
        splits = np.cumsum(indices + avg)
        return np.split(lst, splits[:-1])

    # Find whether it is in cache by id, no need to return feature
    def is_feature_in_local_cache(self, name, feature_ids):
        if self.cache_alg != CACHEALG.JACA:
            return self.cache_l.is_keys_exist(feature_ids.numpy(), namespace=name)
        return isin_cupy(feature_ids.numpy(), self.l_feature_ids_map_keys[name], stream=self.cupy_stream)

    # Find whether it is in cache by id, no need to return feature
    def is_feature_in_cache(self, name, feature_ids, features=None, target_rank=None):
        local_keys_np = self.l_feature_ids_map_keys[name]
        global_keys_np, _ = self.get_g_feature_ids_map_kv(name)
        feature_ids_np = feature_ids.numpy()

        global_found_mask = []
        if self.cache_alg == CACHEALG.JACA:
            local_found_mask = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
            # If the local cache is missing, check the global cache again
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

        # Find which feature_ids exist in local_keys
        found_local_mask_np = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
        found_local_indices_np = np.where(found_local_mask_np)[0]
        missing_feature_ids_np = feature_ids_np[~found_local_mask_np]

        # Extract features in local cache
        if found_local_indices_np.size > 0:
            valid_local_indices = local_ids_np[isin_cupy(local_keys_np, feature_ids_np[found_local_indices_np], stream=self.cupy_stream)]
            with torch.cuda.stream(self.cache_streams[1]):
                local_features = l_feature_pool[valid_local_indices]
                # gpu to gpu
                result[found_local_indices_np].copy_(local_features, non_blocking=True)

        # If the local cache does not find the feature, get it from the global cache
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
        # Find which feature_ids exist in local_keys
        if self.cache_alg == CACHEALG.JACA:
            found_local_mask_np = isin_cupy(feature_ids_np, local_keys_np, stream=self.cupy_stream)
            found_local_indices_np = np.where(found_local_mask_np)[0]
            # Extract features in local cache
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
            # Get feature pool and corresponding index dictionary
            all_keys_np, all_indices_np = self.get_g_feature_ids_map_kv(name)
            # Use numpy's isin to find which feature_ids exist in all_keys
            matching_mask = isin_cupy(feature_ids, all_keys_np, stream=self.cupy_stream)
            match_feature_ids = feature_ids[matching_mask]
            found_global_indices_np = np.where(matching_mask)[0]
            # Find the location of these matching feature_ids in all_keys
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
        # Get feature pool and corresponding index dictionary
        all_keys_np, all_indices_np = self.get_g_feature_ids_map_kv(name)
        # Use numpy's isin to find which feature_ids exist in all_keys
        matching_mask = isin_cupy(feature_ids, all_keys_np, stream=self.cupy_stream)
        # Find the location of these matching feature_ids in all_keys
        keys_mask = isin_cupy(all_keys_np, feature_ids[matching_mask], stream=self.cupy_stream)
        valid_indices_np = all_indices_np[keys_mask]
        # valid_indices_np = find_valid_indices(feature_ids, all_keys_np, all_indices_np)
        if valid_indices_np.size == 0: return None, None
        return feature_ids[matching_mask], torch.from_numpy(valid_indices_np).to(device)

    def add_halo_features_to_local(self, name, node_features):
        # Convert dictionary keys to numpy arrays
        add_keys_np = np.fromiter(node_features.keys(), dtype=np.int32)
        if self.cache_alg != CACHEALG.JACA:
            feats = torch.stack(list(node_features.values()), dim=0)
            self.cache_l.put_batch(add_keys_np, feats, namespace=name)
            return
        ## else:
        if name not in self.l_feature_pool: raise ValueError(f"Local feature pool {name} does not exist")
        # Get local feature pool and corresponding index dictionary
        feature_pool = self.l_feature_pool[name]
        l_feature_ids_map = self.l_feature_ids_map[name]
        existing_keys_np = self.l_feature_ids_map_keys[name]
        # Find which keys to be added already exist
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)


        existing_mask = torch.from_numpy(existing_mask_np)
        # Filter out new keys that are not in existing_keys
        add_keys_np = add_keys_np[~existing_mask_np]
        # If there are no new features to add
        if add_keys_np.size == 0: return

        # Filter the features that need to be added
        features_to_add = torch.stack(list(node_features.values()), dim=0)[~existing_mask]

        # Find the starting position of the free position
        empty_rows_mask = torch.all(feature_pool == 0, dim=1)
        empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
        if empty_rows_indices.size(0) == 0: return
        start_index = empty_rows_indices[0].item()

        # If the remaining space is insufficient, only a part of it is cached
        available_space = feature_pool.size(0) - start_index
        if available_space < len(features_to_add):
            debug(f"Local feature pool {name} does not have enough space to add all features, "
                  f"only caching {available_space} out of {len(features_to_add)} features")
            features_to_add = features_to_add[:available_space]
            add_keys_np = add_keys_np[:available_space]

        # Batch transfer using slices
        end_index = start_index + len(features_to_add)
        feature_pool[start_index:end_index].copy_(features_to_add.cpu(), non_blocking=True)

        # Update the index to make sure each name has its own index
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

        # Update existing feature_idsures
        if update_keys_np.size > 0:
            existing_keys_mask = isin_cupy(existing_keys_np, update_keys_np)
            existing_keys_indices = np.nonzero(existing_keys_mask)[0]
            l_feature_pool[existing_keys_indices].copy_(features[existing_indices], non_blocking=True)
            feature_len += existing_indices.size

        # Sort by weight, insert into available space
        # with TimerRecord(prefix='aaa'):
        if use_weight_update:
            node_weight_keys = self.global_id_counter[0].numpy()
            node_weight_values = self.global_id_counter[1].numpy()
            add_keys_weight_keys_mask = isin_cupy(node_weight_keys, add_keys_np)
            add_keys_weight_keys_indices = np.nonzero(add_keys_weight_keys_mask)[0]
            add_keys_weight_values = node_weight_values[add_keys_weight_keys_indices]
            sorted_add_keys_weight_values_indices = np.argsort(add_keys_weight_values)
            # Sort descending
            if use_descending: sorted_add_keys_weight_values_indices = sorted_add_keys_weight_values_indices[::-1]
            add_keys_np = add_keys_np[sorted_add_keys_weight_values_indices]
            # features = features[sorted_add_keys_weight_values_indices]

        # Add new features
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
        # Build a numpy array of keys to be added
        add_keys_np = np.fromiter(node_features.keys(), dtype=np.int32)
        if self.cache_alg != CACHEALG.JACA:
            feats = torch.stack(list(node_features.values()), dim=0)
            self.cache_g.put_batch(add_keys_np, feats, namespace=name)
            return
        ## else:
        # Get feature pools and indexes
        g_feature_pool = self.g_feature_pool[name][0]
        existing_keys_np, _ = self.get_g_feature_ids_map_kv(name)

        # Find existing features
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)
        existing_mask = torch.from_numpy(existing_mask_np)
        add_keys_np = add_keys_np[~existing_mask_np]
        if add_keys_np.size == 0: return
        # Filter the features that need to be added
        features_to_add = torch.stack(list(node_features.values()), dim=0)[~existing_mask]
        # Find the starting position of the free position
        start_index = torch.nonzero(g_feature_pool[:, 0] == 0).view(-1).min().item()
        # If the remaining space is insufficient, only a part of it is cached
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
        # Find existing and new keys
        existing_mask_np = isin_cupy(add_keys_np, existing_keys_np)
        existing_indices = np.nonzero(existing_mask_np)[0]
        update_keys_np = add_keys_np[existing_indices]
        add_keys_np = add_keys_np[~existing_mask_np]

        # Update existing features
        if update_keys_np.size > 0:
            existing_keys_mask = isin_cupy(existing_keys_np, update_keys_np)
            existing_keys_indices = np.nonzero(existing_keys_mask)[0]
            self.g_feature_pool[name][1][:existing_keys_indices.size].copy_(features[existing_indices], non_blocking=True)
            self.g_feature_pool[name][0][existing_keys_indices].copy_(self.g_feature_pool[name][1][:existing_keys_indices.size], non_blocking=True)
            feature_len += existing_indices.size

        # Add new features
        if add_keys_np.size > 0:
            features = features[~existing_mask_np]
            empty_rows_mask = ~torch.any(g_feature_pool != 0, dim=1)
            empty_rows_indices = torch.nonzero(empty_rows_mask).view(-1)
            if empty_rows_indices.size(0) == 0: return feature_len
            start_index = empty_rows_indices[0].item()
            available_space = g_feature_pool.size(0) - start_index
    
            # Sort by weight, insert into available space
            if use_weight_update:
                node_weight_keys = self.global_id_counter[0].numpy()
                node_weight_values = self.global_id_counter[1].numpy()
                add_keys_weight_keys_mask = isin_cupy(node_weight_keys, add_keys_np)
                add_keys_weight_keys_indices = np.nonzero(add_keys_weight_keys_mask)[0]
                add_keys_weight_values = node_weight_values[add_keys_weight_keys_indices]
                sorted_add_keys_weight_values_indices = np.argsort(add_keys_weight_values)
                # Sort descending
                if use_descending: sorted_add_keys_weight_values_indices = sorted_add_keys_weight_values_indices[::-1]
                add_keys_np = add_keys_np[sorted_add_keys_weight_values_indices]

            if available_space < features.size(0):
                # Cache replacement: Replace existing ones with high weights with low scores based on the node weights.
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
        #         # Cache replacement: Replace existing ones with high weights with low scores based on the node weights.
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
        # Create a Barrier to ensure all processes are synchronized
        self.cache_barrier = manager.Barrier(gpus_num)
        self.lsize = None
        self.gsize = gsize
        self.dims = dims
        self.cache_alg = cache_alg
        self.cache_comm_queue = [manager.Queue() for _ in range(gpus_num)]
        
        # Initialize the global feature pool
        self.g_feature_pool = {
            "forward0": [torch.zeros(size=(gsize, dims[0]), dtype=self.feat_type).share_memory_(), None],
            "forward1": [torch.zeros(size=(gsize, dims[1]), dtype=self.feat_type).share_memory_(), None],
            "forward2": [torch.zeros(size=(gsize, dims[2]), dtype=self.feat_type).share_memory_(), None],
            "backward2": [torch.zeros(size=(gsize, dims[2]), dtype=self.feat_type).share_memory_(), None],
            "backward1": [torch.zeros(size=(gsize, dims[1]), dtype=self.feat_type).share_memory_(), None],
        }
        # Node global id <=> Global feature pool location index
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
        # Counter of global ID
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
                logger.info(f"[{part_id}-{i}]This time, {len(temp.keys())}/{len(halo_global_ids)} halo nodes are added to enter")
        logger.info(f"The final number of valid nodes：{len(halo_node_features.keys())}")
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
        # Loading pkl file /home/sxf/Desktop/gnn/dist_gnn_fullbatch
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
                _, ori_idxs = torch.sort(torch.tensor(self.global_id_counter[1])[indices], descending=use_descending)  # Sort descending

                # method 3:
                # agg_scores = _get_agg_scores(g, halo_local_masks, nodes_feats, model_type)
                # indices = np.nonzero(np.isin(self.global_id_counter[0].numpy(), halo_global_ids.numpy()))[0]
                # scores = (agg_scores[0] + agg_scores[1]) * self.global_id_counter[1][indices]
                # _, ori_idxs = torch.sort(scores, descending=True)
                # self.global_id_counter[1][indices] = scores


                # select top k
                top_k_ids = halo_global_ids[ori_idxs[:k] if k > 0 else ori_idxs]
                # Batch extraction of halo node features from partition i
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
                logger.info(f"[{part_id}-{i}]This time, {len(temp.keys())}/{len(halo_global_ids)} halo nodes are added to enter")
        logger.info(f"The final number of valid nodes: {len(halo_node_features.keys())}")
        return halo_node_features, max_subg_size

    @staticmethod
    def cal_capacity(part_size, dataset, gpus_list, f_dims, part_dir='data/part_data', k=-1, our_partition=False):
        # reserved_mem is a manually set value, used to exclude memory that needs to be occupied during training, such as gradients, etc. For example, 10MB.
        reserved_mem_gpu = 0  # 10 * 1024
        reserved_mem_cpu = 0  # 10 * 1024
        # cpu_mem is a manually set value, indicating the available memory of the CPU. For example, 256GB.
        cpu_mem = 786 * 1024 * 1024
        # Actual cache capacity per GPU
        gpus_capacity = {}
        # Maximum cache capacity on the CPU
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
                # Here is to get which halo nodes belong to partition i in the current partition part_id. Expressed as halo_global_ids
                halo_local_masks = (nodes_feats['part_id'] == i)
                halo_local_idx = torch.nonzero(halo_local_masks).view(-1)
                if halo_local_idx.numel() == 0: continue
                halo_global_ids = nodes_feats[dgl.NID][halo_local_idx]
                for global_id in halo_global_ids:
                    if global_id.item() in halo_node_ids: continue
                    halo_node_ids.add(global_id.item())
            # The maximum required cache nodes in the current partition part_id
            max_capacity = len(halo_node_ids)
            gpu_id = gpu_ids[part_id]
            # Based on the maximum number of cache nodes, the available memory of GPU, the vertex dimensions and other information, we can obtain the cache capacity that the current GPU can actually bear.
            gpu_memory = gpu.gpu_capability[gpu_id][0]  # MB
            temp = (gpu_memory - reserved_mem_gpu) * 1024 * 1024  / sum(f_dims) / 4
            gpus_capacity[part_id] = min(temp, max_capacity)
        temp = (cpu_mem - reserved_mem_cpu) * 1024 * 1024  / sum(f_dims) / 4
        cpu_capacity = min(temp, sum(gpus_capacity.values()))
        return cpu_capacity, gpus_capacity

def warmup(device):
    # Define a network
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
    # Initialize local cache
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

