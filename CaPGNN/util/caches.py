import threading
from collections import OrderedDict
from enum import Enum, auto
import numpy as np
import torch


class CACHEALG(Enum):
    JACA = auto()
    FIFO = auto()
    LRU = auto()
    RAND = auto()
    LFU = auto()
    LFUA = auto()

CACHEMAP = {'jaca':CACHEALG.JACA, 'lru':CACHEALG.LRU, 'fifo':CACHEALG.FIFO,
            'rand':CACHEALG.RAND, 'lfu':CACHEALG.LFU, 'lfua':CACHEALG.LFUA,}


# Base class (optional for shared structure)
class BaseCache:
    def __init__(self):
        self.cache = {}  # namespace -> OrderedDict
        self.reset_stats()
        self.lock = threading.Lock()

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_size": sum(len(ns) for ns in self.cache.values()),
        }

    def _get_ns_cache(self, namespace):
        if namespace not in self.cache: self.cache[namespace] = OrderedDict()
        return self.cache[namespace]

    def get_exist_mask(self, ids, namespace="default"):
        keys = [id_ for id_ in ids]
        ns_cache = self._get_ns_cache(namespace)
        return np.array([id_ in ns_cache for id_ in keys], dtype=bool)

    def is_keys_exist(self, ids, namespace="default"):
        return self.get_exist_mask(ids, namespace)

    def missing(self, ids, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        return [id_ for id_ in ids if id_ not in ns_cache]

    def put_batch(self, ids, features, namespace="default"):
        with self.lock:  # 使用锁确保线程安全
            for id_, feature in zip(ids, features):
                self.put(id_, feature, namespace)

    def get_batch(self, ids, namespace="default"):
        with self.lock:  # 使用锁确保线程安全
            cached_values = []
            cached_indices = []

            is_torch = isinstance(ids, torch.Tensor)
            is_numpy = isinstance(ids, np.ndarray)
            ids_list = ids.tolist() if (is_torch or is_numpy) else ids
            for idx, id_ in enumerate(ids_list):
                val = self.get(id_, namespace)
                if val is not None:
                    cached_values.append(val)
                    cached_indices.append(idx)
        if len(cached_values) == 0: return None, None
        if is_torch:
            cached_indices = torch.tensor(cached_indices, dtype=torch.long)
        elif is_numpy:
            cached_indices = np.array(cached_indices, dtype=int)

        if isinstance(cached_values[0], torch.Tensor):
            target_device = cached_values[0].device
            cached_values = [v.to(target_device) if isinstance(v, torch.Tensor) else v for v in cached_values]
            cached_values = torch.stack(cached_values)
        elif isinstance(cached_values[0], np.ndarray):
            cached_values = np.stack(cached_values)
        return cached_values, cached_indices

# FIFO Cache
class FIFOCache(BaseCache):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def put(self, id_, feature, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        if id_ not in ns_cache:
            if len(ns_cache) >= self.capacity:
                ns_cache.popitem(last=False)
                self.evictions += 1
        ns_cache[id_] = feature

    def get(self, id_, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        if id_ in ns_cache:
            self.hits += 1
            return ns_cache[id_]
        else:
            self.misses += 1
            return None

# LRU Cache
class LRUCache(BaseCache):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def put(self, id_, feature, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        if id_ in ns_cache: ns_cache.move_to_end(id_)
        ns_cache[id_] = feature
        if len(ns_cache) > self.capacity:
            ns_cache.popitem(last=False)
            self.evictions += 1

    def get(self, id_, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        if id_ in ns_cache:
            ns_cache.move_to_end(id_)
            self.hits += 1
            return ns_cache[id_]
        else:
            self.misses += 1
            return None

# RAND Cache
class RANDCache(BaseCache):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def put(self, id_, feature, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        if id_ not in ns_cache:
            if len(ns_cache) >= self.capacity:
                # 随机选择一个键进行淘汰
                random_key = np.random.choice(list(ns_cache.keys()))
                del ns_cache[random_key]
                self.evictions += 1
        ns_cache[id_] = feature

    def get(self, id_, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        if id_ in ns_cache:
            self.hits += 1
            return ns_cache[id_]
        else:
            self.misses += 1
            return None

# LFU Cache
class LFUCache(BaseCache):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def put(self, id_, feature, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        if id_ in ns_cache:
            ns_cache[id_]['count'] += 1
        else:
            if len(ns_cache) >= self.capacity:
                # 找到访问频率最低的键
                min_count = float('inf')
                min_key = None
                for key, value in ns_cache.items():
                    if value['count'] < min_count:
                        min_count = value['count']
                        min_key = key
                if min_key is not None:
                    del ns_cache[min_key]
                    self.evictions += 1
            ns_cache[id_] = {'feature': feature, 'count': 1}

    def get(self, id_, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        if id_ in ns_cache:
            self.hits += 1
            ns_cache[id_]['count'] += 1
            return ns_cache[id_]['feature']
        else:
            self.misses += 1
            return None

# LFUA Cache
class LFUACache(BaseCache):
    def __init__(self, capacity, aging_factor=0.9):
        super().__init__()
        self.capacity = capacity
        self.aging_factor = aging_factor

    def put(self, id_, feature, namespace="default"):
        ns_cache = self._get_ns_cache(namespace)
        if id_ in ns_cache:
            ns_cache[id_]['count'] += 1
        else:
            if len(ns_cache) >= self.capacity:
                # 找到访问频率最低的键
                min_count = float('inf')
                min_key = None
                for key, value in ns_cache.items():
                    if value['count'] < min_count:
                        min_count = value['count']
                        min_key = key
                if min_key is not None:
                    del ns_cache[min_key]
                    self.evictions += 1
            ns_cache[id_] = {'feature': feature, 'count': 1}

    def get(self, id_, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        if id_ in ns_cache:
            self.hits += 1
            ns_cache[id_]['count'] *= self.aging_factor
            ns_cache[id_]['count'] += 1
            return ns_cache[id_]['feature']
        else:
            self.misses += 1
            return None

    def _age_counts(self, namespace="default"):
        ns_cache = self.cache.get(namespace, {})
        for key in ns_cache:
            ns_cache[key]['count'] *= self.aging_factor



# 全局缓存实例池（策略名 -> 实例）
_cache_singletons = {}

def create_cache(alg: CACHEALG, capacity: int=None, singleton: bool=True, throw_err=True):
    if singleton and alg in _cache_singletons: return _cache_singletons[alg]
    # 选择策略构造类
    if alg == CACHEALG.FIFO:
        print(">> 创建了 FIFO Cache <<")
        instance = FIFOCache(capacity)
    elif alg == CACHEALG.LRU:
        print(">> 创建了 LRU Cache <<")
        instance = LRUCache(capacity)
    elif alg == CACHEALG.RAND:
        print(">> 创建了 RAND Cache <<")
        instance = RANDCache(capacity)
    else:
        if not throw_err: return None
        raise ValueError(f"Unsupported strategy: {alg}")
    if singleton: _cache_singletons[alg] = instance
    return instance

def has_cache_instance(strategy: CACHEALG, capacity: int) -> bool:
    return strategy in _cache_singletons

def list_cache_instances():
    return list(_cache_singletons.keys())

def clear_cache_instances():
    _cache_singletons.clear()


# ------------test----------------- #
def test_fifo_cache():
    import numpy as np
    cache = FIFOCache(capacity=3)
    ids = ['id1', 'id2', 'id3']
    features = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    # 批量 put
    cache.put_batch(ids, features)
    # 批量 get
    res = cache.get_batch(['id2', 'id4'])  # id4 不存在
    print("Get batch:", res)
    # 查看缺失项
    miss = cache.missing(['id2', 'id3', 'id5'])
    print("Missing:", miss)
    # FIFO 测试
    cache.put('id4', np.array([7, 8]))  # 淘汰 id1
    print("After FIFO:")
    print(list(cache.items()))
    print("Stats:", cache.stats())

def test_lru_cache():
    import numpy as np
    cache = LRUCache(capacity=3)
    ids = ['id1', 'id2', 'id3']
    features = [np.array([1]), np.array([2]), np.array([3])]
    cache.put_batch(ids, features)
    # 模拟访问
    print(cache.get('id1'))  # 会更新 id1 为最近使用
    # 添加新项，会淘汰最久未使用的（id2）
    cache.put('id4', np.array([4]))
    print("Cache keys after LRU eviction:")
    # 查看缓存状态和统计信息
    print("Cache keys:", list(cache.cache.keys()))
    print("Stats:", cache.stats())
# ------------test----------------- #

if __name__ == '__main__':
    print('#' * 50)
    test_fifo_cache()
    print('*'* 50)
    test_lru_cache()
    print('#' * 50)
