import numpy as np
from collections import deque

class ARC_Cache:
    def __init__(self, cache_size):
        self.cache_size = cache_size
        self.HASHSIZE = 1000000
        self.Hash = np.zeros(self.HASHSIZE, dtype=np.int32)
        self.mrug = deque()  # Ghost cache for recent LRU
        self.mru = deque()   # Recent LRU cache
        self.mfu = deque()   # Frequent LRU cache
        self.mfug = deque()  # Ghost cache for frequent LRU
        self.p = 0.0  # Balancing parameter
        self.HitCount = 0
        self.MissCount = 0

    def queue_insert(self, v, i):
        if len(v) == self.cache_size:
            v.popleft()  # Remove the oldest element if full
        v.append(i)
        return v

    def movefrom(self, v, w, x):
        if x in v:
            v.remove(x)
            if len(w) >= self.cache_size:
                w.popleft()
            w.append(x)
        return v, w

    def Replace(self, i):
        if len(self.mru) > 0 and (len(self.mru) > self.p or (i in self.mfug and self.p == len(self.mru))):
            self.mru, self.mrug = self.movefrom(self.mru, self.mrug, self.mru[0])
        elif len(self.mfu) > 0:
            self.mfu, self.mfug = self.movefrom(self.mfu, self.mfug, self.mfu[0])

    def arc_lookup(self, i):
        if self.Hash[i % self.HASHSIZE] > 0:
            if i in self.mru:
                self.HitCount += 1
                self.mru, self.mfu = self.movefrom(self.mru, self.mfu, i)
                return True
            elif i in self.mfu:
                self.HitCount += 1
                self.mfu, self.mfu = self.movefrom(self.mfu, self.mfu, i)
                return True
            elif i in self.mrug:
                self.MissCount += 1
                self.p = min(float(self.cache_size), self.p + max(len(self.mfug) / len(self.mrug), 1.0))
                self.Replace(i)
                self.mrug, self.mfu = self.movefrom(self.mrug, self.mfu, i)
                return False
            elif i in self.mfug:
                self.MissCount += 1
                self.p = max(0.0, self.p - max(len(self.mrug) / len(self.mfug), 1.0))
                self.Replace(i)
                self.mfug, self.mfu = self.movefrom(self.mfug, self.mfu, i)
                return False
            else:
                self.MissCount += 1
                if len(self.mru) + len(self.mrug) == self.cache_size:
                    if len(self.mru) < self.cache_size:
                        self.Hash[self.mrug[0] % self.HASHSIZE] -= 1
                        self.mrug.popleft()
                        self.Replace(i)
                elif len(self.mru) + len(self.mrug) < self.cache_size:
                    if len(self.mru) + len(self.mfu) + len(self.mrug) + len(self.mfug) >= self.cache_size:
                        if len(self.mru) + len(self.mfu) + len(self.mrug) + len(self.mfug) == 2 * self.cache_size:
                            self.Hash[self.mfug[0] % self.HASHSIZE] -= 1
                            self.mfug.popleft()
                        self.Replace(i)
                self.queue_insert(self.mru, i)
                self.Hash[i % self.HASHSIZE] += 1
                return False
        else:
            self.MissCount += 1
            if len(self.mru) + len(self.mrug) == self.cache_size:
                if len(self.mru) < self.cache_size:
                    self.Hash[self.mrug[0] % self.HASHSIZE] -= 1
                    self.mrug.popleft()
                    self.Replace(i)
                else:
                    self.Hash[self.mru[0] % self.HASHSIZE] -= 1
                    self.mru.popleft()
            elif len(self.mru) + len(self.mrug) < self.cache_size:
                if len(self.mru) + len(self.mrug) + len(self.mfu) + len(self.mfug) >= self.cache_size:
                    if len(self.mru) + len(self.mrug) + len(self.mfu) + len(self.mfug) == 2 * self.cache_size:
                        self.Hash[self.mfug[0] % self.HASHSIZE] -= 1
                        self.mfug.popleft()
                    self.Replace(i)
            self.queue_insert(self.mru, i)
            self.Hash[i % self.HASHSIZE] += 1
            return False

    def get_hit_ratio(self):
        total_requests = self.HitCount + self.MissCount
        if total_requests > 0:
            return self.HitCount / total_requests
        else:
            return 0.0

    def reset_counters(self):
        self.HitCount = 0
        self.MissCount = 0
        self.p = 0.0
        self.Hash[:] = 0
        self.mrug.clear()
        self.mru.clear()
        self.mfu.clear()
        self.mfug.clear()

if __name__ == "__main__":
    arc_cache = ARC_Cache(cache_size=1000)
    for i in [1,2,3,4,5,1,2,3,4,6]:
        hit = arc_cache.arc_lookup(i)
        if hit: print(f"命中请求: {i}")
    hit_ratio = arc_cache.get_hit_ratio()
    print(f"命中率: {hit_ratio:.4f}")
