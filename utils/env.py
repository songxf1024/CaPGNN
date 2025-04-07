import os
import time
import torch
import torch.distributed as dist
import tempfile
from timer import DistTimer
from logger import DistLogger


class DistEnv:
    def __init__(self, rank, world_size, backend='nccl', gpus_note='', log_prefix='', force_single_gpu=False):
        self.force_single_gpu = force_single_gpu
        assert(rank>=0)
        assert(world_size>0)
        # world_size就是nprocs
        self.rank, self.world_size = rank, world_size
        self.backend = backend
        self.init_device()
        if not force_single_gpu: self.init_dist_groups()
        self.logger = DistLogger(self, gpus_note, log_prefix=log_prefix)
        self.timer = DistTimer(self)
        try:
            self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist'), self.world_size)
        except:
            self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist-sxf'), self.world_size)
        self.csr_enabled = False
        self.half_enabled = False
        DistEnv.env = self  # no global...

    def __repr__(self):
        return '<DistEnv %d/%d %s>'%(self.rank, self.world_size, self.backend)

    def init_device(self):
        if torch.cuda.device_count()>=1:
            self.device = torch.device('cuda', self.rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

    def all_reduce_sum(self, tensor):
        if not self.force_single_gpu: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.world_group)

    def broadcast(self, tensor, src):
        if not self.force_single_gpu: dist.broadcast(tensor, src=src, group=self.world_group)

    def all_gather_then_cat(self, src_t):
        recv_list = [torch.zeros_like(src_t) for _ in range(self.world_size)]
        dist.all_gather(recv_list, src_t, group=self.world_group)
        return torch.cat(recv_list, dim=0)

    def barrier_all(self):
        if not self.force_single_gpu: dist.barrier(self.world_group)

    def init_dist_groups(self):
        # 初始化分布式库所使用的默认进程组。
        # backend指定使用哪个后端， rank是当前的GPU_ID， world_size是分布式环境中的进程数量， init_method是指定如何初始化分布式环境的URL字符串。
        # 在使用'env://'时，PyTorch会根据环境变量中设置的信息来初始化分布式训练环境。
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size, init_method='env://')
        # dist.new_group  是 PyTorch 中用于创建分布式训练中的进程组的方法。
        # 在分布式训练中，进程组可以是整个集合的一个子集，这些进程共享相同的通信通道。
        # 在这里， dist.new_group用于创建一些点对点通信的进程组，它们将在后续的代码中使用。
        self.world_group = dist.new_group(list(range(self.world_size)))
        self.p2p_group_dict = {}
        # 创建一些仅包含特定进程对的新组
        for src in range(self.world_size):
            for dst in range(src+1, self.world_size):
                self.p2p_group_dict[(src, dst)] = dist.new_group([src, dst])
                self.p2p_group_dict[(dst, src)] = self.p2p_group_dict[(src, dst)]




class DistUtil:
    def __init__(self, env):
        self.env = env


if __name__ == '__main__':
    pass

