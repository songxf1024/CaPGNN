import os
import pickle
from multiprocessing import Event
from multiprocessing.pool import ThreadPool
from typing import List
import dgl
from gpu import gpu_capability, get_gpu_capability
from .conversion import *
from .processing import *
from ..communicator import Communicator as comm
from ..helper import BitType, DistGNNType
from ..util import Timer, Recorder
from .reducer import Reducer

class DecompGraph(object):
    '''
    每个分区的分解图class，管理中心图和边缘图，以及它们相互获取信息的缓冲区。
    '''
    def __init__(self,
                 central_graph: DGLHeteroGraph,
                 marginal_geaph :DGLHeteroGraph,
                 src_marginal_idx: Tensor,
                 src_central_idx: Tensor):
        self.central_graph = central_graph
        self.marginal_graph = marginal_geaph
        self._src_marginal_idx = src_marginal_idx
        self._src_central_idx = src_central_idx
        self.copy_buffers: List[Tuple[Tensor, Tensor]] = []  # copy messages from central/marginal graphs to marginal/central graphs
    
    @property
    def src_marginal_idx(self):
        return self._src_marginal_idx
    
    @property
    def src_central_idx(self):
        return self._src_central_idx
    
    def to(self, device: torch.device):
        self.central_graph = self.central_graph.to(device)
        self.marginal_graph = self.marginal_graph.to(device)
        self._src_central_idx = self._src_central_idx.to(device)
        self._src_marginal_idx = self._src_marginal_idx.to(device)
    
    def init_copy_buffers(self, feats_dim: int, hidden_dim: int, num_layers: int, device: torch.device):
        src_marginal_size = self.src_marginal_idx.size(0)
        src_central_size = self.src_central_idx.size(0)
        self.copy_buffers.append((torch.zeros(size=(src_marginal_size, feats_dim), device=device), torch.zeros(size=(src_central_size, feats_dim), device=device)))
        for _ in range(num_layers - 1):
            self.copy_buffers.append((torch.zeros(size=(src_marginal_size, hidden_dim), device=device), torch.zeros(size=(src_central_size, hidden_dim), device=device)))
    
    def get_copy_buffers(self, layer: int) -> Tuple[Tensor, Tensor]:
        return self.copy_buffers[layer]

import torch.distributed as dist
def get_boundary(node_dict, gpb):
    rank, size = dist.get_rank(), dist.get_world_size()
    device = 'cuda'
    boundary = [None] * size
    for i in range(1, size):
        left = (rank - i + size) % size
        right = (rank + i) % size
        belong_right = (node_dict['part_id'] == right)
        num_right = belong_right.sum().view(-1)
        if dist.get_backend() == 'gloo':
            num_right = num_right.cpu()
            num_left = torch.tensor([0])
        else:
            num_left = torch.tensor([0], device=device)
        req = dist.isend(num_right, dst=right)
        dist.recv(num_left, src=left)
        start = gpb.partid2nids(right)[0].item()
        v = node_dict[dgl.NID][belong_right] - start
        if dist.get_backend() == 'gloo':
            v = v.cpu()
            u = torch.zeros(num_left, dtype=torch.long)
        else:
            u = torch.zeros(num_left, dtype=torch.long, device=device)
        req.wait()
        req = dist.isend(v, dst=right)
        dist.recv(u, src=left)
        u, _ = torch.sort(u)
        if dist.get_backend() == 'gloo':
            boundary[left] = u.cuda()
        else:
            boundary[left] = u
        req.wait()

    return boundary

def get_recv_shape(node_dict):
    '''计算分布式训练中每个进程应该接收的数据量'''
    rank, size = dist.get_rank(), dist.get_world_size()
    recv_shape = []
    for i in range(size):
        if i == rank:
            recv_shape.append(None)
        else:
            # 计算其他进程需要发送给当前进程的数据量
            t = (node_dict['part_id'] == i).int().sum().item()
            recv_shape.append(t)
    return recv_shape


class GraphEngine(object):
    '''
    管理图和各种节点特征（如特征、标签、mask等）。
    '''
    def __init__(self,
                 args,
                 epoches: int,
                 part_dir,
                 dataset,
                 msg_precision_type: str,
                 model_type: DistGNNType,
                 storage_server,
                 use_parallel=False,):
        if args['our_cache'] or args['do_reorder']:
            print(">> 使用pkl <<")
            gpus_list = args['gpus_list']
            # 按指标值降序排序，并获取对应的GPU设备key。值越大，计算成本越大，GPU越弱。
            sorted_gpu_ids = gpus_list  # get_gpu_capability(gpus_list)
            partition_file = f'{part_dir}/{dataset}/{len(gpus_list)}part/{dataset}_processed_partitions_{args["our_partition"]}_{sorted(gpus_list)}.pkl'
            with open(partition_file, 'rb') as f: assig_graphs_gpus = pickle.load(f)
            g_info = assig_graphs_gpus[sorted_gpu_ids[comm.get_rank()]]
            if args['cvt_fmts']:
                g_info.graph = g_info.graph.formats(['coo', 'csr', 'csc'])  # ['coo', 'csr', 'csc']
                g_info.graph.create_formats_()
                # print(g_info.graph.is_pinned())
                # g_info.graph.pin_memory_()
                # print(g_info.graph.is_pinned())

            self._is_bidirected = g_info.is_bidirected
            self._use_parallel = use_parallel
            self._bit_type = BitType.FULL
            # set device
            self._device = comm.ctx.device
            self.graph = g_info.graph
            self.g_info = g_info
            # set node info and idx
            self._num_remove, self._num_inner, self._num_marginal, self._num_central = g_info.num_remote, g_info.num_inner, g_info.num_marginal, g_info.num_central
            self._send_idx, self._recv_idx, self._scores = g_info.send_idx, g_info.recv_idx, g_info.agg_scores
            self._total_send_idx = g_info.total_send_idx
            # set feats and labels
            self.feats = g_info.node_feats['feat']
            self.labels = g_info.node_feats['label']
            # set masks
            self.train_mask = torch.nonzero(g_info.node_feats['train_mask']).squeeze()
            self.val_mask = torch.nonzero(g_info.node_feats['val_mask']).squeeze()
            self.test_mask = torch.nonzero(g_info.node_feats['test_mask']).squeeze()
            self.gpb = g_info.gpb
            self._init_stream_ctx()
            print(f"GPU {torch.cuda.get_device_name(self._device)}: {self.graph}")
            self.boundary = get_boundary(g_info.node_feats, g_info.gpb)
            # print(self.boundary)
            self.recv_shape = get_recv_shape(g_info.node_feats)
            print(self.recv_shape)
        else:
            # load original graph and feats
            original_graph, original_feats, gpb, is_bidirected = convert_partition(part_dir, dataset)
            # get send_idx, recv_idx, and scores
            original_send_idx, recv_idx, scores = get_send_recv_idx_scores(original_graph, original_feats, gpb, part_dir, dataset, model_type)
            # 重新排列图中节点ID的顺序（从 0 到 N-1：中心节点->边缘节点->边远节点）
            reordered_graph, reordered_feats, reordered_send_idx, num_marginal, num_central = reorder_graph(original_graph, original_feats, original_send_idx)
            send_idx, total_idx = convert_send_idx(reordered_send_idx)
            reordered_graph.ndata['in_degrees'] = reordered_feats['in_degrees']
            reordered_graph.ndata['out_degrees'] = reordered_feats['out_degrees']
            num_inner = torch.count_nonzero(reordered_feats['inner_node']).item()
            num_remote = reordered_graph.num_nodes() - num_inner
            assert num_inner == num_central + num_marginal
            self._is_bidirected = is_bidirected
            self._use_parallel = use_parallel
            self._bit_type = BitType.FULL
            # set node info and idx
            self._num_remove, self._num_inner, self._num_marginal, self._num_central = num_remote, num_inner, num_marginal, num_central
            self._send_idx, self._recv_idx, self._scores = send_idx, recv_idx, scores
            self._total_send_idx = total_idx
            # set feats and labels
            self.feats = reordered_feats['feat']
            self.labels = reordered_feats['label']
            # set masks
            self.train_mask = torch.nonzero(reordered_feats['train_mask']).squeeze()
            self.val_mask = torch.nonzero(reordered_feats['val_mask']).squeeze()
            self.test_mask = torch.nonzero(reordered_feats['test_mask']).squeeze()
            # set device
            self._device = comm.ctx.device
            self.graph = reordered_graph
            # pop unnecessary feats
            reordered_feats.pop('in_degrees')
            reordered_feats.pop('out_degrees')
            reordered_feats.pop('feat')
            reordered_feats.pop('label')
            reordered_feats.pop('train_mask')
            reordered_feats.pop('val_mask')
            reordered_feats.pop('test_mask')
            reordered_feats.pop('part_id')
            reordered_feats.pop(dgl.NID)
            self.g_info = None
            self.gpb = None
        print(f'[Load] {comm.get_rank()}=>{self.graph.formats()}')
        self.storage_server = storage_server
        self.our_cache = args['our_cache']
        self.our_partition = args['our_partition']
        self.use_pipeline = args['use_pipeline']
        self.cache_alg = args['cache_alg']
        self.reducer = Reducer()
        # move to device
        self._move()
        # !!!set backward graph after moving!!! 
        self._set_bwd_graph(use_parallel, self._is_bidirected)
        # init the timer for recording the time cost
        self.timer = Timer(device=self._device)
        # init the recorder for recording metrics
        self.recorder = Recorder(epoches)
        # graphSAGE aggregator type if needed
        self._agg_type: str = None
        # init marginal thread
        self.marginal_pool = ThreadPool(processes=2)
        self.pick_pool = ThreadPool(processes=2)

        self._n_layers = args["n_layers"]
        self._comm_stream = torch.cuda.Stream()
        self._corr_stream = torch.cuda.Stream()
        self._f_cpu_event, self._b_cpu_event = [None] * self._n_layers, [None] * self._n_layers
        self._f_cuda_event, self._b_cuda_event = [None] * self._n_layers, [None] * self._n_layers
        for i in range(self._n_layers):
            self._f_cpu_event[i] = Event()
            self._b_cpu_event[i] = Event()
            self._f_cuda_event[i] = torch.cuda.Event()
            self._b_cuda_event[i] = torch.cuda.Event()
        # self._pool = ThreadPool(processes=2 * self._n_layers)
        self.curr_epoch = 0
        # set ctx
        GraphEngine.ctx = self
    
    def __repr__(self):
        return f'<GraphEngine(rank: {comm.get_rank()}, remote nodes: {self.num_remove} central nodes: {self.num_central}, marginal nodes: {self.num_marginal})>'
    
    def _init_stream_ctx(self):
        # init streams for resources isolation (central graph computation uses default stream)
        self.marginal_stream = torch.cuda.Stream(device=comm.ctx.device)
        self.cache_stream = [torch.cuda.Stream(device=comm.ctx.device) for _ in range(4)]
        # init cuda event to synchronize streams
        self.quant_cuda_event = torch.cuda.Event()
        self.comp_cuda_event = torch.cuda.Event() # used by central graph computation
        # init cpu event since quant/dequant and communication operations on marginal graph are called asynchronously
        self.quant_cpu_event = Event()
        self.comp_cpu_event = Event() # used by central graph computation
        # init marginal thread
        # self.marginal_pool = ThreadPool(processes=1)
        # self.pick_pool = ThreadPool(processes=1)

        
    def _set_bwd_graph(self, use_parallel: bool, is_bidirected: bool):
        # 重叠通信和计算 (暂时先不管这部分，只用顺序执行)
        if use_parallel:
            # 对于单向图，生成一个反向图. 只生成结构，不复制节点和边的数据
            if not is_bidirected:
                reverse_graph = dgl.reverse(self.graph, copy_ndata=False, copy_edata=False)
                central_graph, marginal_graph, src_marginal_idx, src_central_idx = decompose_graph(reverse_graph, self.num_central, self.num_inner)
                self.bwd_graph = DecompGraph(central_graph, marginal_graph, src_marginal_idx, src_central_idx)
            else:
                self.bwd_graph = self.graph
        else:
            if not is_bidirected:
                self.bwd_graph = dgl.reverse(self.graph, copy_ndata=False, copy_edata=False)
            else:
                self.bwd_graph = self.graph

    def _move(self):
        # model the foward graph
        if self._use_parallel:
            self.graph.to(self._device)
        else:
            self.graph = self.graph.to(self._device)
        # move feats and labels
        self.feats = self.feats.to(self._device)
        self.labels = self.labels.to(self._device)
        # move masks
        self.train_mask = self.train_mask.to(self._device)
        self.val_mask = self.val_mask.to(self._device)
        # self.test_mask = self.test_mask.to(self._device)
        # g_info
        if self.our_cache:
            self.g_info.node_feats['feat'] = self.feats  # self.g_info.node_feats['feat'].to(self._device)
            self.g_info.node_feats[dgl.NID] = self.g_info.node_feats[dgl.NID].to(self._device)

    '''
    *************************************************
    ***************** getter methods ****************
    *************************************************
    '''

    @property
    def device(self):
        return self._device

    @property
    def is_bidirected(self):
        return self._is_bidirected
    
    @property
    def use_parallel(self):
        return self._use_parallel
    
    @property
    def bit_type(self):
        return self._bit_type
    
    @property
    def agg_type(self):
        assert self._agg_type is not None, 'please set the aggregator type first.'
        return self._agg_type
    
    @agg_type.setter
    def agg_type(self, agg_type: str):
        self._agg_type = agg_type
    
    @property
    def num_remove(self):
        return self._num_remove
    
    @property
    def num_inner(self):
        return self._num_inner
    
    @property
    def num_marginal(self):
        return self._num_marginal
    
    @property
    def num_central(self):
        return self._num_central
    
    @property
    def send_idx(self):
        return self._send_idx
    
    @property
    def recv_idx(self):
        return self._recv_idx
    
    @property
    def scores(self):
        return self._scores
    
    @property
    def total_send_idx(self):
        return self._total_send_idx


    
        
        
        