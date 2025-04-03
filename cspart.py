# 算力感知的图分区。根据图的复杂度和GPU的算力来划分子图。
import csv
import os
import random
from enum import Enum
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    import torchdata
    torchdata._warning_shown = True
except ImportError:
    pass
import time
from collections import Counter
import networkx as nx
import numpy as np
import statistics
import argparse
import copy
from typing import Dict, Tuple, List
from torch import Tensor
from CaPGNN.communicator import Basic_Buffer_Type
from CaPGNN.helper import graph_patition_store
from CaPGNN.manager.processing import _get_agg_scores
import pickle
import torch
import dgl
from dgl import DGLHeteroGraph
from gpu import gpu_capability, get_gpu_capability, gpu_memory_enough, cal_gpus_capability
from collections import defaultdict, Counter
import tools

class DistGNNType(Enum):
    DistGCN = 0
    DistSAGE = 1

MODEL_MAP: Dict[str, DistGNNType] = {'gcn': DistGNNType.DistGCN, 'sage': DistGNNType.DistSAGE}

def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class GraphInfo(object):
    def __init__(self, graph=None, node_feats=None, gpb=None,
                 is_bidirected=None, send_idx=None, recv_idx=None,
                 agg_scores=None, total_send_idx=None, num_inner=None,
                 num_remote=None, num_marginal=None, num_central=None,
                 device=None):
        self.graph = graph
        self.gpb = gpb
        self.node_feats = node_feats
        self.is_bidirected = is_bidirected    # bool，是否为无向图(判断条件：每个节点的入度==出度)
        self.send_idx = send_idx              # dict，每个远程分区i上，send_idx的起止位置。如{0: (0, 17660), 2: (17660, 36595)}
        self.total_send_idx = total_send_idx  # list，cat所有远程分区i上，send_idx的具体ID。是对应分区的局部ID。如[xx,xx,xx,...,xx]
        self.recv_idx = recv_idx              # dict，本地分区上，属于远程分区i的HALO节点，相对于内点ID的偏移。如{0: (0, 3, 6, 16), 2: (8, 9)}
        self.agg_scores = agg_scores          # dict，本地分区上，属于远程分区i的HALO节点的分数。如{0: ([x,x]前向, [x,x]后向), 2: ([x,x]前向, [x,x]后向)}
        self.num_inner = num_inner            # int，当前分区上的内部节点数(不包含HALO节点)
        self.num_remote = num_remote          # int，当前分区上的远程节点数(即HALO节点)
        self.num_marginal = num_marginal      # int，
        self.num_central = num_central        # int，
        self.device = device

    def __str__(self):
        torch.set_printoptions(edgeitems=2, threshold=10)
        def format_tensor(tensor):
            return str(tensor).replace('\n', ' ')
        def format_value(value):
            return '{\n' + ',\n'.join(f"    -- {k!r}: {format_tensor(v)}" for k, v in value.items()) + '\n  }' if isinstance(value, dict) else str(value)
        attributes = ',\n>> '.join(f"{key}={format_value(value)}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}(\n>> {attributes}\n)"


class CSPart(object):
    def __init__(self, args):
        self.DEBUG = True
        self.args = args
        self.model_type = MODEL_MAP['gcn']
        self.partition_dir = args['partition_dir']
        self.partition_size = args['partition_size']
        self.dataset_dir = args['dataset_dir']
        self.dataset_name = args['dataset_name']
        self.graph_info: List[GraphInfo] = [GraphInfo() for _ in range(self.partition_size)]
        self.assig_graphs_gpus = {}
        self.sorted_gpu_ids = []
        self.global_id_counter = []

    def log(self, msg):
        if self.DEBUG: print(msg)

    def print_graph(self, rank=-1):
        if not self.DEBUG: return
        if rank != -1:
            self.log(self.graph_info[rank])
        else:
            for rank in range(self.partition_size): self.log(self.graph_info[rank])

    def coarse_graph_patition(self, fast_skip=False, part_method='random', do_partition=False, num_hops=1):
        partition_dir = '{}/{}/{}part'.format(self.partition_dir, self.dataset_name, self.partition_size)
        if fast_skip and os.path.exists(partition_dir): return
        graph_patition_store(self.dataset_name, self.partition_size, self.dataset_dir, self.partition_dir, num_hops=num_hops, part_method=part_method)
        self.log(f"<{self.dataset_name} graph partition done.>")

    def coarse_load_patition(self):
        for rank in range(self.partition_size):
            partition_dir = f'{self.partition_dir}/{self.dataset_name}/{self.partition_size}part'
            part_config = f'{partition_dir}/{self.dataset_name}.json'
            # gpb: RangePartitionBook, 用于管理和处理图数据在分布式环境中的分区信息。
            #   RangePartitionBook 的主要功能是提供节点和边的分区信息。具体而言，它管理了以下内容：
            #      - 节点和边的分区映射：为每个节点和边分配一个分区编号，表示它们属于哪个分区。
            #      - 分区范围：定义每个分区中的节点和边的范围，即每个分区包含的节点和边的ID范围。
            #      - 跨分区通信：在分布式训练中，不同分区的计算节点需要交换节点和边的信息。RangePartitionBook 帮助管理这些跨分区的通信。
            #   RangePartitionBook 提供了一些方法和属性来查询分区信息，例如：
            #      - partid2nids(part_id)：返回指定分区ID的节点ID范围。
            #      - partid2eids(part_id)：返回指定分区ID的边ID范围。
            #      - nid2partid(nid)：返回指定节点ID所在的分区ID。
            #      - eid2partid(eid)：返回指定边ID所在的分区ID。
            g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
            # set graph degrees for GNNs aggregation
            # print(f'{rank}=>{g.formats()} [METIS]')
            save_dir = f'{partition_dir}/graph_degrees'
            # load global degrees information
            in_degrees_global, out_degrees_global = torch.load(f'{save_dir}/in_degrees.pt'), torch.load(f'{save_dir}/out_degrees.pt')
            # g.ndata['orig_id'] = torch.load(f'{save_dir}/orig_id_{rank}.pt')
            # 节点在原始大图中的ID
            orig_id = g.ndata['orig_id']
            nodes_feats['in_degrees'] = in_degrees_global[orig_id]
            nodes_feats['out_degrees'] = out_degrees_global[orig_id]
            is_bidirected = torch.equal(nodes_feats['in_degrees'], nodes_feats['out_degrees'])
            # move all the features to nodes_feats
            node_type = node_type[0]
            # save original degrees for fp and bp
            nodes_feats[dgl.NID] = g.ndata[dgl.NID]
            # 对于只有一个分区的时候，所有节点都算
            nodes_feats['part_id'] = g.ndata.get('part_id', torch.zeros_like(orig_id))
            nodes_feats['inner_node'] = g.ndata['inner_node'].bool()
            nodes_feats['label'] = nodes_feats[node_type + '/label']
            nodes_feats['feat'] = nodes_feats[node_type + '/feat']
            nodes_feats['train_mask'] = nodes_feats[node_type + '/train_mask'].bool()
            nodes_feats['val_mask'] = nodes_feats[node_type + '/val_mask'].bool()
            nodes_feats['test_mask'] = nodes_feats[node_type + '/test_mask'].bool()

            # remove redundant feats
            nodes_feats.pop(node_type + '/val_mask')
            nodes_feats.pop(node_type + '/test_mask')
            nodes_feats.pop(node_type + '/label')
            nodes_feats.pop(node_type + '/feat')
            nodes_feats.pop(node_type + '/train_mask')
            # only remain topology of graph
            g.ndata.clear()
            g.edata.clear()
            #  update
            g_info = self.graph_info[rank]
            g_info.graph = g
            g_info.node_feats = nodes_feats
            g_info.gpb = gpb
            g_info.is_bidirected = is_bidirected
            self.log(f"<partition {rank} load done.>")
        return self.graph_info

    def get_send_recv_idx_scores(self, graph_info, skip_load=False, suffix=''):
        '''
        获取local graph中每个节点的send/recv idx 和 agg 分数。
        '''
        except_list = [('skip' if skip_load else None) for _ in range(self.partition_size)]
        send_idx_list = [None for _ in range(self.partition_size)]
        recv_idx_list = [None for _ in range(self.partition_size)]
        agg_scores_list = [None for _ in range(self.partition_size)]
        if not skip_load:
            for rank in range(self.partition_size):
                current_partition_dir = f'{self.partition_dir}/{self.dataset_name}/{self.partition_size}part/part{rank}'
                try:
                    with open(f'{current_partition_dir}/send_idx{suffix}.pkl', 'rb') as f: send_idx = pickle.load(f)
                    with open(f'{current_partition_dir}/recv_idx{suffix}.pkl', 'rb') as f: recv_idx = pickle.load(f)
                    with open(f'{current_partition_dir}/agg_scores{suffix}.pkl', 'rb') as f: agg_scores = pickle.load(f)
                    graph_info[rank].send_idx = send_idx
                    graph_info[rank].recv_idx = recv_idx
                    graph_info[rank].agg_scores = agg_scores
                    send_idx_list[rank] = send_idx
                    recv_idx_list[rank] = recv_idx
                    agg_scores_list[rank] = agg_scores
                    for k, v in recv_idx.items(): self.log(f'<worker{rank} <=== {len(v)} nodes <=== worker{k}>')
                    for k, v in send_idx.items(): self.log(f'<worker{rank} ===> {len(v)} nodes ===> worker{k}>')
                except IOError as e:
                    except_list[rank] = e
        if skip_load or (not all([except_info is None for except_info in except_list])):
            fail_idx = [i for i, except_info in enumerate(except_list) if except_info is not None]
            self.log(f'<worder {fail_idx} failed to load send/recv idx from disk, begin building...>')
            send_idx_list, recv_idx_list, agg_scores_list = self._build_store_send_recv_idx_scores(graph_info, suffix=suffix)
            for rank in range(self.partition_size):  # fail_idx
                graph_info[rank].send_idx = send_idx_list[rank]
                graph_info[rank].recv_idx = recv_idx_list[rank]
                graph_info[rank].agg_scores = agg_scores_list[rank]
        return graph_info, send_idx_list, recv_idx_list, agg_scores_list

    def _build_store_send_recv_idx_scores(self, graph_info, suffix=''):
        '''
        构建和存储每个分区的发送和接收索引以及聚合分数，并将这些信息保存到磁盘.
        HALO node.
        temp_send_idx: 字典，表示每个分区的发送索引。
        recv_idx: 字典，表示每个分区的接收索引。
        scores: 字典，表示每个分区的聚合分数。
        '''
        # 当前分区上的halo节点，在对应远程分区上局部ID
        temp_send_buffer_list = [{} for _ in range(self.partition_size)]
        # 当前分区上，要发往远程分区的节点的ID（这些节点在当前分区上不是halo节点，在远程分区上是halo节点）
        temp_send_idx_list = [{} for _ in range(self.partition_size)]
        recv_idx_list = [{} for _ in range(self.partition_size)]
        scores_list = [{} for _ in range(self.partition_size)]

        for rank in range(self.partition_size):
            recv_idx: Dict[int, Tensor] = {}
            local_graph = graph_info[rank].graph
            nodes_feats = graph_info[rank].node_feats
            gpb = graph_info[rank].gpb

            # 当前分区rank中内部节点的数量，长度的话，就可以表示外部节点的起始点.
            # 注意，nodes_feats['inner_node']是包含了HALO节点的标识的(False)
            inner_nodes_num = len(torch.nonzero(nodes_feats['inner_node']))
            temp_send_buffer = {}
            for i in range(self.partition_size):
                if i != rank:
                    # 获取当前分区rank中，属于分区i的节点ID (即 分区i => 分区rank 的HALO节点)
                    halo_local_belong2i_mask = (nodes_feats['part_id'] == i)
                    # get forward & backward aggreagtion scores for remote neighbors
                    agg_score = self._get_agg_scores(local_graph, halo_local_belong2i_mask, nodes_feats, self.model_type)
                    # 计算属于分区i的这些HALO节点在当前分区rank中，相对于内点的局部ID或者叫偏移。
                    # torch.nonzero(belong2i) 是取非零值的索引，由于ID连续，因此就可以看做是节点的局部ID。
                    # 因为顺序是 inner node 排在前面，outer node 排在后面，所以可以这么取。
                    halo_local_offset_idx = torch.nonzero(halo_local_belong2i_mask).view(-1) - inner_nodes_num
                    # 计算远程分区i的起始节点的ID (全局 ID) (不包含HALO节点)
                    halo_remote_start_id = gpb.partid2nids(i)[0].item()
                    # halo_remote_start_id2 = graph_info[i].node_feats[dgl.NID][0].item()
                    # 计算在远程分区i上, 相对于远程分区i而言的节点的ID (即 这些节点在分区i上的局部ID或者叫偏移);
                    # dgl.NID是个特殊的键，表示要取所有节点的全局ID。
                    # 这里的全局ID也可以代表节点在全局ID列表中的位置索引，比如[20, 50]，是ID值，也是位置索引。
                    halo_remote_offset_ids = nodes_feats[dgl.NID][halo_local_belong2i_mask] - halo_remote_start_id
                    # 这里的score可以反映每个HALO节点的重要性
                    temp_send_buffer[i] = (halo_remote_offset_ids, agg_score)
                    # 表示这些HALO节点后面需要接收数据
                    recv_idx[i] = halo_local_offset_idx
                    # halo_local_offset_idx表示HALO节点在当前分区的ID(从外点位置开始的)，halo_remote_offset_ids表示HALO节点在所属分区上的ID。
                    # 这里的HALO节点来自分区i。
                    # 注意，总体来看，ID都是从0开始的。这也方便了如果从对方分区i的角度来看时，这些ID就已经是局部ID，不需要再次进行处理。
            temp_send_buffer_list[rank] = temp_send_buffer
            recv_idx_list[rank] = recv_idx
            # print recv_idx in debug mode
            for k, v in recv_idx.items(): self.log(f'<worker{rank} recv {len(v)} nodes from worker{k}>')
        for rank in range(self.partition_size):
            temp_send_idx: Dict[int, Tensor] = {}
            scores: Dict[int, Tensor] = {}
            for part_i in range(self.partition_size):
                if part_i != rank:
                    # 检查分区i是否有与当前rank分区有关联的节点。(或者说，当前分区是否有节点在远程分区当halo节点)
                    # 从分区i的角度来看，这些节点是来自分区rank的HALO节点，是分区i需要recv_idx的。
                    # 因此，对于分区rank而言，相对地是send_idx的。
                    # 由于在前面处理的时候，remote_ids已经直接表示其在所属分区上的ID，因此这里可以直接赋值，不需要再进行局部ID范围的调整。
                    if rank in temp_send_buffer_list[part_i].keys():
                        temp_send_idx[part_i] = temp_send_buffer_list[part_i][rank][0]  # data from part_i to rank
                        scores[part_i] = temp_send_buffer_list[part_i][rank][1]         # score from part_i to rank
            for k, v in temp_send_idx.items(): self.log(f'<worker{rank} send {len(v)} nodes to worker{k}>')
            temp_send_idx_list[rank] = temp_send_idx
            scores_list[rank] = scores

        for rank in range(self.partition_size):
            current_partition_dir = f'{self.partition_dir}/{self.dataset_name}/{self.partition_size}part/part{rank}'
            with open(f'{current_partition_dir}/send_idx{suffix}.pkl', 'wb') as f: pickle.dump(temp_send_idx_list[rank], f)
            with open(f'{current_partition_dir}/recv_idx{suffix}.pkl', 'wb') as f: pickle.dump(recv_idx_list[rank], f)
            with open(f'{current_partition_dir}/agg_scores{suffix}.pkl', 'wb') as f: pickle.dump(scores_list[rank], f)
        return temp_send_idx_list, recv_idx_list, scores_list

    def _get_agg_scores(self, local_graph: dgl.DGLHeteroGraph, belong2i_mask: Tensor, nodes_feats: Dict[str, Tensor], model_type: DistGNNType):
        '''
        返回本地图中每个节点的前向和后向的聚合得分，这些得分用于评估节点的重要性或影响力。
        '''
        halo_nodes = local_graph.nodes()[belong2i_mask]
        fp_local_halo_out_node_ids = local_graph.out_edges(halo_nodes)[1]      # 当前分区图中，halo节点的前向邻居（即出边的目标节点）
        fp_local_out_degrees = local_graph.out_degrees(halo_nodes)  # 当前分区图中，halo节点的出度
        bp_neighbor_ids = local_graph.in_edges(halo_nodes)[0]       # 当前分区图中，halo节点的后向邻居（即入边的源节点）
        bp_local_degrees = local_graph.in_degrees(halo_nodes)       # 当前分区图中，halo节点的入度

        # 删除的点：
        #    一方面，有限删除对当前分区的影响最小，这样对精度影响也小。
        #    另一方面，优先保留出现次数多的，这样可以避免额外可能的通信。
        if model_type is DistGNNType.DistGCN:
            # 前向聚合得分:
            # - 计算前向邻居的全局度数 fp_global_out_degrees。
            # - 对每个前向邻居的入度进行开方倒数的计算，并按节点的出度进行分割和求和，得到前向聚合得分 fp_agg_score。
            # - 局部图中，每个halo节点的出度，占多少个该节点在全局图中入度的根号的倒数？

            # 每个halo节点所指向的内点在全图中的入度数。值越大，说明这个内点在前向传播时要接收的数据越多。
            fp_global_in_degrees = nodes_feats['in_degrees'][fp_local_halo_out_node_ids].float().clamp(min=1) 
            # 每个halo节点在全图中的出度数。出度越多，说明该节点在全图中可能比较重要。但是在当前分区中，由于他是halo节点，所以不一定。
            fp_global_out_degrees = nodes_feats['out_degrees'][belong2i_mask]                        
            # 其实是对fp_global_in_degrees的归一化（平方根的倒数）。值越大，score越小，说明这个halo节点对某个内点会产生的影响越小，因为内点有很多其他可接收的点。
            score = torch.pow(fp_global_in_degrees, -0.5).split(fp_local_out_degrees.tolist())     
            # 对每个halo节点的邻居进行加权求和。对出度数取平方根的倒数，则出度越多，值越小。出度越多，说明即使这个halo节点在当前分区被删除，但在其他分区也可能会被用到。
            fp_agg_score = torch.tensor([torch.sum(score[i] * torch.pow(fp_global_out_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
            # 后向聚合得分:
            # - 计算后向邻居的全局度数 bp_global_degrees。
            # - 对每个后向邻居的出度进行开方倒数的计算，并按节点的入度进行分割和求和，得到后向聚合得分 bp_agg_score。
            bp_global_out_degrees = nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1)
            bp_global_in_degrees = nodes_feats['in_degrees'][belong2i_mask]
            score = torch.pow(bp_global_out_degrees, -0.5).split(bp_local_degrees.tolist())
            bp_agg_score = torch.tensor([torch.sum(score[i] * torch.pow(bp_global_in_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
        elif model_type is DistGNNType.DistSAGE:
            # 前向聚合得分:
            # - 对每个前向邻居的入度进行倒数的计算，并按节点的出度进行分割和求和，得到前向聚合得分 fp_agg_score。
            score = torch.pow(nodes_feats['in_degrees'][fp_local_halo_out_node_ids].float().clamp(min=1), -1).split(fp_local_out_degrees.tolist())
            fp_agg_score = torch.tensor([torch.sum(value) for value in score])
            # 后向聚合得分:
            # - 对每个后向邻居的出度进行倒数的计算，并按节点的入度进行分割和求和，得到后向聚合得分 bp_agg_score。
            score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -1).split(bp_local_degrees.tolist())
            bp_agg_score = torch.tensor([torch.sum(value) for value in score])
        else:
            raise NotImplementedError(f'{model_type} is not implemented yet.')
        return (fp_agg_score, bp_agg_score)

    def convert_send_idx(self, original_send_idx: Basic_Buffer_Type) -> Tuple[Dict[int, Tuple[int, int]], Tensor]:
        '''
        计算在每个分区i上，send_idx的起止位置(converted_send_idx[i])和具体的ID(total_idx[:xxx])
        注意：此函数应在图重新排序后调用。
        '''
        offset = 0
        converted_send_idx: Dict[int, Tuple[int, int]] = {}
        total_idx = []
        # 计算发往其他每个分区的节点的起止ID范围
        for k, v in original_send_idx.items():
            converted_send_idx[k] = (offset, offset + len(v))
            offset += len(v)
            total_idx.append(v)
        total_idx = torch.cat(total_idx) if len(total_idx) else total_idx
        # converted_send_idx只记录起止范围, total_idx记录实际的ID
        return converted_send_idx, total_idx

    def _reorder(self,
                 input_graph: DGLHeteroGraph,
                 nodes_feats: Dict[str, Tensor],
                 send_idx: Basic_Buffer_Type,
                 num_inner: int,
                 num_central: int,
                 m_mask: torch.Tensor,
                 c_mask: torch.Tensor):
        # 获取索引
        c_indices = c_mask.nonzero(as_tuple=True)[0][:num_central]
        m_indices = m_mask.nonzero(as_tuple=True)[0][:num_inner - num_central]

        # 生成新编号映射
        new_id = torch.full((num_inner,), -1, dtype=torch.long)  # 初始化为 -1，防止未赋值
        new_id[c_indices] = torch.arange(num_central, dtype=torch.long)
        new_id[m_indices] = torch.arange(num_central, num_inner, dtype=torch.long)

        # 确保所有 num_inner 内的节点都被正确映射
        assert (new_id >= 0).all(), "new_id 赋值不完整，可能有未定义索引！"

        # 重新编号边索引
        u, v = input_graph.edges()
        mask_u = u < num_inner
        mask_v = v < num_inner
        u[mask_u] = new_id[u[mask_u].long()]
        v[mask_v] = new_id[v[mask_v].long()]

        # 构造新的 DGL 图
        reordered_graph = dgl.heterograph(
            {etype: (u, v) for etype in input_graph.etypes}) if input_graph.is_hetero else dgl.graph((u, v))

        # 重新排列节点特征
        for key in nodes_feats:
            new_feats = torch.zeros_like(nodes_feats[key][:num_inner])
            new_feats[new_id] = nodes_feats[key][:num_inner]
            nodes_feats[key] = new_feats

        # 重新映射 send_idx
        for key in send_idx:
            mask = send_idx[key] < num_inner
            send_idx[key][mask] = new_id[send_idx[key][mask]]

        return reordered_graph, nodes_feats, send_idx

    def _reorder2(self,
                 input_graph: DGLHeteroGraph,
                 nodes_feats: Dict[str, Tensor],
                 send_idx: Basic_Buffer_Type,
                 num_inner: int,
                 num_central: int,
                 m_mask,
                 c_mask):
        '''
        reorder local nodes and return new graph and feats dict
        - 在原graph中，中心节点的ID(如[0,2,5,6,7])和边缘节点的ID(如[1,3,4,8,9])可能是不连续的，这里对他们进行重排以便后续处理。
        - 具体来说，中心节点的ID范围排在前面(如0~10)，边缘节点的ID范围排在后面(如11~20), 对于HALO节点不处理(如21~50)。
        - 注意，只是ID范围变了，而他们的位置没有变，还是交错的(如[0,1,11,12,2,13,3,14,15,16,4,...])。
        '''
        # 生成一个序列用于记录节点ID
        new_id = torch.zeros(size=(num_inner,), dtype=torch.long)
        # 对于 c_mask 中的前 num_inner 个元素，将 new_id 对应位置设为从 0 到 num_central-1 的值
        # c_mask和m_mask均包含内部节点和HALO节点，因此需要取前num_inner个，即都是内部节点
        new_id[c_mask[:num_inner]] = torch.arange(num_central, dtype=torch.long)
        # 对于 m_mask 中的前 num_inner 个元素，将 new_id 对应位置设为从 num_central 到 num_inner-1 的值
        new_id[m_mask[:num_inner]] = torch.arange(num_central, num_inner, dtype=torch.long)
        # 获取输入图的边（u 和 v）
        u, v = input_graph.edges()
        # 对于小于 num_inner 的 u 和 v 值，使用 new_id 进行替换。
        # 因为graph中的u、v包含了HALO节点，因此小于num_inner是为了只取内部节点。
        # 重新映射内部节点ID到new_id的范围内, 对于HALO节点不处理。
        u[u < num_inner] = new_id[u[u < num_inner].long()]
        v[v < num_inner] = new_id[v[v < num_inner].long()]
        # 使用更新后的 u 和 v 创建一个新的图
        reordered_graph = dgl.graph((u, v))
        # 遍历 nodes_feats 中的每个键，将其特征根据 new_id 进行重新排列。
        # 由于new_id中节点的位置没有变，只是他的值变了，所以左边的位置跟右边的位置对应，可以直接赋值。
        for key in nodes_feats: nodes_feats[key][new_id] = nodes_feats[key].clone()[0:num_inner]
        # 遍历 send_idx 中的每个键，按照 new_id 对其进行重新排列
        for key in send_idx: send_idx[key] = new_id[send_idx[key]]
        return reordered_graph, nodes_feats, send_idx

    def reorder_graph(self, graph_info):
        # 将图的节点和特征重新排序为内部节点、边缘节点和中心节点。
        #   marginal nodes: 边缘节点, 在其他设备上具有远程邻居的节点;
        #   central nodes: 中心节点, 没有远程邻居的节点. 即内部节点中不是边缘节点的部分;
        for rank in range(self.partition_size):
            g_info = graph_info[rank]
            original_graph: DGLHeteroGraph = g_info.graph
            nodes_feats: Dict[str, Tensor] = g_info.node_feats
            send_idx: Basic_Buffer_Type = g_info.send_idx

            # 获取内部节点掩码(包含了HALO节点，但其值是False)
            inner_mask = nodes_feats['inner_node']
            # 计算值为True的数量，也就是真正的内部节点的数量
            num_inner = torch.count_nonzero(inner_mask).item()
            # 获取远程节点。除了内部节点，剩下的都是远程节点(也就是取出HALO节点)
            halo_remote_nodes = original_graph.nodes()[~inner_mask]
            # 通过获取远程节点的出边, 来提取内部边缘节点marginal_nodes
            _, v = original_graph.out_edges(halo_remote_nodes)
            # 可能有重复，即指向了同一个目标节点，需要去重
            marginal_nodes = torch.unique(v)
            # 标记边缘节点和中心节点(即内部节点中不是边缘节点的部分)，包含HALO节点
            marginal_mask = torch.zeros_like(inner_mask, dtype=torch.bool)
            marginal_mask[marginal_nodes] = True
            # 内部节点中，不是marginal_nodes的都认为是central_nodes;
            # 为了长度统一，因此拼接了HALO节点，不过都是False
            central_mask = torch.concat([~marginal_mask[:num_inner], marginal_mask[num_inner:]])
            num_marginal = torch.count_nonzero(marginal_mask).item()
            num_central = torch.count_nonzero(central_mask).item()

            # 让中心节点的ID范围排在前面(如0~10)，边缘节点的ID范围排在后面(如11~20), 对于HALO节点不处理(如21~50)。
            # reordered_graph, reordered_feats, reordered_send_idx = self._reorder(original_graph, nodes_feats,
            #                                                                      send_idx, num_inner, num_central,
            #                                                                      marginal_mask, central_mask)
            reordered_graph, reordered_feats, reordered_send_idx = original_graph, nodes_feats, send_idx

            reordered_graph.ndata['in_degrees'] = reordered_feats['in_degrees']
            reordered_graph.ndata['out_degrees'] = reordered_feats['out_degrees']
            num_inner = torch.count_nonzero(reordered_feats['inner_node']).item()
            num_remote = reordered_graph.num_nodes() - num_inner
            assert reordered_graph.num_nodes() == original_graph.num_nodes()
            assert reordered_graph.num_edges() == original_graph.num_edges()
            assert num_inner == num_central + num_marginal
            # 计算在每个远程分区i上，send_idx的起止位置和具体的ID.
            # 如: 对于当前是分区1来说，有send_idx={0: (0, 17660), 2: (17660, 36595)} total_idx=[xx,xx,xx,...,xx]
            send_idx, total_idx = self.convert_send_idx(reordered_send_idx)

            g_info.graph = reordered_graph
            g_info.node_feats = nodes_feats
            g_info.send_idx = send_idx
            g_info.total_send_idx = total_idx
            g_info.num_inner = num_inner
            g_info.num_remote = num_remote
            g_info.num_marginal = num_marginal
            g_info.num_central = num_central
        return graph_info

    def assignment_graphs(self, gpus, graph_info):
        '''
        将GPU与分区绑定。注意需要按照算力从弱到强依次跟图绑定，即self.graph_info[0]绑定的是最弱的GPU。
        子图的分区ID与sorted_gpu_ids的索引对应，与GPU ID没有关系。
        '''
        assert len(gpus.split(',')) == len(graph_info), ">> GPU数与分区数不相同 <<"
        gpu_list = list(map(int, gpus.split(',')))
        # 按指标值降序排序，并获取对应的GPU设备key。值越大，计算成本越大，GPU越弱。
        self.sorted_gpu_ids = get_gpu_capability(gpu_list)
        # 首次是随便指定GPU和graph
        for i, g in enumerate(graph_info):
            print(f"<分配 {torch.cuda.get_device_name(self.sorted_gpu_ids[i])}[{self.sorted_gpu_ids[i]}] 给 subgraph {i}>")
            g.device = torch.device(f"cuda:{self.sorted_gpu_ids[i]}")
            # g.num_remote = torch.count_nonzero(g.graph.nodes_feats['outer_node']).item()
            self.assig_graphs_gpus[self.sorted_gpu_ids[i]] = g
        return self.assig_graphs_gpus

    def check_graph_connectivity(self, graph):
        # 将 DGL 图转换为 NetworkX 图
        nx_graph = graph.to_networkx().to_undirected()
        # 使用 NetworkX 检查连通性
        if nx.is_connected(nx_graph):
            print("The graph is connected.")
            return True
        else:
            connected_components = list(nx.connected_components(nx_graph))
            print(f"The graph is not connected. It has {len(connected_components)} components.")
            print("Sizes of each component:", [len(comp) for comp in connected_components])
            return False

    def calculate_Memory(self,  g_info=None, beta=0, num_nodes=None, num_edges=None, feat=None):
        Vi = num_nodes or g_info.graph.num_nodes()
        Ei = num_edges or g_info.graph.num_edges()
        di = feat or g_info.node_feats['feat']
        Mi = (Vi * 4 + Ei * 4 * 2 + di.shape[0] * di.shape[1] * 4 + beta * 4)/1024/1024
        return Mi

    def calculate_T_comp_bk(self, gpu_id, g_info=None, alpa=0.5, num_nodes=None, num_edges=None):
        Vi = num_nodes or g_info.num_inner  # g_info.graph.num_nodes()
        Ei = num_edges or g_info.graph.num_edges()
        s_ms, m_mm = gpu_capability[gpu_id][1:3]
        # T_comp_i = alpa*(Vi*Vi-2*Ei)*s_ms + (1-alpa)*Vi*Vi*m_mm
        # T_comp_i = alpa*(2*Ei)*s_ms + (1-alpa)*Vi*Vi*m_mm
        T_comp_i = Ei*s_ms
        return T_comp_i

    def calculate_T_comp(self, gpu_id, g_info=None, alpa=0.35, num_nodes=None, num_edges=None):
        Vi = num_nodes or g_info.num_inner  # g_info.graph.num_nodes()
        Ei = num_edges or g_info.graph.num_edges()
        s_ms, m_mm = gpu_capability[gpu_id][1:3]
        T_comp_i = alpa*Ei*s_ms + (1-alpa)*Vi*m_mm
        # T_comp_i = Ei*s_ms
        return T_comp_i

    def calculate_T_comm(self, gpu_id, g_info=None, E_outer=None):
        # return 0
        n = len(self.assig_graphs_gpus.keys())
        # E_outer = E_outer or g_info.num_remote
        H2D_m, D2H_m, D2D_m = gpu_capability[gpu_id][3:6]
        T_comp_i = E_outer * ((H2D_m+D2H_m)*(1-1/n)+D2D_m*1/n)
        return T_comp_i

    def calculate_target(self, assig_graphs_gpus):
        T_comp = []
        T_comm = []
        mem_usage = []
        halo_edges = []
        for gpu_id in self.sorted_gpu_ids:
            g_info = assig_graphs_gpus[gpu_id]
            nodes_feats = g_info.node_feats
            part_id = g_info.gpb.partid
            halo_local_masks = (nodes_feats['part_id'] != part_id)
            halo_local_node_ids = g_info.graph.nodes()[halo_local_masks]
            edges_set = set()
            node_edges = g_info.graph.out_edges(halo_local_node_ids, form='eid').tolist()
            for edge_id in node_edges:
                if edge_id not in edges_set:
                    edges_set.add(edge_id)
            halo_edges.append(len(edges_set))
            T_comp.append(self.calculate_T_comp(gpu_id, g_info=g_info))
            T_comm.append(self.calculate_T_comm(gpu_id, E_outer=len(edges_set)))
            mem_usage.append(self.calculate_Memory(g_info=assig_graphs_gpus[gpu_id]))
        lam = [p + m for p, m in zip(T_comp, T_comm)]
        lam_std = statistics.stdev(lam)
        lam_mean = statistics.mean(lam)
        return lam, lam_mean, lam_std, halo_edges

    def get_halo_count(self):
        temp = Counter()
        for rank in range(self.partition_size):
            part_config = f'{self.partition_dir}/{self.dataset_name}/{self.partition_size}part/{self.dataset_name}.json'
            g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
            halo_nodes = g.nodes()[~g.ndata['inner_node'].bool()].numpy()
            halo_global_ids = g.ndata[dgl.NID][halo_nodes].numpy()
            temp.update(halo_global_ids)
        self.global_id_counter = [np.fromiter(temp.keys(), dtype=np.int32), np.fromiter(temp.values(), dtype=np.int32)]
        return self.global_id_counter

    def adjust_once(self, assig_graphs_gpus):
        result = [False for _ in range(len(assig_graphs_gpus))]
        records = []
        # 4. 从最弱的GPU开始选起
        for index in range(len(self.sorted_gpu_ids)):
            gpu_id = self.sorted_gpu_ids[index]
            # 计算λ及其平均值λ_mean
            lam, lam_mean, lam_std, halo_edges = self.calculate_target(assig_graphs_gpus)
            initial_halo_edges = halo_edges[index]
            print(f"gpu_id: {gpu_id}[{index}], lam - mean: {lam[index] - lam_mean}[{lam[index]} - {lam_mean}]")

            # 找到该分区上所有的halo节点
            g_info = assig_graphs_gpus[gpu_id]
            local_graph = g_info.graph
            nodes_feats = g_info.node_feats
            gpb = g_info.gpb
            part_id = gpb.partid  # 该graph所属的分区ID

            # 选择λ_i大于λ_mean的GPU。小于等于的先跳过
            nodes = local_graph.num_nodes()
            edges = local_graph.num_edges()
            if lam[index] <= lam_mean and gpu_memory_enough(gpu_id=gpu_id, num_nodes=nodes, num_edges=edges, feats=nodes_feats["feat"], beta=1e8):
                result[index] = True
                print(f"分区{part_id}无变化")
                print(f"节点数无变化： {g_info.graph.num_nodes()}")
                print(f"边数的无变化： {g_info.graph.num_edges()}")
                print(f"分数的无变化： {lam[index]}")
                records.append([part_id, g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index], '=>', g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index]])
                continue

            halo_local_offset_start = len(torch.nonzero(nodes_feats['inner_node']))
            halo_local_masks = (nodes_feats['part_id'] != part_id)
            halo_nodes = local_graph.nodes()[halo_local_masks]
            # 获取halo节点在当前分区上的索引
            halo_local_idxs = torch.nonzero(halo_local_masks).view(-1)
            halo_global_ids = nodes_feats[dgl.NID][halo_local_idxs]


            # 计算这些halo节点的聚合分数
            # agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            # sorted_scores, ori_idxs = torch.sort(agg_scores[0] + agg_scores[1], descending=False)

            agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            indices = torch.nonzero(torch.isin(torch.tensor(self.global_id_counter[0]), halo_global_ids)).view(-1)
            scores = (agg_scores[0] + agg_scores[1]) * torch.tensor(self.global_id_counter[1])[indices]
            sorted_scores, ori_idxs = torch.sort(scores, descending=False)

            # 选出分数最高的一个
            nodes_to_remove_id = []
            nodes_to_remove_idx = []
            # 初始的剩余节点数和边数
            remaining_inner_nodes = torch.count_nonzero(nodes_feats['inner_node']).item()
            remaining_all_nodes = local_graph.num_nodes()
            remaining_edges = local_graph.num_edges()
            initial_halo_nodes_count = torch.count_nonzero(halo_local_masks)
            # 跟踪已删除的边以避免重复计数
            removed_edges_set = set()
            # 预估移除节点和边数的影响
            estimated_lam_reduction = 0

            for i, (score, idx) in enumerate(zip(sorted_scores, ori_idxs)):
                max_score_idx = idx.item()
                halo_local_node_id = halo_nodes[max_score_idx].item()
                # 获取节点的边并考虑重复
                max_score_node_edges = local_graph.out_edges(halo_local_node_id, form='eid').tolist()
                for edge_id in max_score_node_edges:
                    if edge_id not in removed_edges_set:
                        removed_edges_set.add(edge_id)
                        remaining_edges -= 1
                # 检查该节点是否是内点，只有在是内点时才减少 remaining_inner_nodes
                # if nodes_feats['inner_node'][halo_local_node_id].item(): remaining_inner_nodes -= 1
                remaining_all_nodes -= 1
                remaining_halo_nodes = initial_halo_nodes_count - len(nodes_to_remove_id)
                remaining_halo_edges = initial_halo_edges - len(removed_edges_set)  # len(removed_edges_set.intersection(max_score_node_edges))

                # 估算移除后的 T_comp 和 T_comm
                t1 = self.calculate_T_comp(gpu_id, num_nodes=remaining_inner_nodes, num_edges=remaining_edges)
                t2 = self.calculate_T_comm(gpu_id, E_outer=remaining_halo_edges)
                estimated_lam_reduction = t1 + t2
                # 如果预估的lambda变化接近lam_mean - lam[index]，则停止选择
                if estimated_lam_reduction <= (lam[index] + lam_mean)/2 and gpu_memory_enough(gpu_id=gpu_id, num_nodes=remaining_all_nodes, num_edges=remaining_edges, feats=nodes_feats["feat"], beta=1e8): break
                nodes_to_remove_id.append(halo_local_node_id)
                nodes_to_remove_idx.append(halo_local_idxs[max_score_idx])
            # 删除选定的节点
            if len(nodes_to_remove_id) > 0:
                print(f"为分区{part_id}删除了{len(nodes_to_remove_id)}个节点")
                new_graph = dgl.remove_nodes(local_graph, torch.tensor(nodes_to_remove_id))
                new_node_feats = {key: value[~torch.isin(torch.arange(value.size(0)), torch.tensor(nodes_to_remove_idx))] for key, value in nodes_feats.items()}
                print(f"节点数变化： {g_info.graph.num_nodes()} => {new_graph.num_nodes()}")
                print(f"边数的变化： {g_info.graph.num_edges()} => {new_graph.num_edges()}")
                print(f"分数的变化： {lam[index]} => {estimated_lam_reduction}")
                records.append([part_id, g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index], '=>', new_graph.num_nodes(), new_graph.num_edges(), estimated_lam_reduction])
                g_info.graph = new_graph
                g_info.node_feats = new_node_feats
                g_info.num_inner = torch.sum(new_node_feats['inner_node']).item()
                g_info.send_idx = None
                g_info.recv_idx = None
                g_info.agg_scores = None
                g_info.num_remote = None
                g_info.num_marginal = None
                g_info.num_central = None
                assig_graphs_gpus[gpu_id] = g_info
            else:
                result[index] = True
                print(f"分区{part_id}无变化")
                print(f"节点数无变化： {g_info.graph.num_nodes()}")
                print(f"边数的无变化： {g_info.graph.num_edges()}")
                print(f"分数的无变化： {lam[index]}")
                records.append([part_id, g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index], '=>', g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index]])

            # 计算并对比目标值，若仍大于，则继续删除
            print('*' * 50)
        return assig_graphs_gpus, result, records

    def do_partition(self, assig_graphs_gpus, threshold=0.01):
        if os.path.exists('records.csv'): os.remove('records.csv')
        with open('records.csv', 'a+', newline='') as csvfile:
            csv.writer(csvfile).writerow(['part_id', 'Nodes', 'Edges', 'Scores', 'Nodes_new', 'Edges_new', 'Scores_new'])
        while True:
            assig_graphs_gpus, result, records = self.adjust_once(assig_graphs_gpus)
            with open('records.csv', 'a+', newline='') as csvfile:
                csv.writer(csvfile).writerows(records)
            print('#'*80)
            lam, lam_mean, lam_std, _ = self.calculate_target(assig_graphs_gpus)
            if lam_std < lam_mean * threshold: break
            if all(result): break

        print('done!')
        print('')
        print('*' * 50)

        # 效果信息输出
        lam_prev, lam_mean_prev, lam_std_prev, _ = self.calculate_target(self.assig_graphs_gpus)
        lam, lam_mean, lam_std, _ = self.calculate_target(assig_graphs_gpus)
        for index in range(len(self.sorted_gpu_ids)):
            gpu_id = self.sorted_gpu_ids[index]
            g_info_prev = self.assig_graphs_gpus[gpu_id]
            g_info = assig_graphs_gpus[gpu_id]
            print(f"GPU: {gpu_id}")
            print(f"分区: {g_info.gpb.partid}")
            print(f"节点数: {g_info_prev.graph.num_nodes()}[in:{g_info_prev.num_inner}] => {g_info.graph.num_nodes()}[in:{g_info.num_inner}]")
            print(f"边数: {g_info_prev.graph.num_edges()} => {g_info.graph.num_edges()}")
            print(f"lam: {lam_prev[index]} => {lam[index]}")
            print(f"lam_mean: {lam_mean_prev} => {lam_mean}")
            print(f"lam_std: {lam_std_prev} => {lam_std}")
            # self.check_graph_connectivity(g_info.graph)
            print('-' * 40)
        print('*' * 50)

        return assig_graphs_gpus

    def reduce_halo_by_memory(self, assig_graphs_gpus):
        for index in range(len(self.sorted_gpu_ids)):
            gpu_id = self.sorted_gpu_ids[index]
            g_info = assig_graphs_gpus[gpu_id]
            local_graph = g_info.graph
            nodes_feats = g_info.node_feats
            gpb = g_info.gpb
            part_id = gpb.partid
            nodes = local_graph.num_nodes()
            edges = local_graph.num_edges()
            if gpu_memory_enough(gpu_id=gpu_id, num_nodes=nodes, num_edges=edges, feats=nodes_feats["feat"], beta=1e8):
                print(f"分区{part_id}无需删除节点")
                continue

            halo_local_masks = (nodes_feats['part_id'] != part_id)
            # 获取halo节点在当前分区上的索引
            halo_local_idxs = torch.nonzero(halo_local_masks).view(-1)
            # 计算这些halo节点的聚合分数
            agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            # 简单起见，先用forward_score，之后再考虑forward+backward。对于无向图，两者应该是一样的。
            forward_scores = agg_scores[0]
            # backward_scores = agg_scores[1]
            scores = forward_scores  # + backward_scores
            sorted_scores, ori_idxs = torch.sort(scores, descending=False)

            # 选出分数最高/低的一个
            nodes_to_remove = []
            remaining_all_nodes = local_graph.num_nodes()
            remaining_edges = local_graph.num_edges()
            # 跟踪已删除的边以避免重复计数
            removed_edges_set = set()
            initial_halo_nodes_count = len(halo_local_idxs)
            # 预估移除节点和边数的影响
            for i, (score, idx) in enumerate(zip(sorted_scores, ori_idxs)):
                max_score_idx = idx.item()
                halo_local_node_id = halo_local_idxs[max_score_idx].item()
                max_score_node_edges = local_graph.out_edges(halo_local_node_id, form='eid').tolist()
                for edge_id in max_score_node_edges:
                    if edge_id not in removed_edges_set:
                        removed_edges_set.add(edge_id)
                        remaining_edges -= 1
                remaining_all_nodes -= 1

                if gpu_memory_enough(gpu_id, num_nodes=remaining_all_nodes, num_edges=remaining_edges, feats=nodes_feats["feat"], beta=1e8): break
                nodes_to_remove.append(halo_local_node_id)
                remaining_halo_nodes = initial_halo_nodes_count - len(nodes_to_remove)
                if remaining_halo_nodes <= 0: break

            # 删除选定的节点
            if nodes_to_remove:
                print(f"为分区{part_id}删除了{len(nodes_to_remove)}个节点")
                nodes_to_remove_tensor = torch.tensor(nodes_to_remove)
                new_graph = dgl.remove_nodes(local_graph, nodes_to_remove_tensor)
                new_node_feats = {key: value[~torch.isin(torch.arange(value.size(0)), nodes_to_remove_tensor)] for
                                  key, value in nodes_feats.items()}
                print(f"节点数变化：{g_info.graph.num_nodes()} => {new_graph.num_nodes()}")
                print(f"边数的变化：{g_info.graph.num_edges()} => {new_graph.num_edges()}")
                g_info.graph = new_graph
                g_info.node_feats = new_node_feats
                g_info.num_inner = torch.sum(new_node_feats['inner_node']).item()
                assig_graphs_gpus[gpu_id] = g_info
            # 计算并对比目标值，若仍大于，则继续删除
            print('*' * 50)
        return assig_graphs_gpus

    def overlapping_our(self, part_size, dataset, gpus_list, part_dir='data/part_data', k=-1, our_partition=False):
        max_subg_size = 0
        all_part_nodes_feats = []
        halo_node_features = {}
        model_type = DistGNNType.DistGCN
        # 加载pkl文件  /home/sxf/Desktop/gnn/dist_gnn_fullbatch
        partition_file = f'{part_dir}/{dataset}/{part_size}part/{dataset}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl'
        with open(partition_file, 'rb') as f:
            assig_graphs_gpus = pickle.load(f)

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

                agg_scores = self._get_agg_scores(g, halo_local_masks, nodes_feats, model_type)
                _, ori_idxs = torch.sort(agg_scores[0] + agg_scores[1], descending=True)

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
                print(f"[{part_id}-{i}]本次添加{len(temp.keys())}/{len(halo_global_ids)}个halo节点进来")
        return halo_node_features, max_subg_size


def auto_select_dtype(feat_tensor):
    """
    根据节点特征张量的实际内容，自动选择合适的数据类型。

    参数:
    feat_tensor (torch.Tensor): 节点特征张量 (float32 类型)

    返回:
    torch.dtype: 推荐的数据类型
    """
    # 检查所有元素是否可以看作是整数
    # 计算小数部分并判断其是否接近 0
    if torch.all((feat_tensor - torch.round(feat_tensor)) == 0):
        # 如果所有数都接近整数，则转换为整数类型
        max_val = torch.max(feat_tensor)
        min_val = torch.min(feat_tensor)
        if min_val >= torch.iinfo(torch.int8).min and max_val <= torch.iinfo(torch.int8).max:
            return torch.int8
        elif min_val >= torch.iinfo(torch.int16).min and max_val <= torch.iinfo(torch.int16).max:
            return torch.int16
        elif min_val >= torch.iinfo(torch.int32).min and max_val <= torch.iinfo(torch.int32).max:
            return torch.int32
        else:
            return torch.int64
    else:
        # 如果是浮点数，根据浮点数的范围和标准差来判断
        min_val = torch.min(feat_tensor)
        max_val = torch.max(feat_tensor)
        # float16 的数值范围是大约 6.1e-5 到 65504
        if min_val >= torch.iinfo(torch.float16).min and max_val <= torch.iinfo(torch.float16).max:
            std = torch.std(feat_tensor)
            # 如果特征的范围较小，且精度要求不高，则可以用 float16
            return torch.float16 if std < 1e-2 else torch.float32
        else:
            return torch.float32

def partition1():
    parser = argparse.ArgumentParser(description='graph partition scripts')
    parser.add_argument('--dataset_dir', type=str, default='/mnt/disk/sxf/data/dataset')
    parser.add_argument('--partition_dir', type=str, default='/mnt/disk/sxf/data/part_data')
    parser.add_argument("-p", "--our_partition", type=int, default=1, help="Set our_partition to True (1) or False (0).")
    parser.add_argument("-n", "--part_num", type=int, help="Set the number of partitions.")
    parser.add_argument("-d", "--dataset_index", type=int, help="Set the dataset index.")
    parser.add_argument("-g", "--gpus_index", type=int, help="Set the dataset index.")
    args = parser.parse_args()
    set_random_seeds(42)

    dataset_groups = ['ogbn-arxiv',             # 0 
                      'ogbn-products',          # 1
                      'cite',                   # 2    
                      'cora',                   # 3
                      'flickr',                 # 4
                      'yelp',                   # 5
                      'reddit',                 # 6
                      'amazonProducts',         # 7
                      'amazonCoBuyComputer',    # 8
                      'coauthorPhysics',        # 9
                      'coraFull',               # 10
                      'tolokers',               # 11
                    ]
    gpu_groups = {
        '228': [
            # 0        1        2        3        4        5        6        7         8        9
            [],
            ['0', ],
            ['0,2', '0,4', '0,1'],
            ['0,2,4', ],
            ['4,6,0,2', '0,1,2,3', '1,3,4,6'],
            ['0,1,2,4,6', ],
            ['3,1,6,4,0,2'],
            ['5,3,1,6,4,0,2'],
            ['5,7,3,1,6,4,0,2'],
        ],
        '229': [
            # 0        1        2        3        4        5        6        7         8        9
            [],
            ['2', ],
            ['2,3', ],
            ['2,3,7', ],
            ['2,3,4,7', ]
        ]
    }

    # python cspart.py --dataset_index=4 --part_num=5 --our_partition=1
    # python cspart.py -p=1 -d=6 -n=4 -g=0
    # ------------参数主要修改区---------------- #
    test_load_part          = False
    threshold               = 0.01
    dataset_index           = args.dataset_index    or 6
    part_num                = args.part_num         or 4
    gpus_index              = args.gpus_index       or 0
    our_partition           = args.our_partition    # or 1
    part_method             = ['metis', 'random'][0]
    # --------------------------------------- #
    our_partition           = [False, True][our_partition]
    args.partition_size     = part_num
    args.dataset_name       = dataset_groups[dataset_index]
    gpus                    = gpu_groups[tools.outer_ip][part_num][gpus_index]
    gpus_list = list(map(int, gpus.split(',')))
    cal_gpus_capability(gpus_list)
    print(f'>> 数据集: {args.dataset_name}')
    print(f'>> GPU列表: [{gpus_index}] => {gpus}')
    print(f'>> 分区阈值: {threshold}')
    print(f'>> 分区数量: {part_num}')
    print(f'>> 分区方法: {part_method}')
    print(f'>> 分区策略: {our_partition}')

    args = vars(args)
    app = CSPart(args)
    # 1. 使用现成的分区算法进行预先分区；
    app.coarse_graph_patition(fast_skip=True, part_method=part_method, num_hops=1)
    # 2. 构建局部加权图，即只为边缘的节点计算加权值；
    # 加载分区的子图和特征
    app.coarse_load_patition()
    # dtype = auto_select_dtype(app.graph_info[0].node_feats['feat'])
    # print('>> 原来的的数据类型：', app.graph_info[0].node_feats['feat'].dtype)
    # print('>> 合适的数据类型：', dtype)
    # 2.1 获取每个分区的发送、接收的顶点ID，以及边缘节点的度分数
    app.get_send_recv_idx_scores(app.graph_info, skip_load=False, suffix='_init')
    # 2,2 重新排列图中节点ID的顺序（从 0 到 N-1：中心节点->边缘节点->边远节点
    app.reorder_graph(app.graph_info)
    # app.print_graph(0)
    # 3. 为每个GPU预先分配一个分区
    assig_graphs_gpus = app.assignment_graphs(gpus, app.graph_info)
    app.get_halo_count()
    new_assig_graphs_gpus = copy.deepcopy(assig_graphs_gpus)
    if our_partition and len(gpus.split(',')) > 1: new_assig_graphs_gpus = app.do_partition(new_assig_graphs_gpus, threshold=threshold)
    else: print(">> 不需要分区")
    print('=' * 100)

    new_graph_info = []
    for index in range(len(app.sorted_gpu_ids)): new_graph_info.append(new_assig_graphs_gpus[app.sorted_gpu_ids[index]])
    app.get_send_recv_idx_scores(new_graph_info, skip_load=True)
    app.reorder_graph(new_graph_info)
    for index in range(len(app.sorted_gpu_ids)): assig_graphs_gpus[app.sorted_gpu_ids[index]] = new_graph_info[index]

    # 保存分区结果
    partition_file = os.path.join(app.partition_dir, f'{app.dataset_name}/{app.partition_size}part/{app.dataset_name}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl')
    with open(partition_file, 'wb') as f: pickle.dump(assig_graphs_gpus, f)
    print(f"Processed partitions saved to {partition_file}")

    # 加载pkl文件
    if test_load_part:
        print("测试加载pkl文件")
        with open(partition_file, 'rb') as f: _ = pickle.load(f)
    print("=" * 100)
    print(f"训练时，请务必按照以下的GPU顺序设置CUDA_VISIBLE_DEVICES：{','.join(map(str, app.sorted_gpu_ids))}")

    halo_node_features, _ = app.overlapping_our(part_size=args['partition_size'], dataset=args['dataset_name'], gpus_list=gpus_list, k=-1, part_dir='/mnt/disk/sxf/data/part_data', our_partition=our_partition)
    print(f"最终有效的节点数：[{args['dataset_name']}][{our_partition}][{part_num}][{gpus_index}] => {len(halo_node_features.keys())}")
    del app
    torch.cuda.empty_cache()
    print('#' * 50)



# 计算halo节点重复度
def overlapping(dataset_name, partition_size):
    partition_dir = './data/part_data'
    part_config = f'{partition_dir}/{dataset_name}/{partition_size}part/{dataset_name}.json'

    # 存储每个分区的 halo 节点全局 ID 和总节点数
    halo_global_ids_by_partition = defaultdict(set)
    total_nodes_by_partition = {}
    # in_degrees_by_partition = defaultdict(list)
    # out_degrees_by_partition = defaultdict(list)

    # 循环读取每个分区的 halo 节点
    for rank in range(partition_size):
        g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
        # 获取halo节点（inner_node 为 False 的节点）
        halo_nodes = g.nodes()[~g.ndata['inner_node'].bool()].numpy()
        # 使用 'dgl.NID' 来获取 halo 节点的全局 ID
        halo_global_ids = g.ndata[dgl.NID][halo_nodes].numpy()
        # 存储该分区的 halo 节点全局 ID
        halo_global_ids_by_partition[rank] = set(halo_global_ids)
        # 存储该分区的总节点数
        total_nodes_by_partition[rank] = g.num_nodes()
        # 获取该分区 halo 节点的原始 ID
        orig_ids = g.ndata['orig_id'][halo_nodes].numpy()
        # 存储入度和出度
        # in_degrees_by_partition[rank] = in_degrees_global[orig_ids].numpy()
        # out_degrees_by_partition[rank] = out_degrees_global[orig_ids].numpy()
        
    # 输出每个分区的 halo 节点数量和总节点数量
    for rank in range(partition_size):
        halo_count = len(halo_global_ids_by_partition[rank])
        total_count = total_nodes_by_partition[rank]
        # avg_in_degree = in_degrees_by_partition[rank].mean() if halo_count > 0 else 0
        # avg_out_degree = out_degrees_by_partition[rank].mean() if halo_count > 0 else 0
        # print(f"Part {rank} halo nodes: {halo_count}/{total_count}, Avg In-Degree: {avg_in_degree:.2f}, Avg Out-Degree: {avg_out_degree:.2f}")
        print(f"Part {rank} halo nodes: {halo_count}/{total_count}")

    # 存储总的重合halo节点
    total_overlapping_halo_nodes_set = set()
    total_overlapping_halo_nodes = 0
    # 比较各分区之间的 halo 节点重复情况并输出结果
    for i in range(partition_size):
        for j in range(i + 1, partition_size):
            # 获取两个分区的 halo 节点
            halo_i = halo_global_ids_by_partition[i]
            halo_j = halo_global_ids_by_partition[j]
            # 计算重合的 halo 节点数
            overlapping_halo_nodes = halo_i.intersection(halo_j)
            total_overlapping_halo_nodes += len(overlapping_halo_nodes)
            # 更新总的重合halo节点集合
            total_overlapping_halo_nodes_set.update(overlapping_halo_nodes)
            # 输出结果
            print(f"Part {i} - Part {j} -> Overlapping halo nodes: {len(overlapping_halo_nodes)}")

    print('*' * 50)
    print(f"Total overlapping halo nodes across all partitions: {total_overlapping_halo_nodes}")
    # 输出总的重合halo节点数
    print(f"Total overlapping halo nodes across all partitions (set): {len(total_overlapping_halo_nodes_set)}")




if __name__ == '__main__':
    set_random_seeds(42)
    ## only partition
    # parser = argparse.ArgumentParser(description='graph partition scripts')
    # parser.add_argument('--dataset_name', type=str, default='ogbn-products', help='training dataset')
    # parser.add_argument('--dataset_dir', type=str, default='data/dataset')
    # parser.add_argument('--partition_dir', type=str, default='data/part_data')
    # parser.add_argument('--partition_size', type=int, default=4)
    # parser.add_argument('--num_hops', type=int, default=1)
    # args = parser.parse_args()
    # args = vars(args)
    # app = CSPart(args)
    # app.coarse_graph_patition(fast_skip=False, part_method=['metis', 'random'][1], num_hops=args['num_hops'])
    # overlapping(args['dataset_name'], args['partition_size'])

    ## cspart method1
    partition1()

