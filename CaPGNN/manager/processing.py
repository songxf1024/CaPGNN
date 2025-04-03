from enum import Enum

import dgl
import networkx as nx
import torch
import logging
from torch import Tensor
from dgl.distributed import GraphPartitionBook
from typing import Dict, Tuple
import numpy as np
from ..communicator import Communicator as comm
from ..communicator import Basic_Buffer_Type

class DistGNNType(Enum):
    DistGCN = 0
    DistSAGE = 1

logger = logging.getLogger('trainer')



def get_send_recv_idx_scores(local_graph: dgl.DGLHeteroGraph, nodes_feats: Dict[str, Tensor], gpb: GraphPartitionBook, part_dir: str, dataset: str, model_type: DistGNNType) -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
    '''
    获取local graph中每个节点的send/recv idx 和 agg 分数。
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    current_partition_dir = f'{part_dir}/{dataset}/{world_size}part/part{rank}'
    except_list = [None for _ in range(world_size)]
    except_info = None
    # try loading send/recv idx from disk
    try:
        send_idx = np.load(f'{current_partition_dir}/send_idx.npy', allow_pickle=True).item()
        recv_idx = np.load(f'{current_partition_dir}/recv_idx.npy', allow_pickle=True).item()
        agg_scores = np.load(f'{current_partition_dir}/agg_scores.npy', allow_pickle=True).item()
    except IOError as e:
        except_info = str(e)
    # check if all processes have loaded send/recv idx successfully
    comm.all_gather_any(except_list, except_info)
    # if not, build send/recv idx and store them to the disk
    if not all([except_info is None for except_info in except_list]):
        fail_idx = [i for i, except_info in enumerate(except_list) if except_info is not None]
        logger.info(f'<worder {fail_idx} failed to load send/recv idx from disk, begin building...>')
        print(f'<worder {fail_idx} failed to load send/recv idx from disk, begin building...>')
        send_idx, recv_idx, agg_scores = _build_store_send_recv_idx_scores(local_graph, nodes_feats, gpb, current_partition_dir, model_type)
    return send_idx, recv_idx, agg_scores


def _build_store_send_recv_idx_scores(local_graph: dgl.DGLHeteroGraph, nodes_feats: Dict[str, Tensor], gpb: GraphPartitionBook, part_dir: str, model_type: DistGNNType) -> Tuple[Basic_Buffer_Type, Basic_Buffer_Type, Basic_Buffer_Type]:
    '''
    构建和存储每个分区的发送和接收索引以及聚合分数，并将这些信息保存到磁盘.
    HALO node.
    temp_send_idx: 字典，表示每个分区的发送索引。
    recv_idx: 字典，表示每个分区的接收索引。
    scores: 字典，表示每个分区的聚合分数。
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    temp_send_idx: Dict[int, Tensor] = {}
    recv_idx: Dict[int, Tensor] = {}
    scores: Dict[int, Tensor] = {}
    # 当前分区rank中内部节点的数量，长度的话，就可以表示外部节点的起始点
    outer_start = len(torch.nonzero(nodes_feats['inner_node']))
    # build recv_idx and temp_idx
    temp_buffer: Dict[int, Tuple(Tensor, Tensor)] = {}
    for i in range(world_size):
        if i != rank:
            # 获取当前分区rank中，属于分区i的节点ID (即 分区i => 分区rank 的HALO节点)
            belong2i = (nodes_feats['part_id'] == i)
            # get forward & backward aggreagtion scores for remote neighbors
            agg_score = _get_agg_scores(local_graph, belong2i, nodes_feats, model_type)
            # 计算属于分区i的这些HALO节点在当前分区rank中，相对于内点的局部ID或者叫偏移。
            # torch.nonzero(belong2i) 是取非零值的索引，由于ID连续，因此就可以看做是节点的局部ID。
            # 因为顺序是 inner node 排在前面，outer node 排在后面，所以可以这么取。
            local_ids = torch.nonzero(belong2i).view(-1) - outer_start
            # 计算远程分区i的起始节点的ID (全局 ID) (不包含HALO节点)
            start = gpb.partid2nids(i)[0].item()
            # 计算在远程分区i上, 相对于远程分区i而言的节点的ID (即 这些节点在分区i上的局部ID或者叫偏移);
            # dgl.NID是个特殊的键，表示要取所有节点的全局ID
            remote_ids = nodes_feats[dgl.NID][belong2i] - start
            # 这里的score可以反映每个HALO节点的重要性
            temp_buffer[i] = (remote_ids, agg_score)
            # 表示后面需要从这些HALO节点接收数据
            recv_idx[i] = local_ids  # data recv from i
            # local_ids表示HALO节点在当前分区的ID(从外点位置开始的)，remote_ids表示HALO节点在所属分区上的ID。
            # 这里的HALO节点来自分区i。
            # 注意，总体来看，ID都是从0开始的。这也方便了如果从对方分区i的角度来看时，这些ID就已经是局部ID，不需要再次进行处理。
    # print recv_idx in debug mode
    for k, v in recv_idx.items():
        logger.debug(f'<worker{rank} recv {len(v)} nodes from worker{k}>')
        print(f'<worker{rank} recv {len(v)} nodes from worker{k}>')
    # build temp_send_idx and scores
    temp_buffer_list = [None for _ in range(world_size)]
    comm.all_gather_any(temp_buffer_list, temp_buffer)
    #
    for i in range(world_size):
        if i is not rank:
            # 检查分区i是否由于当前rank分区有关联的节点。
            # 从分区i的角度来看，这些节点是来自分区rank的HALO节点，是分区i需要recv_idx的。
            # 因此，对于分区rank而言，相对地是send_idx的。
            # 由于在前面处理的时候，remote_ids已经直接表示其在所属分区上的ID，因此这里可以直接赋值，不需要再进行局部ID范围的调整。
            if rank in temp_buffer_list[i].keys():
                temp_send_idx[i] = temp_buffer_list[i][rank][0]   # data from i to rank
                scores[i] = temp_buffer_list[i][rank][1]          # score from i to rank
    # print temp_send_idx in debug mode
    for k, v in temp_send_idx.items():
        logger.debug(f'<worker{rank} send {len(v)} nodes to worker{k}>')
        print(f'<worker{rank} send {len(v)} nodes to worker{k}>')
    # store send_idx, recv_idx and scores to disk
    np.save(f'{part_dir}/send_idx.npy', temp_send_idx)
    np.save(f'{part_dir}/recv_idx.npy', recv_idx)
    np.save(f'{part_dir}/agg_scores.npy', scores)
    return temp_send_idx, recv_idx, scores

def _get_agg_scores(local_graph: dgl.DGLHeteroGraph, belong_mask: Tensor, nodes_feats: Dict[str, Tensor], model_type: DistGNNType) -> Tuple[Tensor, Tensor]:
    '''
    返回本地图中每个节点的前向和后向的聚合得分，这些得分用于评估节点的重要性或影响力。
    '''
    fp_neighbor_ids = local_graph.out_edges(local_graph.nodes()[belong_mask])[1]  # 节点的前向邻居（即，出边的目标节点）
    fp_local_degrees = local_graph.out_degrees(local_graph.nodes()[belong_mask])  # 节点的出度
    bp_neighbor_ids = local_graph.in_edges(local_graph.nodes()[belong_mask])[0]   # 节点的后向邻居（即，入边的源节点）。
    bp_local_degrees = local_graph.in_degrees(local_graph.nodes()[belong_mask])   # 节点的入度。
    if model_type is DistGNNType.DistGCN:
        # 前向聚合得分:
        # - 计算前向邻居的全局度数 fp_global_degrees。
        # - 对每个前向邻居的入度进行开方倒数的计算，并按节点的出度进行分割和求和，得到前向聚合得分 fp_agg_score。
        fp_global_degrees = nodes_feats['out_degrees'][belong_mask]
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -0.5).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(score[i] * torch.pow(fp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
        # 后向聚合得分:
        # - 计算后向邻居的全局度数 bp_global_degrees。
        # - 对每个后向邻居的出度进行开方倒数的计算，并按节点的入度进行分割和求和，得到后向聚合得分 bp_agg_score。
        bp_global_degrees = nodes_feats['in_degrees'][belong_mask]
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -0.5).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(score[i] * torch.pow(bp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
    elif model_type is DistGNNType.DistSAGE:
        # 前向聚合得分:
        # - 对每个前向邻居的入度进行倒数的计算，并按节点的出度进行分割和求和，得到前向聚合得分 fp_agg_score。
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -1).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(value) for value in score])
        # 后向聚合得分:
        # - 对每个后向邻居的出度进行倒数的计算，并按节点的入度进行分割和求和，得到后向聚合得分 bp_agg_score。
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -1).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(value) for value in score])
    else:
        raise NotImplementedError(f'{model_type} is not implemented yet.')
    return (fp_agg_score, bp_agg_score)
        
        

        
    
    

