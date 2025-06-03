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
    Gets the send/recv idx and agg scores for each node in the local graph.
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
    Build and store send and receive indexes and aggregate scores for each partition and save this information to disk.
    HALO node.
    temp_send_idx: dictionary, representing the sending index of each partition. 
    recv_idx: dictionary, representing the received index of each partition. scores: dictionary, representing the aggregate scores for each partition.
    '''
    rank, world_size = comm.get_rank(), comm.get_world_size()
    temp_send_idx: Dict[int, Tensor] = {}
    recv_idx: Dict[int, Tensor] = {}
    scores: Dict[int, Tensor] = {}
    # The number and length of internal nodes in the current partition rank can represent the starting point of the external node.
    outer_start = len(torch.nonzero(nodes_feats['inner_node']))
    # build recv_idx and temp_idx
    temp_buffer: Dict[int, Tuple(Tensor, Tensor)] = {}
    for i in range(world_size):
        if i != rank:
            # Get the node ID belonging to partition i in the current partition rank (that is, the HALO node of partition i => partition rank)
            belong2i = (nodes_feats['part_id'] == i)
            # get forward & backward aggreagtion scores for remote neighbors
            agg_score = _get_agg_scores(local_graph, belong2i, nodes_feats, model_type)
            # Calculate the local ID or offset of these HALO nodes belonging to partition i in the current partition rank.
            # torch.nonzero(belong2i) is an index that takes a non-zero value. Since the ID is continuous, it can be regarded as the local ID of the node. 
            # Because the order is that inner node is ranked first and outer node is ranked behind, so you can choose this.
            local_ids = torch.nonzero(belong2i).view(-1) - outer_start
            # Calculate the ID of the starting node of the remote partition i (global ID) (not including HALO nodes)
            start = gpb.partid2nids(i)[0].item()
            # Calculate the ID of the nodes on the remote partition i (that is, the local ID or offset of these nodes on partition i); 
            # dgl.NID is a special key that indicates that the global ID of all nodes is to be taken
            remote_ids = nodes_feats[dgl.NID][belong2i] - start
            # The score here can reflect the importance of each HALO node
            temp_buffer[i] = (remote_ids, agg_score)
            # Indicates that data needs to be received from these HALO nodes later
            recv_idx[i] = local_ids  # data recv from i
            # local_ids represents the ID of the HALO node in the current partition (starting from the outer point position), and remote_ids represents the ID of the HALO node on the local partition. 
            # The HALO node here comes from partition i. 
            # Note that overall, IDs start from 0. This also makes it convenient if you look at the other party's partition i from the perspective of the other party's partition i, these IDs are already local IDs and do not need to be processed again.
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
            # Check whether partition i is associated with nodes due to the current rank partition. 
            # From the perspective of partition i, these nodes are HALO nodes from partition rank, and partition i requires recv_idx. 
            # Therefore, for partition rank, it is relatively send_idx. 
            # Since remote_ids has directly represented its ID on the partition to which it belongs, it can be assigned directly here without adjusting the local ID range.
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
    Returns the forward and backward aggregation scores for each node in the local map that are used to evaluate the importance or influence of the node.
    '''
    fp_neighbor_ids = local_graph.out_edges(local_graph.nodes()[belong_mask])[1]  # The forward neighbor of the node (i.e., the target node that exits the edge)
    fp_local_degrees = local_graph.out_degrees(local_graph.nodes()[belong_mask])  # Output of node
    bp_neighbor_ids = local_graph.in_edges(local_graph.nodes()[belong_mask])[0]   # The backward neighbor of the node (i.e., the source node entering the edge).
    bp_local_degrees = local_graph.in_degrees(local_graph.nodes()[belong_mask])   # The entry of the node.
    if model_type is DistGNNType.DistGCN:
        # Forward Aggregation Score:
        # - Calculate the global degrees of the forward neighbor fp_global_degrees. 
        # - Calculate the incoming degree of each forward neighbor and divide and sum according to the outgoing degree of the node to obtain the forward aggregation score fp_agg_score.
        fp_global_degrees = nodes_feats['out_degrees'][belong_mask]
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -0.5).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(score[i] * torch.pow(fp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
        # Backward aggregation score:
        # - Calculate the global degrees of backward neighbors bp_global_degrees. 
        # - Calculate the square countdown of the outgoing degree of each backward neighbor, and divide and sum according to the incoming degree of the node to obtain the backward aggregation score bp_agg_score.
        bp_global_degrees = nodes_feats['in_degrees'][belong_mask]
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -0.5).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(score[i] * torch.pow(bp_global_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
    elif model_type is DistGNNType.DistSAGE:
        # Forward Aggregation Score:
        # - The incoming degree of each forward neighbor is calculated in detail, and segmented and summed according to the outgoing degree of the node to obtain the forward aggregation score fp_agg_score.
        score = torch.pow(nodes_feats['in_degrees'][fp_neighbor_ids].float().clamp(min=1), -1).split(fp_local_degrees.tolist())
        fp_agg_score = torch.tensor([sum(value) for value in score])
        # Backward aggregation score:
        # - The outgoing degree of each backward neighbor is counted in detail, and the incoming degree of the node is divided and summed to obtain the backward aggregation score bp_agg_score.
        score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -1).split(bp_local_degrees.tolist())
        bp_agg_score = torch.tensor([sum(value) for value in score])
    else:
        raise NotImplementedError(f'{model_type} is not implemented yet.')
    return (fp_agg_score, bp_agg_score)
        
        

        
    
    

