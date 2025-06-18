# Graph partitioning of computing power perception. Digit the molecular graph based on the complexity of the graph and the computing power of the GPU.
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
        self.is_bidirected = is_bidirected    # bool，Is it an undirected graph (judgment condition: incoming degree of each node == outgoing degree)
        self.send_idx = send_idx              # dict，On each remote partition i, the start and end position of send_idx. For example {0: (0, 17660), 2: (17660, 36595)}
        self.total_send_idx = total_send_idx  # list，cat on all remote partitions i, send_idx's specific ID. It is the local ID of the corresponding partition. Such as [xx,xx,xx,...,xx]
        self.recv_idx = recv_idx              # dict，On the local partition, the HALO node belonging to the remote partition i is offset from the inner point ID. Such as {0: (0, 3, 6, 16), 2: (8, 9)}
        self.agg_scores = agg_scores          # dict，On the local partition, the score of the HALO node belonging to the remote partition i. For example {0: ([x,x] forward, [x,x] backward), 2: ([x,x] forward, [x,x] backward)}
        self.num_inner = num_inner            # int，Number of internal nodes on the current partition (not including HALO nodes)
        self.num_remote = num_remote          # int，The number of remote nodes on the current partition (i.e., HALO nodes)
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

def generate_random_node_features(g, feat_dim=128, num_classes=10, train_ratio=0.8, val_ratio=0.1, nodes_feats=None):
    """
    Generate random node features and labels for DGL graphs (isomorphic or heterogeneous), including feat, label, train/val/test mask.
    """
    nodes_feats = nodes_feats or {}
    ntypes = g.ntypes if isinstance(g, dgl.DGLHeteroGraph) else ['_N']
    for ntype in ntypes:
        inner_mask = g.nodes[ntype].data.get('inner_node', None)
        if inner_mask is None: continue
        inner_idx = inner_mask.nonzero(as_tuple=True)[0]
        N_inner = inner_idx.size(0)

        feat = torch.randn(N_inner, feat_dim)
        label = torch.randint(0, num_classes, (N_inner,), dtype=torch.long)
        idx_perm = torch.randperm(N_inner)
        n_train = int(train_ratio * N_inner)
        n_val = int(val_ratio * N_inner)
        train_mask = torch.zeros(N_inner, dtype=torch.bool)
        val_mask = torch.zeros(N_inner, dtype=torch.bool)
        test_mask = torch.zeros(N_inner, dtype=torch.bool)
        train_mask[idx_perm[:n_train]] = True
        val_mask[idx_perm[n_train:n_train + n_val]] = True
        test_mask[idx_perm[n_train + n_val:]] = True

        nodes_feats[f'{ntype}/feat'] = feat
        nodes_feats[f'{ntype}/label'] = label
        nodes_feats[f'{ntype}/train_mask'] = train_mask
        nodes_feats[f'{ntype}/val_mask'] = val_mask
        nodes_feats[f'{ntype}/test_mask'] = test_mask
    return nodes_feats


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
            # gpb: RangePartitionBook, is used to manage and process partition information of graph data in a distributed environment.
            #  The main function of RangePartitionBook is to provide partition information for nodes and edges. Specifically, it manages the following:
            #      - Partition mapping for nodes and edges: Assign a partition number to each node and edge to indicate which partition they belong to.
            #      - Partition scope: Define the range of nodes and edges in each partition, that is, the ID range of nodes and edges contained in each partition.
            #      - Cross-partition communication: In distributed training, compute nodes of different partitions need to exchange information of nodes and edges. RangePartitionBook helps manage these cross-partition communications.
            #   RangePartitionBook provides some methods and properties to query partition information, such as:
            #      - partid2nids(part_id): Returns the node ID range of the specified partition ID.
            #      - partid2eids(part_id)：Returns the edge ID range of the specified partition ID.
            #      - nid2partid(nid)：Returns the partition ID where the specified node ID is located.
            #      - eid2partid(eid)：Returns the partition ID where the specified edge ID is located.
            g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
            if len(nodes_feats.keys()) == 0:
                nodes_feats = {k: v for k, v in g.ndata.items()}
                # note: only for training purposes, not for accuracy
                generate_random_node_features(g=g, feat_dim=128, num_classes=64, train_ratio=0.8, val_ratio=0.1, nodes_feats=nodes_feats)
            # set graph degrees for GNNs aggregation
            # print(f'{rank}=>{g.formats()} [METIS]')
            save_dir = f'{partition_dir}/graph_degrees'
            # load global degrees information
            in_degrees_global, out_degrees_global = torch.load(f'{save_dir}/in_degrees.pt'), torch.load(f'{save_dir}/out_degrees.pt')
            # g.ndata['orig_id'] = torch.load(f'{save_dir}/orig_id_{rank}.pt')
            # The ID of the node in the original large image
            orig_id = g.ndata['orig_id']
            nodes_feats['in_degrees'] = in_degrees_global[orig_id]
            nodes_feats['out_degrees'] = out_degrees_global[orig_id]
            is_bidirected = torch.equal(nodes_feats['in_degrees'], nodes_feats['out_degrees'])
            # move all the features to nodes_feats
            node_type = node_type[0]
            # save original degrees for fp and bp
            nodes_feats[dgl.NID] = g.ndata[dgl.NID]
            # When there is only one partition, all nodes are counted
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
        Gets the send/recv idx and agg scores for each node in the local graph.
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
        Build and store send and receive indexes and aggregate scores for each partition and save this information to disk.
        HALO node.
        temp_send_idx: dictionary, representing the sending index of each partition. 
        recv_idx: dictionary, representing the received index of each partition. 
        scores: dictionary, representing the aggregate scores for each partition.
        '''
        # The halo node on the current partition has a local ID on the corresponding remote partition.
        temp_send_buffer_list = [{} for _ in range(self.partition_size)]
        # On the current partition, the ID of the node to be sent to the remote partition (these nodes are not halo nodes on the current partition, but halo nodes on the remote partition)
        temp_send_idx_list = [{} for _ in range(self.partition_size)]
        recv_idx_list = [{} for _ in range(self.partition_size)]
        scores_list = [{} for _ in range(self.partition_size)]

        for rank in range(self.partition_size):
            recv_idx: Dict[int, Tensor] = {}
            local_graph = graph_info[rank].graph
            nodes_feats = graph_info[rank].node_feats
            gpb = graph_info[rank].gpb

            # The number and length of internal nodes in the current partition rank can represent the starting point of the external node.
            # Note that nodes_feats['inner_node'] contains the HALO node's identity (False)
            inner_nodes_num = len(torch.nonzero(nodes_feats['inner_node']))
            temp_send_buffer = {}
            for i in range(self.partition_size):
                if i != rank:
                    # Get the node ID belonging to partition i in the current partition rank (that is, the HALO node of partition i => partition rank)
                    halo_local_belong2i_mask = (nodes_feats['part_id'] == i)
                    # get forward & backward aggreagtion scores for remote neighbors
                    agg_score = self._get_agg_scores(local_graph, halo_local_belong2i_mask, nodes_feats, self.model_type)
                    # Calculate the local ID or offset of these HALO nodes belonging to partition i in the current partition rank.
                    # Because the order is that inner node is ranked first and outer node is ranked behind, so it can be taken like this.
                    # Because the order is that inner node is ranked first and outer node is ranked behind, so it can be taken like this.
                    halo_local_offset_idx = torch.nonzero(halo_local_belong2i_mask).view(-1) - inner_nodes_num
                    # Calculate the ID of the starting node of the remote partition i (global ID) (not including HALO nodes)
                    halo_remote_start_id = gpb.partid2nids(i)[0].item()
                    # halo_remote_start_id2 = graph_info[i].node_feats[dgl.NID][0].item()
                    # Compute the ID of the nodes on the remote partition i (that is, the local ID or offset of these nodes on the partition i); 
                    # dgl.NID is a special key that indicates that the global ID of all nodes is to be taken.
                    # he global ID here can also represent the node's position index in the global ID list. For example, [20, 50] is an ID value and also a position index.
                    halo_remote_offset_ids = nodes_feats[dgl.NID][halo_local_belong2i_mask] - halo_remote_start_id
                    # The score here can reflect the importance of each HALO node
                    temp_send_buffer[i] = (halo_remote_offset_ids, agg_score)
                    # Indicates that these HALO nodes need to receive data later
                    recv_idx[i] = halo_local_offset_idx
                    # halo_local_offset_idx represents the ID of the HALO node in the current partition (starting from the outer point position), and halo_remote_offset_ids represents the ID of the HALO node on the partition to which it belongs. 
                    # The HALO node here comes from partition i. 
                    # Note that overall, IDs start from 0. This also makes it convenient if you look at the other party's partition i from the perspective of the other party's partition i, these IDs are already local IDs and do not need to be processed again.
            temp_send_buffer_list[rank] = temp_send_buffer
            recv_idx_list[rank] = recv_idx
            # print recv_idx in debug mode
            for k, v in recv_idx.items(): self.log(f'<worker{rank} recv {len(v)} nodes from worker{k}>')
        for rank in range(self.partition_size):
            temp_send_idx: Dict[int, Tensor] = {}
            scores: Dict[int, Tensor] = {}
            for part_i in range(self.partition_size):
                if part_i != rank:
                    # Check whether partition i has nodes associated with the current rank partition. (Or, is there any nodes in the current partition that are used as halo nodes in the remote partition) 
                    # From the perspective of partition i, these nodes are HALO nodes from partition rank, and partition i requires recv_idx. 
                    # Therefore, for partition rank, it is relatively send_idx. 
                    # Since remote_ids has directly represented its ID on the partition to which it belongs, it can be assigned directly here without adjusting the local ID range.
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
        Returns the forward and backward aggregation scores for each node in the local map that are used to evaluate the importance or influence of the node.
        '''
        halo_nodes = local_graph.nodes()[belong2i_mask]
        fp_local_halo_out_node_ids = local_graph.out_edges(halo_nodes)[1]   # In the current partition graph, the forward neighbor of the halo node (i.e. the target node that exits the edge)
        fp_local_out_degrees = local_graph.out_degrees(halo_nodes)          # In the current partition diagram, the output degree of the halo node
        bp_neighbor_ids = local_graph.in_edges(halo_nodes)[0]               # In the current partition graph, the backward neighbor of the halo node (i.e. the source node entering the edge)
        bp_local_degrees = local_graph.in_degrees(halo_nodes)               # In the current partition graph, the entry of the halo node

        # Delete points: 
        # On the one hand, limited deletion has the least impact on the current partition, which also has a small impact on accuracy. 
        # On the other hand, priority is given to retaining more frequent occurrences, which can avoid additional possible communications.
        if model_type is DistGNNType.DistGCN:
            # Forward aggregation score:
            # - Calculate the global degrees of the forward neighbor fp_global_out_degrees. 
            # - Calculate the incoming degree of each forward neighbor and divide and sum according to the outgoing degree of the node to obtain the forward aggregation score fp_agg_score. 
            # - In the local graph, how many nodes in the outgoing degree of each halo node occupy the invoice number of the node in the global graph?

            # The number of incoming points to the inner point to by each halo node in the whole graph. The larger the value, the more data this inner point needs to receive when propagating forward.
            fp_global_in_degrees = nodes_feats['in_degrees'][fp_local_halo_out_node_ids].float().clamp(min=1) 
            # The output degree of each halo node in the whole graph. The more output, the more important the node may be in the whole graph. But in the current partition, since it is a halo node, it is not necessarily true.
            fp_global_out_degrees = nodes_feats['out_degrees'][belong2i_mask]                        
            # In fact, it is a normalization of fp_global_in_degrees (the reciprocal of square root). The larger the value, the smaller the score, which means that the halo node will have a smaller impact on an inner point, because there are many other points that can be received by the inner point.
            score = torch.pow(fp_global_in_degrees, -0.5).split(fp_local_out_degrees.tolist())     
            # A weighted sum of neighbors of each halo node. Take the inverse of the square root of the output degree, the more the output degree, the smaller the value. The more output, the more the halo node is deleted in the current partition, it may be used in other partitions.
            fp_agg_score = torch.tensor([torch.sum(score[i] * torch.pow(fp_global_out_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
            # Backward aggregation score:
            # - Calculate the global degrees of backward neighbors bp_global_degrees.
            # - Calculate the square countdown of the outgoing degree of each backward neighbor, and divide and sum according to the incoming degree of the node to obtain the backward aggregation score bp_agg_score.
            bp_global_out_degrees = nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1)
            bp_global_in_degrees = nodes_feats['in_degrees'][belong2i_mask]
            score = torch.pow(bp_global_out_degrees, -0.5).split(bp_local_degrees.tolist())
            bp_agg_score = torch.tensor([torch.sum(score[i] * torch.pow(bp_global_in_degrees[i].float().clamp(min=1), -0.5)) for i in range(len(score))])
        elif model_type is DistGNNType.DistSAGE:
            # Forward Aggregation Score:
            # - The incoming degree of each forward neighbor is calculated inverted, and segmented and summed according to the outgoing degree of the node to obtain the forward aggregation score fp_agg_score.
            score = torch.pow(nodes_feats['in_degrees'][fp_local_halo_out_node_ids].float().clamp(min=1), -1).split(fp_local_out_degrees.tolist())
            fp_agg_score = torch.tensor([torch.sum(value) for value in score])
            # Backward aggregation score:
            # - The outgoing degree of each backward neighbor is calculated inversely, and segmented and summed according to the incoming degree of the node to obtain the backward aggregation score bp_agg_score.
            score = torch.pow(nodes_feats['out_degrees'][bp_neighbor_ids].float().clamp(min=1), -1).split(bp_local_degrees.tolist())
            bp_agg_score = torch.tensor([torch.sum(value) for value in score])
        else:
            raise NotImplementedError(f'{model_type} is not implemented yet.')
        return (fp_agg_score, bp_agg_score)

    def convert_send_idx(self, original_send_idx: Basic_Buffer_Type) -> Tuple[Dict[int, Tuple[int, int]], Tensor]:
        '''
        Calculate the start and end position of send_idx (converted_send_idx[i]) and the specific ID (total_idx[:xxx]) on each partition i. 
        Note: This function should be called after the graph is reordered.
        '''
        offset = 0
        converted_send_idx: Dict[int, Tuple[int, int]] = {}
        total_idx = []
        # Calculate the start and end ID range of nodes sent to each other partition
        for k, v in original_send_idx.items():
            converted_send_idx[k] = (offset, offset + len(v))
            offset += len(v)
            total_idx.append(v)
        total_idx = torch.cat(total_idx) if len(total_idx) else total_idx
        # converted_send_idx only records the start and end range, total_idx records the actual ID
        return converted_send_idx, total_idx

    def _reorder(self,
                 input_graph: DGLHeteroGraph,
                 nodes_feats: Dict[str, Tensor],
                 send_idx: Basic_Buffer_Type,
                 num_inner: int,
                 num_central: int,
                 m_mask: torch.Tensor,
                 c_mask: torch.Tensor):
        # Get index
        c_indices = c_mask.nonzero(as_tuple=True)[0][:num_central]
        m_indices = m_mask.nonzero(as_tuple=True)[0][:num_inner - num_central]

        # Generate new numbered map
        new_id = torch.full((num_inner,), -1, dtype=torch.long)  # Initialize to -1 to prevent unassigned values
        new_id[c_indices] = torch.arange(num_central, dtype=torch.long)
        new_id[m_indices] = torch.arange(num_central, num_inner, dtype=torch.long)

        # Make sure all nodes within num_inner are mapped correctly
        assert (new_id >= 0).all(), "The new_id assignment is incomplete, and there may be an undefined index!"

        # Renumber edge index
        u, v = input_graph.edges()
        mask_u = u < num_inner
        mask_v = v < num_inner
        u[mask_u] = new_id[u[mask_u].long()]
        v[mask_v] = new_id[v[mask_v].long()]

        # Construct a new DGL diagram
        reordered_graph = dgl.heterograph(
            {etype: (u, v) for etype in input_graph.etypes}) if input_graph.is_hetero else dgl.graph((u, v))

        # Rearrange node features
        for key in nodes_feats:
            new_feats = torch.zeros_like(nodes_feats[key][:num_inner])
            new_feats[new_id] = nodes_feats[key][:num_inner]
            nodes_feats[key] = new_feats

        # Remap send_idx
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
        - In the original graph, the ID of the center node (such as [0, 2, 5, 6, 7]) and the ID of the edge node (such as [1, 3, 4, 8, 9]) may be discontinuous, and they are rearranged here for subsequent processing.
        - Specifically, the ID range of the central node is ranked first (such as 0~10), and the ID range of the edge node is ranked behind (such as 11~20), and it is not processed for the HALO node (such as 21~50).
        - Note that the ID range has changed, but their positions have not changed, and they are still interlaced (such as [0,1,11,12,2,13,3,14,15,16,4,...]).
        '''
        # Generate a sequence for recording node ID
        new_id = torch.zeros(size=(num_inner,), dtype=torch.long)
        # For the first num_inner elements in c_mask, set the corresponding position of new_id to the value from 0 to num_central-1
        # c_mask and m_mask both contain internal nodes and HALO nodes, so the first num_inners need to be taken, that is, they are all internal nodes
        new_id[c_mask[:num_inner]] = torch.arange(num_central, dtype=torch.long)
        # For the first num_inner elements in m_mask, set the corresponding position of new_id to the value from num_central to num_inner-1
        new_id[m_mask[:num_inner]] = torch.arange(num_central, num_inner, dtype=torch.long)
        # Get edges (u and v) of the input graph
        u, v = input_graph.edges()
        # For u and v values ​​smaller than num_inner, use new_id for replacement.
        # Because u and v in graph contain HALO nodes, it is smaller than num_inner to only take internal nodes.
        # Remap the internal node ID to the range of new_id, and will not be processed for HALO nodes.
        u[u < num_inner] = new_id[u[u < num_inner].long()]
        v[v < num_inner] = new_id[v[v < num_inner].long()]
        # Create a new graph using the updated u and v
        reordered_graph = dgl.graph((u, v))
        # Iterate through each key in nodes_feats and rearrange its features according to new_id.
        # Since the position of the node in new_id has not changed, its value has changed, so the left position corresponds to the right position, and you can directly assign the value.
        for key in nodes_feats: nodes_feats[key][new_id] = nodes_feats[key].clone()[0:num_inner]
        # Iterate through each key in send_idx and rearrange it by new_id
        for key in send_idx: send_idx[key] = new_id[send_idx[key]]
        return reordered_graph, nodes_feats, send_idx

    def reorder_graph(self, graph_info):
        # Reorder the nodes and features of the graph into internal nodes, edge nodes, and central nodes.
        #   marginal nodes: edge nodes, nodes with remote neighbors on other devices;
        #   central nodes: a central node, a node without a remote neighbor. That is, the part of the internal node that is not an edge node;
        for rank in range(self.partition_size):
            g_info = graph_info[rank]
            original_graph: DGLHeteroGraph = g_info.graph
            nodes_feats: Dict[str, Tensor] = g_info.node_feats
            send_idx: Basic_Buffer_Type = g_info.send_idx

            # Get the internal node mask (including the HALO node, but its value is False)
            inner_mask = nodes_feats['inner_node']
            # The number of calculated values ​​is True, that is, the number of real internal nodes
            num_inner = torch.count_nonzero(inner_mask).item()
            # Get the remote node. Except for the internal nodes, the rest are remote nodes (that is, take out the HALO nodes)
            halo_remote_nodes = original_graph.nodes()[~inner_mask]
            # By obtaining the out edge of the remote node, the internal edge node marginal_nodes is extracted
            _, v = original_graph.out_edges(halo_remote_nodes)
            # There may be duplication, that is, it points to the same target node and needs to be replicated.
            marginal_nodes = torch.unique(v)
            # Mark edge nodes and center nodes (i.e., parts of the internal nodes that are not edge nodes), including HALO nodes
            marginal_mask = torch.zeros_like(inner_mask, dtype=torch.bool)
            marginal_mask[marginal_nodes] = True
            # In the internal node, those that are not marginal_nodes are considered central_nodes;
            # In order to have uniform lengths, HALO nodes are spliced, but they are all False
            central_mask = torch.concat([~marginal_mask[:num_inner], marginal_mask[num_inner:]])
            num_marginal = torch.count_nonzero(marginal_mask).item()
            num_central = torch.count_nonzero(central_mask).item()

            # Let the ID range of the central node be ranked first (such as 0~10), and the ID range of the edge node be ranked behind (such as 11~20), and the HALO node is not processed (such as 21~50).
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
            # Calculate the start and end position and specific ID of send_idx on each remote partition i.
            # For example: For now partition 1, there is send_idx={0: (0, 17660), 2: (17660, 36595)} total_idx=[xx,xx,xx,...,xx]
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
        Bind the GPU to the partition. Note that you need to bind to the graph in order from weak to strong computing power, that is, self.graph_info[0] is bound to the weakest GPU. 
        The partition ID of the subgraph corresponds to the index of sorted_gpu_ids and has nothing to do with the GPU ID.
        '''
        assert len(gpus.split(',')) == len(graph_info), ">> The number of GPUs is different from the number of partitions <<"
        gpu_list = list(map(int, gpus.split(',')))
        # Sort by indices value descending order and get the corresponding GPU device key. The larger the value, the greater the calculation cost and the weaker the GPU.
        self.sorted_gpu_ids = get_gpu_capability(gpu_list)
        # The first time is to specify the GPU and graph casually
        for i, g in enumerate(graph_info):
            print(f"<assign {torch.cuda.get_device_name(self.sorted_gpu_ids[i])}[{self.sorted_gpu_ids[i]}] to subgraph {i}>")
            g.device = torch.device(f"cuda:{self.sorted_gpu_ids[i]}")
            # g.num_remote = torch.count_nonzero(g.graph.nodes_feats['outer_node']).item()
            self.assig_graphs_gpus[self.sorted_gpu_ids[i]] = g
        return self.assig_graphs_gpus

    def check_graph_connectivity(self, graph):
        # Convert DGL diagram to NetworkX diagram
        nx_graph = graph.to_networkx().to_undirected()
        # Check connectivity with NetworkX
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
        # Start with the weakest GPU
        for index in range(len(self.sorted_gpu_ids)):
            gpu_id = self.sorted_gpu_ids[index]
            # Calculate λ and its average value λ_mean
            lam, lam_mean, lam_std, halo_edges = self.calculate_target(assig_graphs_gpus)
            initial_halo_edges = halo_edges[index]
            print(f"gpu_id: {gpu_id}[{index}], lam - mean: {lam[index] - lam_mean}[{lam[index]} - {lam_mean}]")

            # Find all halo nodes on this partition
            g_info = assig_graphs_gpus[gpu_id]
            local_graph = g_info.graph
            nodes_feats = g_info.node_feats
            gpb = g_info.gpb
            part_id = gpb.partid  # The partition ID of the graph belongs to

            # Select a GPU with λ_i greater than λ_mean. Skip first if less than or equal
            nodes = local_graph.num_nodes()
            edges = local_graph.num_edges()
            if lam[index] <= lam_mean and gpu_memory_enough(gpu_id=gpu_id, num_nodes=nodes, num_edges=edges, feats=nodes_feats["feat"], beta=1e8):
                result[index] = True
                print(f"Partition {part_id} has no change")
                print(f"No change in the number of nodes: {g_info.graph.num_nodes()}")
                print(f"No change in edge count: {g_info.graph.num_edges()}")
                print(f"No change in scores: {lam[index]}")
                records.append([part_id, g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index], '=>', g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index]])
                continue

            halo_local_offset_start = len(torch.nonzero(nodes_feats['inner_node']))
            halo_local_masks = (nodes_feats['part_id'] != part_id)
            halo_nodes = local_graph.nodes()[halo_local_masks]
            # Get the index of the halo node on the current partition
            halo_local_idxs = torch.nonzero(halo_local_masks).view(-1)
            halo_global_ids = nodes_feats[dgl.NID][halo_local_idxs]


            # Calculate the aggregate scores of these halo nodes
            # agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            # sorted_scores, ori_idxs = torch.sort(agg_scores[0] + agg_scores[1], descending=False)

            agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            indices = torch.nonzero(torch.isin(torch.tensor(self.global_id_counter[0]), halo_global_ids)).view(-1)
            scores = (agg_scores[0] + agg_scores[1]) * torch.tensor(self.global_id_counter[1])[indices]
            sorted_scores, ori_idxs = torch.sort(scores, descending=False)

            # Choose the one with the highest score
            nodes_to_remove_id = []
            nodes_to_remove_idx = []
            # The number of remaining nodes and edges in the initial
            remaining_inner_nodes = torch.count_nonzero(nodes_feats['inner_node']).item()
            remaining_all_nodes = local_graph.num_nodes()
            remaining_edges = local_graph.num_edges()
            initial_halo_nodes_count = torch.count_nonzero(halo_local_masks)
            # Track deleted edges to avoid duplicate counting
            removed_edges_set = set()
            # Estimate the impact of removing nodes and edges
            estimated_lam_reduction = 0

            for i, (score, idx) in enumerate(zip(sorted_scores, ori_idxs)):
                max_score_idx = idx.item()
                halo_local_node_id = halo_nodes[max_score_idx].item()
                # Get the edge of the node and consider duplication
                max_score_node_edges = local_graph.out_edges(halo_local_node_id, form='eid').tolist()
                for edge_id in max_score_node_edges:
                    if edge_id not in removed_edges_set:
                        removed_edges_set.add(edge_id)
                        remaining_edges -= 1
                # Check whether the node is an inner point, and only reduce remainsing_inner_nodes when it is an inner point
                # if nodes_feats['inner_node'][halo_local_node_id].item(): remaining_inner_nodes -= 1
                remaining_all_nodes -= 1
                remaining_halo_nodes = initial_halo_nodes_count - len(nodes_to_remove_id)
                remaining_halo_edges = initial_halo_edges - len(removed_edges_set)  # len(removed_edges_set.intersection(max_score_node_edges))

                # Estimate the removed T_comp and T_comm
                t1 = self.calculate_T_comp(gpu_id, num_nodes=remaining_inner_nodes, num_edges=remaining_edges)
                t2 = self.calculate_T_comm(gpu_id, E_outer=remaining_halo_edges)
                estimated_lam_reduction = t1 + t2
                # If the estimated lambda change is close to lam_mean - lam[index], then select
                if estimated_lam_reduction <= (lam[index] + lam_mean)/2 and gpu_memory_enough(gpu_id=gpu_id, num_nodes=remaining_all_nodes, num_edges=remaining_edges, feats=nodes_feats["feat"], beta=1e8): break
                nodes_to_remove_id.append(halo_local_node_id)
                nodes_to_remove_idx.append(halo_local_idxs[max_score_idx])
            # Delete selected nodes
            if len(nodes_to_remove_id) > 0:
                print(f"{len(nodes_to_remove_id)} nodes were deleted for partition {part_id}")
                new_graph = dgl.remove_nodes(local_graph, torch.tensor(nodes_to_remove_id))
                new_node_feats = {key: value[~torch.isin(torch.arange(value.size(0)), torch.tensor(nodes_to_remove_idx))] for key, value in nodes_feats.items()}
                print(f"Changes in the number of nodes: {g_info.graph.num_nodes()} => {new_graph.num_nodes()}")
                print(f"Changes in edge count: {g_info.graph.num_edges()} => {new_graph.num_edges()}")
                print(f"Changes in scores: {lam[index]} => {estimated_lam_reduction}")
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
                print(f"Partition {part_id} has no change")
                print(f"No change in the number of nodes: {g_info.graph.num_nodes()}")
                print(f"No change in edge count: {g_info.graph.num_edges()}")
                print(f"No change in scores: {lam[index]}")
                records.append([part_id, g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index], '=>', g_info.graph.num_nodes(), g_info.graph.num_edges(), lam[index]])

            # Calculate and compare the target value. If it is still greater than, continue to delete it.
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

        # Effect information output
        lam_prev, lam_mean_prev, lam_std_prev, _ = self.calculate_target(self.assig_graphs_gpus)
        lam, lam_mean, lam_std, _ = self.calculate_target(assig_graphs_gpus)
        for index in range(len(self.sorted_gpu_ids)):
            gpu_id = self.sorted_gpu_ids[index]
            g_info_prev = self.assig_graphs_gpus[gpu_id]
            g_info = assig_graphs_gpus[gpu_id]
            print(f"GPU: {gpu_id}")
            print(f"Partition: {g_info.gpb.partid}")
            print(f"Number of nodes: {g_info_prev.graph.num_nodes()}[in:{g_info_prev.num_inner}] => {g_info.graph.num_nodes()}[in:{g_info.num_inner}]")
            print(f"Number of edges: {g_info_prev.graph.num_edges()} => {g_info.graph.num_edges()}")
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
                print(f"Partition {part_id} does not require node deletion")
                continue

            halo_local_masks = (nodes_feats['part_id'] != part_id)
            # Get the index of the halo node on the current partition
            halo_local_idxs = torch.nonzero(halo_local_masks).view(-1)
            # Calculate the aggregate scores of these halo nodes
            agg_scores = self._get_agg_scores(local_graph, halo_local_masks, nodes_feats, self.model_type)
            # For simplicity, use forward_score first, and then consider forward+backward. For undirected graphs, both should be the same.
            forward_scores = agg_scores[0]
            # backward_scores = agg_scores[1]
            scores = forward_scores  # + backward_scores
            sorted_scores, ori_idxs = torch.sort(scores, descending=False)

            # Select the highest/lowest score
            nodes_to_remove = []
            remaining_all_nodes = local_graph.num_nodes()
            remaining_edges = local_graph.num_edges()
            # Track deleted edges to avoid duplicate counting
            removed_edges_set = set()
            initial_halo_nodes_count = len(halo_local_idxs)
            # Estimate the impact of removing nodes and edges
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

            # Delete selected nodes
            if nodes_to_remove:
                print(f"{len(nodes_to_remove)} nodes were deleted for partition {part_id}")
                nodes_to_remove_tensor = torch.tensor(nodes_to_remove)
                new_graph = dgl.remove_nodes(local_graph, nodes_to_remove_tensor)
                new_node_feats = {key: value[~torch.isin(torch.arange(value.size(0)), nodes_to_remove_tensor)] for
                                  key, value in nodes_feats.items()}
                print(f"Changes in the number of nodes:{g_info.graph.num_nodes()} => {new_graph.num_nodes()}")
                print(f"Changes in edge count:{g_info.graph.num_edges()} => {new_graph.num_edges()}")
                g_info.graph = new_graph
                g_info.node_feats = new_node_feats
                g_info.num_inner = torch.sum(new_node_feats['inner_node']).item()
                assig_graphs_gpus[gpu_id] = g_info
            # Calculate and compare the target value. If it is still greater than, continue to delete it.
            print('*' * 50)
        return assig_graphs_gpus

    def overlapping_our(self, part_size, dataset, gpus_list, part_dir='data/part_data', k=-1, our_partition=False):
        max_subg_size = 0
        all_part_nodes_feats = []
        halo_node_features = {}
        model_type = DistGNNType.DistGCN
        # Loading pkl file /home/sxf/Desktop/gnn/dist_gnn_fullbatch
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
                print(f"[{part_id}-{i}] This time, {len(temp.keys())}/{len(halo_global_ids)} halo nodes are added to come in.")
        return halo_node_features, max_subg_size


def auto_select_dtype(feat_tensor):
    """
    Automatically select the appropriate data type according to the actual content of the node feature tensor. 
    parameter:
        feat_tensor (torch.Tensor): Node feature tensor (float32 type) Return:
        torch.dtype: Recommended data types
    """
    # Check whether all elements can be regarded as integers 
    # Calculate the fractional part and determine whether it is close to 0
    if torch.all((feat_tensor - torch.round(feat_tensor)) == 0):
        # If all numbers are close to integers, then convert to integer type
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
        # If it is a floating point number, judge based on the range and standard deviation of the floating point number
        min_val = torch.min(feat_tensor)
        max_val = torch.max(feat_tensor)
        # The value range of float16 is approximately 6.1e-5 to 65504
        if min_val >= torch.iinfo(torch.float16).min and max_val <= torch.iinfo(torch.float16).max:
            std = torch.std(feat_tensor)
            # If the range of features is small and the accuracy requirements are not high, it can use float16
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
    # ------------Main parameter modification area---------------- #
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
    print(f'>> Dataset: {args.dataset_name}')
    print(f'>> GPU list: [{gpus_index}] => {gpus}')
    print(f'>> Partition threshold: {threshold}')
    print(f'>> Number of partitions: {part_num}')
    print(f'>> Partition method: {part_method}')
    print(f'>> Partition policy: {our_partition}')

    args = vars(args)
    app = CSPart(args)
    # 1. Pre-partitioning using off-the-shelf partitioning algorithm;
    app.coarse_graph_patition(fast_skip=True, part_method=part_method, num_hops=1)
    # 2. Construct a local weighted graph, that is, calculate the weighted value only for the edge nodes;
    # Loading partition subgraphs and features
    app.coarse_load_patition()
    # dtype = auto_select_dtype(app.graph_info[0].node_feats['feat'])
    # print('>> Original data type:', app.graph_info[0].node_feats['feat'].dtype)
    # print('>> Suitable data types:', dtype)
    # 2.1 Get the sent and received vertex IDs for each partition, and the degree score of the edge node
    app.get_send_recv_idx_scores(app.graph_info, skip_load=False, suffix='_init')
    # 2.2 Rearrange the order of node IDs in the graph (from 0 to N-1: Center Node->Edge Node->Remote Node
    app.reorder_graph(app.graph_info)
    # app.print_graph(0)
    # 3. Pre-allocate a partition for each GPU
    assig_graphs_gpus = app.assignment_graphs(gpus, app.graph_info)
    app.get_halo_count()
    new_assig_graphs_gpus = copy.deepcopy(assig_graphs_gpus)
    if our_partition and len(gpus.split(',')) > 1: new_assig_graphs_gpus = app.do_partition(new_assig_graphs_gpus, threshold=threshold)
    else: print(">> No partition required")
    print('=' * 100)

    new_graph_info = []
    for index in range(len(app.sorted_gpu_ids)): new_graph_info.append(new_assig_graphs_gpus[app.sorted_gpu_ids[index]])
    app.get_send_recv_idx_scores(new_graph_info, skip_load=True)
    app.reorder_graph(new_graph_info)
    for index in range(len(app.sorted_gpu_ids)): assig_graphs_gpus[app.sorted_gpu_ids[index]] = new_graph_info[index]

    # Save partition results
    partition_file = os.path.join(app.partition_dir, f'{app.dataset_name}/{app.partition_size}part/{app.dataset_name}_processed_partitions_{our_partition}_{sorted(gpus_list)}.pkl')
    with open(partition_file, 'wb') as f: pickle.dump(assig_graphs_gpus, f)
    print(f"Processed partitions saved to {partition_file}")

    # Loading pkl file
    if test_load_part:
        print("Test loading pkl file")
        with open(partition_file, 'rb') as f: _ = pickle.load(f)
    print("=" * 100)
    print(f"When training, be sure to set CUDA_VISIBLE_DEVICES in the following GPU order：{','.join(map(str, app.sorted_gpu_ids))}")

    halo_node_features, _ = app.overlapping_our(part_size=args['partition_size'], dataset=args['dataset_name'], gpus_list=gpus_list, k=-1, part_dir='/mnt/disk/sxf/data/part_data', our_partition=our_partition)
    print(f"The final number of valid nodes: [{args['dataset_name']}][{our_partition}][{part_num}][{gpus_index}] => {len(halo_node_features.keys())}")
    del app
    torch.cuda.empty_cache()
    print('#' * 50)



# Calculate the halo node duplication
def overlapping(dataset_name, partition_size):
    partition_dir = './data/part_data'
    part_config = f'{partition_dir}/{dataset_name}/{partition_size}part/{dataset_name}.json'

    # Stores the global ID of halo nodes and the total number of nodes per partition
    halo_global_ids_by_partition = defaultdict(set)
    total_nodes_by_partition = {}
    # in_degrees_by_partition = defaultdict(list)
    # out_degrees_by_partition = defaultdict(list)

    # Looping the halo nodes of each partition
    for rank in range(partition_size):
        g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
        # Get the halo node (the node with inner_node False)
        halo_nodes = g.nodes()[~g.ndata['inner_node'].bool()].numpy()
        # Use 'dgl.NID' to get the global ID of the halo node
        halo_global_ids = g.ndata[dgl.NID][halo_nodes].numpy()
        # The global ID of the halo node that stores the partition
        halo_global_ids_by_partition[rank] = set(halo_global_ids)
        # Total number of nodes that store the partition
        total_nodes_by_partition[rank] = g.num_nodes()
        # Get the original ID of the halo node in this partition
        orig_ids = g.ndata['orig_id'][halo_nodes].numpy()
        # Store incoming and outgoing degrees
        # in_degrees_by_partition[rank] = in_degrees_global[orig_ids].numpy()
        # out_degrees_by_partition[rank] = out_degrees_global[orig_ids].numpy()
        
    # Output the number of halo nodes and total number of nodes per partition
    for rank in range(partition_size):
        halo_count = len(halo_global_ids_by_partition[rank])
        total_count = total_nodes_by_partition[rank]
        # avg_in_degree = in_degrees_by_partition[rank].mean() if halo_count > 0 else 0
        # avg_out_degree = out_degrees_by_partition[rank].mean() if halo_count > 0 else 0
        # print(f"Part {rank} halo nodes: {halo_count}/{total_count}, Avg In-Degree: {avg_in_degree:.2f}, Avg Out-Degree: {avg_out_degree:.2f}")
        print(f"Part {rank} halo nodes: {halo_count}/{total_count}")

    # Store total overlapping halo nodes
    total_overlapping_halo_nodes_set = set()
    total_overlapping_halo_nodes = 0
    # Compare the duplication of halo nodes between each partition and output the result
    for i in range(partition_size):
        for j in range(i + 1, partition_size):
            # Get the halo nodes of two partitions
            halo_i = halo_global_ids_by_partition[i]
            halo_j = halo_global_ids_by_partition[j]
            # Calculate the number of overlapping halo nodes
            overlapping_halo_nodes = halo_i.intersection(halo_j)
            total_overlapping_halo_nodes += len(overlapping_halo_nodes)
            # Update the total overlapping halo node collection
            total_overlapping_halo_nodes_set.update(overlapping_halo_nodes)
            # Output result
            print(f"Part {i} - Part {j} -> Overlapping halo nodes: {len(overlapping_halo_nodes)}")

    print('*' * 50)
    print(f"Total overlapping halo nodes across all partitions: {total_overlapping_halo_nodes}")
    # Output the total number of overlapping halo nodes
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

