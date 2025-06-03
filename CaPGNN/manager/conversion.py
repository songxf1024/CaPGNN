import dgl
import torch
from dgl import DGLHeteroGraph
from dgl.distributed import GraphPartitionBook
from torch import Tensor, BoolTensor
from typing import Dict, Tuple

from ..communicator import Communicator as comm
from ..communicator import Basic_Buffer_Type

'''
*************************************************
*********** graph conversion functions **********
*************************************************
'''
    
def convert_partition(part_dir: str, dataset: str) -> Tuple[dgl.DGLHeteroGraph, Dict[str, Tensor], GraphPartitionBook, bool]:
    rank, world_size = comm.get_rank(), comm.get_world_size()
    part_config = f'{part_dir}/{dataset}/{world_size}part/{dataset}.json'
    # gpb: RangePartitionBook, is used to manage and process partition information of graph data in a distributed environment. 
    # RangePartitionBook's main function is to provide partition information for nodes and edges. Specifically, it manages the following: 
    # - Partition mapping for nodes and edges: Assign a partition number to each node and edge to indicate which partition they belong to. 
    # - Partition scope: Define the range of nodes and edges in each partition, that is, the ID range of nodes and edges contained in each partition. 
    # - Cross-partition communication: In distributed training, compute nodes of different partitions need to exchange information of nodes and edges. 
    # RangePartitionBook helps manage these cross-partition communications. RangePartitionBook provides some methods and properties to query partition information, such as: 
    # - partid2nids(part_id): Returns the node ID range of the specified partition ID. 
    # - partid2eids(part_id): Returns the edge ID range of the specified partition ID. 
    # - nid2partid(nid): Returns the partition ID where the specified node ID is located. 
    # - eid2partid(eid): Returns the partition ID where the specified edge ID is located.
    g, nodes_feats, efeats, gpb, graph_name, node_type, etype = dgl.distributed.load_partition(part_config, rank)
    # set graph degrees for GNNs aggregation
    save_dir = f'graph_degrees/{dataset}'
    # load global degrees information 
    in_degrees_global, out_degrees_global = torch.load(f'{save_dir}/in_degrees.pt'), torch.load(f'{save_dir}/out_degrees.pt')
    degree_ids = g.ndata['orig_id']
    nodes_feats['in_degrees'] = in_degrees_global[degree_ids]
    nodes_feats['out_degrees'] = out_degrees_global[degree_ids]
    if all(nodes_feats['in_degrees'] == nodes_feats['out_degrees']):
        is_bidirected = True
    else:
        is_bidirected = False
    # move all the features to nodes_feats
    node_type = node_type[0]
    # save original degrees for fp and bp
    nodes_feats[dgl.NID] = g.ndata[dgl.NID]
    # When there is only one partition, all nodes are counted
    nodes_feats['part_id'] = g.ndata.get('part_id', torch.zeros_like(degree_ids))
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
    return g, nodes_feats, gpb, is_bidirected

def is_bidirectional(graph):
    u, v = graph.edges()
    edge_set = set(zip(u.numpy(), v.numpy()))
    for src, dst in edge_set:
        if (dst, src) not in edge_set:
            return False
    return True

def reorder_graph(original_graph: DGLHeteroGraph, nodes_feats: Dict[str, Tensor], send_idx: Basic_Buffer_Type) -> Tuple[DGLHeteroGraph, Dict[str, Tensor], Basic_Buffer_Type, int, int]:
    # Reorder the nodes and features of the graph into internal nodes, edge nodes, and central nodes. 
    #     marginal nodes: edge nodes, nodes with remote neighbors on other devices; 
    #     central nodes: central nodes, nodes without remote neighbors;

    # get inner mask and remote nodes
    inner_mask = nodes_feats['inner_node']
    num_inner = torch.count_nonzero(inner_mask).item()
    remote_nodes = original_graph.nodes()[~inner_mask]
    # get marginal dst nodes.
    _, v = original_graph.out_edges(remote_nodes)
    marginal_nodes = torch.unique(v)
    # set marginal mask and central mask
    marginal_mask = torch.zeros_like(inner_mask, dtype=torch.bool)
    marginal_mask[marginal_nodes] = True
    central_mask = torch.concat([~marginal_mask[:num_inner], marginal_mask[num_inner:]])
    num_marginal, num_central = torch.count_nonzero(marginal_mask).item(), torch.count_nonzero(central_mask).item()
    new_graph, nodes_feats, send_idx = _reorder(original_graph, nodes_feats, send_idx, num_inner, num_central, marginal_mask, central_mask)
    assert new_graph.num_nodes() == original_graph.num_nodes()
    assert new_graph.num_edges() == original_graph.num_edges()
    return new_graph, nodes_feats, send_idx, num_marginal, num_central

def _reorder(input_graph: DGLHeteroGraph, nodes_feats: Dict[str, Tensor], send_idx: Basic_Buffer_Type, num_inner: int, num_central: int, m_mask, c_mask) -> Tuple[DGLHeteroGraph, Dict[str, Tensor], Basic_Buffer_Type]:
    '''reorder local nodes and return new graph and feats dict'''
    #
    # get the new ids
    new_id = torch.zeros(size=(num_inner,), dtype=torch.long)
    # For the first num_inner elements in c_mask, set the corresponding position of new_id to the value from 0 to num_central-1
    new_id[c_mask[:num_inner]] = torch.arange(num_central, dtype=torch.long)
    # For the first num_inner elements in m_mask, set the corresponding position of new_id to the value from num_central to num_inner-1
    new_id[m_mask[:num_inner]] = torch.arange(num_central, num_inner, dtype=torch.long)
    # Get edges (u and v) of the input graph
    u, v = input_graph.edges()
    # For u and v values ​​smaller than num_inner, use new_id to replace
    u[u < num_inner] = new_id[u[u < num_inner].long()]
    v[v < num_inner] = new_id[v[v < num_inner].long()]
    # Create a new graph using updated u and v
    reordered_graph = dgl.graph((u, v))
    # Iterate through each key in nodes_feats and rearrange its features according to new_id
    for key in nodes_feats: nodes_feats[key][new_id] = nodes_feats[key].clone()[0:num_inner]
    # Iterate through each key in send_idx and rearrange it by new_id
    for key in send_idx: send_idx[key] = new_id[send_idx[key]]
    return reordered_graph, nodes_feats, send_idx

def convert_send_idx(original_send_idx: Basic_Buffer_Type) -> Tuple[Dict[int, Tuple[int, int]], Tensor]:
    '''
    Convert the original layout of send_idx to offset fashion.
    
    `ATTENTION`: this function should be called after graph reordering.
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

'''
*************************************************
********* graph decomposition functions *********
*************************************************
'''

def decompose_graph(original_graph: DGLHeteroGraph, num_central: int, num_inner: int) -> Tuple[DGLHeteroGraph, DGLHeteroGraph, Tensor, Tensor]:
    '''construct central and marginal graphs from the original_graph'''
    # forward central and marginal graph
    central_graph, src_marginal_idx = _build_central_graph(original_graph, num_central)
    marginal_graph, src_central_idx = _build_marginal_graph(original_graph, num_central, num_inner)
    # set global degrees
    fetch_global_degrees(original_graph, central_graph, marginal_graph, num_central, num_inner, src_marginal_idx, src_central_idx)
    return central_graph, marginal_graph, src_marginal_idx, src_central_idx

def fetch_global_degrees(original_graph: DGLHeteroGraph, central_graph: DGLHeteroGraph, marginal_graph: DGLHeteroGraph, num_central: int, num_inner: int, src_m_idx: Tensor, src_c_idx: Tensor):
    central_in_degrees = torch.concat([original_graph.ndata['in_degrees'][:num_central], original_graph.ndata['in_degrees'][src_m_idx]])
    central_out_degrees = torch.concat([original_graph.ndata['out_degrees'][:num_central], original_graph.ndata['out_degrees'][src_m_idx]])
    central_graph.ndata['in_degrees'] = central_in_degrees
    central_graph.ndata['out_degrees'] = central_out_degrees
    marginal_in_degrees = torch.concat([original_graph.ndata['in_degrees'][src_c_idx], original_graph.ndata['in_degrees'][num_central: num_inner], original_graph.ndata['in_degrees'][num_inner:]])
    marginal_out_degrees = torch.concat([original_graph.ndata['out_degrees'][src_c_idx], original_graph.ndata['out_degrees'][num_central: num_inner], original_graph.ndata['out_degrees'][num_inner:]])
    marginal_graph.ndata['in_degrees'] = marginal_in_degrees
    marginal_graph.ndata['out_degrees'] = marginal_out_degrees

def _build_central_graph(original_graph: DGLHeteroGraph, num_central: int) -> Tuple[DGLHeteroGraph, Tensor]:
    # get src and dst that are both central nodes
    u, v = original_graph.edges()
    only_central = torch.logical_and(u < num_central, v < num_central)
    c_u, c_v = u[only_central], v[only_central]
    _u, _v = [c_u], [c_v]
    # get src that are marginal nodes (used for fetch feats from local feats)
    from_marginal = torch.logical_and(u >= num_central, v < num_central)
    m_u, m_v = u[from_marginal], v[from_marginal]
    # assign new ids for marginal src nodes
    uniq_m_u, m_u_inv = torch.unique(m_u, return_inverse=True)
    new_ids_m_u = (torch.arange(uniq_m_u.size(0)) + num_central)[m_u_inv]
    _u.append(new_ids_m_u)
    _v.append(m_v)
    _u, _v = torch.cat(_u), torch.cat(_v)
    central_graph = dgl.graph((_u, _v))
    return central_graph, uniq_m_u

def _build_marginal_graph(original_graph: dgl.DGLGraph, num_central: int, num_inner: int) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    u, v = original_graph.edges()
    central_mask = original_graph.nodes() < num_central
    # find no-need central nodes
    u_mask = u < num_central  # src central nodes
    v_mask = torch.logical_and(v >= num_central, v < num_inner)  # dst marginal nodes
    from_central = torch.logical_and(u_mask, v_mask)
    need_central = torch.unique(u[from_central])
    central_mask[need_central] = False
    no_need_central = original_graph.nodes()[central_mask]
    # remove no-need edges to central dst nodes and remote dst nodes
    edge_ids = []
    ctr_edge_ids = torch.nonzero(v < num_central).squeeze()
    rmt_edge_ids = torch.nonzero(v >= num_inner).squeeze()
    if len(ctr_edge_ids) != 0:
        edge_ids.append(ctr_edge_ids)
    if len(rmt_edge_ids) != 0:
        edge_ids.append(rmt_edge_ids)
    if len(edge_ids) != 0:
        marginal_graph = dgl.remove_edges(original_graph, torch.cat(edge_ids))  # remove no-need edges first
    marginal_graph = dgl.remove_nodes(marginal_graph, no_need_central)  # remove no-need central nodes
    return marginal_graph, need_central

