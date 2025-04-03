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
    # 对于只有一个分区的时候，所有节点都算
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
    # 将图的节点和特征重新排序为内部节点、边缘节点和中心节点。
    #   marginal nodes: 边缘节点, 在其他设备上具有远程邻居的节点;
    #   central nodes: 中心节点, 没有远程邻居的节点;

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
    # 内部点中，不是marginal_nodes的都认为是central_nodes; 然后再加上HALO的nodes(?)
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
    # 对于 c_mask 中的前 num_inner 个元素，将 new_id 对应位置设为从 0 到 num_central-1 的值
    new_id[c_mask[:num_inner]] = torch.arange(num_central, dtype=torch.long)
    # 对于 m_mask 中的前 num_inner 个元素，将 new_id 对应位置设为从 num_central 到 num_inner-1 的值
    new_id[m_mask[:num_inner]] = torch.arange(num_central, num_inner, dtype=torch.long)
    # 获取输入图的边（u 和 v）
    u, v = input_graph.edges()
    # 对于小于 num_inner 的 u 和 v 值，使用 new_id 进行替换
    u[u < num_inner] = new_id[u[u < num_inner].long()]
    v[v < num_inner] = new_id[v[v < num_inner].long()]
    # 使用更新后的 u 和 v 创建一个新的图 reordered_graph
    reordered_graph = dgl.graph((u, v))
    # 遍历 nodes_feats 中的每个键，将其特征根据 new_id 进行重新排列
    for key in nodes_feats: nodes_feats[key][new_id] = nodes_feats[key].clone()[0:num_inner]
    # 遍历 send_idx 中的每个键，按照 new_id 对其进行重新排列
    for key in send_idx: send_idx[key] = new_id[send_idx[key]]
    return reordered_graph, nodes_feats, send_idx

def convert_send_idx(original_send_idx: Basic_Buffer_Type) -> Tuple[Dict[int, Tuple[int, int]], Tensor]:
    '''
    将 send_idx 的原始布局转换为offset fashion。
    
    `ATTENTION`: this function should be called after graph reordering.
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

