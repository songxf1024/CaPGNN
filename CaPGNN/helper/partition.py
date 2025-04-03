import os
import dgl
import numpy as np
import torch
import torch_geometric
from dgl import DGLHeteroGraph
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset

from .dataset import AmazonProducts, load_yelp


def process_obg_dataset(dataset: str, raw_dir: str):
    '''
    process the ogb dataset, return a dgl graph with node features, labels and train/val/test masks.
    '''
    if dataset in ['ogbn-products',
                   'ogbn-arxiv',   # Nodes: 169343, Edges: 1166243, Classes: 40, Dim: 128
                   'ogbn-proteins']:
        data = DglNodePropPredDataset(name=dataset, root=raw_dir)
    # elif dataset in ['ogbg-molhiv', ]:
    #     data = DglGraphPropPredDataset(name=dataset, root=raw_dir)
    else:
        print('该数据集暂不支持')
        return None
    graph, labels = data[0]
    labels = labels[:, 0]
    graph.ndata['label'] = labels
    # split the dataset into tain/val/test before partitioning
    splitted_idx = data.get_idx_split()
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    return graph

def graph_patition_store(dataset: str, partition_size: int, raw_dir: str = 'dataset', part_dir: str = 'part_data', num_hops=1, part_method='metis'):
    '''
    对数据集进行分区并存储
    we set HALO hop as 1 to save cross-device neighboring nodes' indices for constructing send/recv idx.
    '''
    dataset_map = {
        'reddit': dgl.data.RedditDataset,                            # Nodes: 232965, Edges: 114615892, Classes: 41, Dim: 602
        'amazonProducts': AmazonProducts,
        'cora': dgl.data.CoraGraphDataset,                           # Nodes: 2708,   Edges: 10556,     Classes: 7,  Dim: 1433
        'amazonCoBuyComputer': dgl.data.AmazonCoBuyComputerDataset,  # Nodes: 13752,  Edges: 491722,    Classes: 10
        'yelp': dgl.data.YelpDataset,
        'flickr': dgl.data.FlickrDataset,                             # Nodes: 89250,  Edges: 899756,    Classes: 7,  Dim: 500
        'cite': dgl.data.CiteseerGraphDataset,                       # Nodes: 3327,   Edges: 9228,      Classes: 6,  Dim: 3703
        'coauthorPhysics': dgl.data.CoauthorPhysicsDataset,          # Nodes: 34493,  Edges: 495924,    Classes: 5,  Dim: 8415
        'coraFull': dgl.data.CoraFullDataset,
    }
    if dgl.__version__ > '1.0':
        dataset_map.update({
            'amazonRatings': dgl.data.AmazonRatingsDataset,              # dglv2
            'tolokers': dgl.data.TolokersDataset,
        })


    # the dir to store graph partition
    partition_dir = '{}/{}/{}part'.format(part_dir, dataset, partition_size)
    # if os.path.exists(partition_dir):
    #     return
    if 'ogb' in dataset:
        # OGB的数据集需要处理一下
        graph = process_obg_dataset(dataset, raw_dir)
    elif dataset == 'yelp':
        # PyG的数据集更需要处理一下
        # data = torch_geometric.datasets.Yelp(root=os.path.join(raw_dir, dataset))
        data = dataset_map[dataset](raw_dir=raw_dir)
        graph = load_yelp(raw_dir=raw_dir)
    elif dataset_map.get(dataset):
        # DGL的数据集都可以直接用
        data = dataset_map[dataset](raw_dir=raw_dir)
        graph = data[0]
        # 如果没有提供mask，就需要手动随机生成
        if graph.ndata.get('train_mask') is None:
            print('>> 这里手动生成了mask，可能会有问题，请注意!')
            n = graph.number_of_nodes()
            n_train = int(n * 0.7)
            n_val = int(n * 0.2)
            # n_test = int(n * 0.2)
            idx = np.random.permutation(n)
            idx = np.sort(idx)
            train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
            train_mask[idx[:n_train]] = True
            val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
            val_mask[idx[n_train:n_train+n_val]] = True
            test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
            test_mask[idx[n_train+n_val:]] = True
            graph.ndata['train_mask'] = train_mask
            graph.ndata['val_mask'] = val_mask
            graph.ndata['test_mask'] = test_mask

            # num_nodes = graph.number_of_nodes()
            # train_size = int(num_nodes * 0.7)
            # val_size = int(num_nodes * 0.2)
            # # 随机打乱节点索引
            # perm = torch.randperm(num_nodes)
            # # 生成掩码
            # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # train_mask[perm[:train_size]] = True
            # val_mask[perm[train_size:train_size + val_size]] = True
            # test_mask[perm[train_size + val_size:]] = True
            # # 将掩码添加到图的节点数据中
            # graph.ndata['train_mask'] = train_mask
            # graph.ndata['val_mask'] = val_mask
            # graph.ndata['test_mask'] = test_mask

    else:
        raise ValueError(f'no such dataset: {dataset}')

    # add self loop
    graph.edata.clear()
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    # save global degrees
    in_degrees = graph.in_degrees()
    out_degrees = graph.out_degrees()
    save_dir = f'{partition_dir}/graph_degrees'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    torch.save(in_degrees, f'{save_dir}/in_degrees.pt')
    torch.save(out_degrees, f'{save_dir}/out_degrees.pt')
    print(f'<save degrees: {save_dir}>')
    # partition the whole graph
    print('<begin partition...>')
    orig_ids = dgl.distributed.partition_graph(graph,
                                               graph_name=dataset,
                                               part_method=part_method,
                                               num_parts=partition_size,
                                               out_path=partition_dir,
                                               num_hops=num_hops,
                                               return_mapping=True)
    # torch.save(orig_ids[0], f'{save_dir}/orig_id_nodes.pt')
    # torch.save(orig_ids[1], f'{save_dir}/orig_id_edges.pt')
    # metis_partitions = dgl.metis_partition(graph, k=partition_size, extra_cached_hops=0, reshuffle=True,
    #                                        balance_ntypes=None, balance_edges=False, mode="k-way")
    # # Save each partition to disk
    # for part_id, part_graph in metis_partitions.items():
    #     part_filename = os.path.join(partition_dir, f'part{part_id}.dgl')
    #     dgl.save_graphs(part_filename, part_graph)

    # print('load')
    # partitions = []
    # for part_id in range(2):
    #     # part_filename = os.path.join(partition_dir, f'part{part_id}.dgl')
    #     part_filename = os.path.join(partition_dir, f'part0/graph.dgl')
    #     part_graphs, _ = dgl.load_graphs(part_filename)
    #     partitions.append(part_graphs[0])
    #     print(part_graphs)


if __name__ == '__main__':
    pass

