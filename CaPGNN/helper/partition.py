import os
import dgl
import numpy as np
import pandas
import requests
import torch
import torch_geometric
from dgl import DGLHeteroGraph
from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset
from .dataset import AmazonProducts, load_yelp


def _download(url, path, filename):
    fn = os.path.join(path, filename)
    if os.path.exists(fn): return
    os.makedirs(path, exist_ok=True)
    f_remote = requests.get(url, stream=True)
    sz = f_remote.headers.get("content-length")
    assert f_remote.status_code == 200, "fail to open {}".format(url)
    with open(fn, "wb") as writer:
        for chunk in f_remote.iter_content(chunk_size=1024 * 1024):
            writer.write(chunk)
    print("Download finished.")

def get_friendster(raw_dir, format=None):
    # from dgl_benchmarks
    if isinstance(format, str): format = [format]  # didn't specify format
    if format is None: format = ["csc", "csr", "coo"]
    bin_path = f"{raw_dir}/friendster/friendster_{format}.bin"
    if os.path.exists(bin_path):
        g_list, _ = dgl.load_graphs(bin_path)
        g = g_list[0]
    else:
        print("downloading...")
        # Same as https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
        _download(
            "https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/friendster/com-friendster.ungraph.txt.gz",
            f"{raw_dir}/friendster",
            "com-friendster.ungraph.txt.gz",
        )
        print("reading...")
        df = pandas.read_csv(
            f"{raw_dir}/friendster/com-friendster.ungraph.txt.gz",
            sep="\t",
            skiprows=4,
            header=None,
            names=["src", "dst"],
            compression="gzip",
        )
        src = df["src"].values
        dst = df["dst"].values
        print("construct the graph")
        # the original node IDs of friendster are not consecutive, so we compact it
        g = dgl.compact_graphs(dgl.graph((src, dst))).formats(format)
        dgl.save_graphs(bin_path, [g])
        print("complete")
    return g

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
        print('This dataset is not yet supported')
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
    Partition and store the dataset
    we set HALO hop as 1 to save cross-device neighboring nodes' indices for constructing send/recv idx.
    '''
    dgl_dataset_map = {
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
        dgl_dataset_map.update({
            'amazonRatings': dgl.data.AmazonRatingsDataset,              # dglv2
            'tolokers': dgl.data.TolokersDataset,
        })
    
    pyg_dataset_map = {
        'reddit2': torch_geometric.datasets.Reddit2,
        'wikidata5M': torch_geometric.datasets.Wikidata5M,
    }

    # the dir to store graph partition
    partition_dir = '{}/{}/{}part'.format(part_dir, dataset, partition_size)
    # if os.path.exists(partition_dir):
    #     return
    if 'ogb' in dataset:
        # OGB dataset needs to be processed
        graph = process_obg_dataset(dataset, raw_dir)
    elif dataset == 'yelp':
        # PyG dataset needs to be processed
        # data = torch_geometric.datasets.Yelp(root=os.path.join(raw_dir, dataset))
        data = dgl_dataset_map[dataset](raw_dir=raw_dir)
        graph = load_yelp(raw_dir=raw_dir)
    elif dgl_dataset_map.get(dataset):
        # All DGL datasets can be used directly
        data = dgl_dataset_map[dataset](raw_dir=raw_dir)
        graph = data[0]
        # If no mask is provided, it is necessary to generate it manually randomly
        if graph.ndata.get('train_mask') is None:
            print('>> The mask is generated manually here, there may be problems, please note!')
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
            # # Randomly disrupt node index
            # perm = torch.randperm(num_nodes)
            # # Generate mask
            # train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            # train_mask[perm[:train_size]] = True
            # val_mask[perm[train_size:train_size + val_size]] = True
            # test_mask[perm[train_size + val_size:]] = True
            # # Add mask to the node data of the graph
            # graph.ndata['train_mask'] = train_mask
            # graph.ndata['val_mask'] = val_mask
            # graph.ndata['test_mask'] = test_mask
    elif pyg_dataset_map.get(dataset):
        data = pyg_dataset_map[dataset](root=os.path.join(raw_dir, dataset))  # , name=dataset
        graph = torch_geometric.utils.to_dgl(data[0])
    elif dataset == 'friendster':
        graph = get_friendster(raw_dir)
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

