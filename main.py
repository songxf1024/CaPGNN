import os
import time

from CaPGNN.util.caches import CACHEALG, create_cache, CACHEMAP
from gpu import cal_gpus_capability

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    import torchdata
    torchdata._warning_shown = True
except ImportError:
    pass
from multiprocessing import Manager
import tools
from CaPGNN.server import StorageServer
from CaPGNN.util.utils import set_cpu_affinity, set_random_seeds
import argparse
import warnings
import torch.nn as nn
import torch.distributed as dist
import torch
from tqdm import tqdm
from CaPGNN import Trainer
from cspart import GraphInfo
import torch
import swanlab
torch.set_printoptions(precision=8, sci_mode=False)
warnings.filterwarnings('ignore')
# only for debug !!!!
# os.environ["NCCL_BLOCKING_WAIT"] = "1"
# os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# A simple forward transmission
def warmup(device):
    # Define a network
    model = nn.Sequential(
        nn.Linear(8, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    inputs = torch.rand(32, 8)
    model.to(device)
    inputs = inputs.to(device)
    for _ in tqdm(range(300), desc='warmup...'):
        out = model(inputs)
        loss = out.sum()
        loss.backward()
    torch.cuda.synchronize()

def main(rank, args, storage_server):
    if rank==0 and args.swanlab: swanlab.init(project="dist_gnn", experiment_name=f"[{rank}] {args.experiment_name}",)
    # cpu_affinity = set_cpu_affinity(rank=0, num_cores_per_gpu=20, start_core_index=40)
    # print(f">> Current process is bound to CPU cores {cpu_affinity}")
    # Turn off exception detection and performance analysis to reduce overhead during training.
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(args.num_parts)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_ALGO'] = 'Tree'
    os.environ['NCCL_CHECKS_DISABLE '] = '1'
    os.environ['NCCL_CHECK_POINTERS '] = '1'

    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(rank)
    print(f"GPU {rank}: {device_name}")
    warmup(device)

    lcache_size = args.lcache_size[rank] if type(args.lcache_size)==dict else args.lcache_size
    create_cache(alg=args.cache_alg, capacity=lcache_size, singleton=True, throw_err=False)
    storage_server.init_local_sources(lcache_size, device, rank)
    trainer = Trainer(args, storage_server)
    if args.our_cache:  # and args.cache_alg==CACHEALG.JACA:  #  and rank==0
        storage_server.get_halo_count(part_size=args.num_parts, dataset=args.dataset, gpus_list=args.gpus_list, part_dir='/mnt/disk/sxf/data/part_data', our_partition=args.our_partition)
        halo_node_feats, max_subg_size = storage_server.extract_halo_features_by_score(part_size=args.num_parts, dataset=args.dataset, gpus_list=args.gpus_list, k=args.cache_topk, part_dir='/mnt/disk/sxf/data/part_data', our_partition=args.our_partition)
        storage_server.cache_server.vl_pm_size = max_subg_size
        storage_server.cache_server.init_vl_pool(max_subg_size)
        if rank==args.num_parts-1:
            print(f'>> Maximum number of subgraph nodes: {max_subg_size}')
            storage_server.add_halo_features_to_global('forward0', halo_node_feats)
        # storage_server.add_halo_features_to_local('forward0', halo_node_feats)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    # Start training based on configuration
    time_record = trainer.train(rank)
    # Save training records
    trainer.save(time_record, suffix='csv')
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed full graph training')
    parser.add_argument('--dataset', type=str, help='training dataset')
    parser.add_argument('--model_name', type=str, default='sage', help='model for training, gcn or sage')
    parser.add_argument('--num_parts', type=int, default=2, help='number of partitions')
    parser.add_argument('--backend', type=str, default='gloo', help='backend for distributed training')
    parser.add_argument('--init_method', type=str, default='env://', help='init method for distributed training')
    parser.add_argument('--mode', type=str, default='Vanilla')
    parser.add_argument('--assign_scheme', type=str, default='uniform')
    parser.add_argument('--logger_level', type=str, default='INFO', help='logger level')
    parser.add_argument('--gcache_size', type=int, default=5000, help='')
    parser.add_argument('--lcache_size', type=int, default=5000, help='')
    parser.add_argument("-d", "--dataset_index", type=int, help="Set the dataset index.")
    parser.add_argument("-g", "--gpus_index", type=int, help="Set the dataset index.")
    parser.add_argument("-n", "--part_num", type=int, help="Set the number of partitions.")
    args = parser.parse_args()

    # ------------Parameter main definition area---------------- #
    '''
        // The GPUs on the two servers are different, and the other configurations are the same
        228: 
        0, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processnt=82)
        1, NVIDIA GeForce RTX 3060, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=12044MB, multi_processnt=28)
        2, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processnt=82)
        3, NVIDIA GeForce RTX 3060, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=12044MB, multi_processnt=28)
        4, NVIDIA A40, _CudaDeviceProperties(name='NVIDIA A40', major=8, minor=6, total_memory=45416MB, multi_processor_count=84)
        5, NVIDIA GeForce GTX 1660 Ti, _CudaDeviceProperties(name='NVIDIA GeForce GTX 1660 Ti', major=7, minor=5, total_memory=5936MB, multi_prr_count=24)
        6, NVIDIA A40, _CudaDeviceProperties(name='NVIDIA A40', major=8, minor=6, total_memory=45416MB, multi_processor_count=84)
        7, NVIDIA GeForce GTX 1660 Ti, _CudaDeviceProperties(name='NVIDIA GeForce GTX 1660 Ti', major=7, minor=5, total_memory=5936MB, multi_prr_count=24)
        229:
        0, NVIDIA GeForce GTX 1650, _CudaDeviceProperties(name='NVIDIA GeForce GTX 1650', major=7, minor=5, total_memory=3902MB, multi_processor_count=14)
        1, NVIDIA GeForce GTX 1650, _CudaDeviceProperties(name='NVIDIA GeForce GTX 1650', major=7, minor=5, total_memory=3902MB, multi_processor_count=14)
        2, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processor_count=82)
        3, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processor_count=82)
        4, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processor_count=82)
        5, NVIDIA GeForce RTX 2060 SUPER, _CudaDeviceProperties(name='NVIDIA GeForce RTX 2060 SUPER', major=7, minor=5, total_memory=7974MB, multi_processor_count=34)
        6, NVIDIA GeForce RTX 2060 SUPER, _CudaDeviceProperties(name='NVIDIA GeForce RTX 2060 SUPER', major=7, minor=5, total_memory=7974MB, multi_processor_count=34)
        7, NVIDIA GeForce RTX 3090, _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processor_count=82)
    '''
    gpu_groups          = {
        '228': [
            # 0        1        2        3        4        5        6        7         8        9
            [],
            ['0', ],
            ['0,2', '4,0', '1,0'],
            ['4,0,2', ],
            ['6,4,0,2', '3,1,0,2', '3,1,6,4'],
            ['1,6,4,0,2', ],
            ['3,1,6,4,0,2'],
            ['5,3,1,6,4,0,2'],
            ['7,5,3,1,6,4,0,2'],
        ],
        '229': [
            # 0        1        2        3        4        5        6        7         8        9
            [],
            ['2', ],
            ['2,3', ],
            ['2,7,3', ],
            ['4,2,7,3', ]
        ]
    }
    dataset_groups      = [
        # dataset                 , feature dimensions
        ('ogbn-arxiv'             , 128),   # 0 
        ('ogbn-products'          , 100),   # 1
        ('cite'                   , 3703),  # 2    
        ('cora'                   , 1433),  # 3
        ('flickr'                 , 500),   # 4
        ('yelp'                   , 300),   # 5
        ('reddit'                 , 602),   # 6
        ('amazonProducts'         , 200),   # 7
        ('amazonCoBuyComputer'    , 767),   # 8
        ('coauthorPhysics'        , 8415),  # 9
        ('coraFull'               , 8710),  # 10
    ]
    policy_map  = {
        'adaqp'    : {'our_cache': False, 'our_partition': False},
        'cache'    : {'our_cache': True , 'our_partition': False},
        'parti'    : {'our_cache': False, 'our_partition': True },
        'all'      : {'our_cache': True , 'our_partition': True },
    }
    # --------------------------------------- #
    # python main.py --dataset_index=6 --part_num=4 --gpus_index=0
    # python main.py -d=6              -n=4         -g=0
    # ------------Main parameter modification area---------------- #
    dataset_index       = args.dataset_index    or 6
    partition_num       = args.part_num         or 6
    gpus_index          = args.gpus_index       if args.gpus_index is not None else 0
    args.num_epoches    = 200                                        # Total training rounds
    args.learning_rate  = 0.01                                       # Learning rate. Can be used with scheduler
    args.model_name     = ['gcn', 'sage'][0]                         # Which model to use
    policy              = 3                                          # Which strategy to use
    our_policy          = ['adaqp', 'cache', 'parti', 'all'][policy] # Which strategy to use
    args.cache_alg      = ['jaca', 'lru', 'fifo', 'rand', ][0]       # Which cache method to use
    args.use_pipeline   = [False, True][1]                           # Whether to use pipelines
    args.eval           = [False, True][0]                           # Whether each round is over to verify. Note that it will increase timing
    args.scaler         = [False, True][1]                           # Whether to use precision scaling
    args.pretrain       = [False, True][1]                           # Whether to perform pre-training

    args.usecast        = [False, True][1]                           # Whether to use mix precision
    args.reducer        = [False, True][0]                           # Whether to use asynchronous gradient synchronization
    args.cvt_fmts       = [False, True][1]                           # Whether to generate all sparse matrix formats
    args.do_reorder     = [False, True][1]                           # Whether to use sub-graph rearrangement. It is recommended to keep it True
    args.swanlab        = [False, True][0]                           # Whether to upload records to swanlab. valid when eval=True
    args.scheduler      = [False, True][0]                           # Whether to use automatic learning rate. valid when eval=True
    # --------------------------------------- #
    # ------------Automatic parameter configuration area---------------- #
    args.port           = f'29{dataset_index%10}{partition_num}{policy}'
    args.swanlab        = args.swanlab and args.eval
    args.our_cache      = policy_map[our_policy]['our_cache']
    args.our_partition  = policy_map[our_policy]['our_partition']
    args.cache_alg      = CACHEMAP[args.cache_alg]
    dims                = [dataset_groups[dataset_index][1], 256, 256]
    args.n_layers       = len(dims)
    args.gpus           = gpu_groups[tools.outer_ip][partition_num][gpus_index]
    args.gpus_list      = list(map(int, args.gpus.split(',')))
    args.num_parts      = partition_num
    args.dataset        = dataset_groups[dataset_index][0]
    args.cache_topk     = -1
    args.experiment_name = f'model={args.model_name}|policy={our_policy}|dataset={dataset_index}|parts={partition_num}|cache={args.gcache_size}/{args.lcache_size}|pipeline={"T" if args.use_pipeline else "F"}|eval={"T" if args.eval else "F"}|pretrain={"T" if args.pretrain else "F"}'
    # --------------------------------------- #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cal_gpus_capability(args.gpus_list)
    # set_random_seeds(42)
    print("-" * 50)
    print(f'>> List of GPUs used: [{gpus_index}] => {args.gpus}')
    print(f'>> The name of the dataset: {args.dataset}')
    print(f'>> Feature dimensions of each layer: {dims}')
    print(f'>> Perform a verification test: {args.eval}')
    print(f'>> Using node rearrangement: {args.do_reorder}')
    print(f'>> Convert formats in advance: {args.cvt_fmts}')
    print(f'>> √Using caching policies: {args.our_cache}')
    print(f'>> √Using partitioning methods: {args.our_partition}')
    print(f'>> √Using pipelines: {args.use_pipeline}')
    print(f'>> Models used: {args.model_name}')
    if args.our_cache:
        # 5000,10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,
        # for cache_size in [10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,]:
        #     args.lcache_size = cache_size  # int(0.2*args.lcache_size)
        #     args.gcache_size = cache_size  # int(0.2*args.gcache_size)
        args.gcache_size, args.lcache_size = StorageServer.cal_capacity(part_size=args.num_parts, dataset=args.dataset, gpus_list=args.gpus_list, f_dims=dims, part_dir='/mnt/disk/sxf/data/part_data', our_partition=args.our_partition)
        print('>> CPU cache capacity: ', args.gcache_size)
        print('>> GPU cache capacity: ', args.lcache_size)
    print("-" * 50)

    manager = Manager()
    storage_server = StorageServer(manager, gpus_num=partition_num, gsize=args.gcache_size, dims=dims, cache_alg=args.cache_alg)
    torch.multiprocessing.spawn(main, (args, storage_server), args.num_parts, join=True, daemon=True, start_method='spawn')
    print('=' * 50)

    del manager, storage_server
    time.sleep(2)
    print('\n\nEnd all!')

