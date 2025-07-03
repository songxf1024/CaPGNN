import os
import time
from CaPGNN.util.caches import CACHEALG, create_cache, CACHEMAP
import gpu
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

# A simple forward propagation
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

def main(local_rank, args, storage_server):
    if local_rank==0 and args.swanlab: swanlab.init(project="dist_gnn", experiment_name=f"[{local_rank}] {args.experiment_name}",)
    # cpu_affinity = set_cpu_affinity(local_rank=0, num_cores_per_gpu=20, start_core_index=40)
    # print(f">> Current process is bound to CPU cores {cpu_affinity}")
    # Disable anomaly detection and performance analysis to reduce overhead during training.
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    os.environ['MASTER_ADDR'] = '202.38.247.228'
    os.environ['MASTER_PORT'] = args.port
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno1'  # 'lo'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'  # 'lo'
    os.environ['NODE_RANK'] = str(args.server_id)
    os.environ['RANK'] = str(args.server_id * args.num_parts[args.server_id] + local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['LOCAL_WORLD_SIZE'] = str(args.num_parts[args.server_id])
    os.environ['WORLD_SIZE'] = str(args.server_num*args.num_parts[args.server_id])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_ALGO'] = 'Tree'
    os.environ['NCCL_CHECKS_DISABLE'] = '1'
    os.environ['NCCL_CHECK_POINTERS'] = '1'
    os.environ['NCCL_DEBUG'] = '1'

    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(local_rank)
    print(f"GPU {local_rank}: {device_name}")
    warmup(device)

    lcache_size = args.lcache_size[local_rank] if type(args.lcache_size)==dict else args.lcache_size
    create_cache(alg=args.cache_alg, capacity=lcache_size, singleton=True, throw_err=False)
    t1 = time.time()
    storage_server.init_local_sources(lcache_size, device, local_rank)
    print('>> init cache: ', time.time() - t1)
    trainer = Trainer(args, storage_server)
    if args.our_cache:  # and args.cache_alg==CACHEALG.JACA:  #  and rank==0
        storage_server.get_halo_count(args=args, part_dir='/mnt/disk/sxf/data/part_data')
        halo_node_feats, max_subg_size = storage_server.extract_halo_features_by_score(args=args, k=args.cache_topk, part_dir='/mnt/disk/sxf/data/part_data')
        storage_server.cache_server.vl_pm_size = max_subg_size
        storage_server.cache_server.init_vl_pool(max_subg_size)
        if local_rank==args.num_parts[args.server_id]-1:
            print(f'>> Maximum number of subgraph nodes: {max_subg_size}')
            storage_server.add_halo_features_to_global('forward0', halo_node_feats)
            print(f'[{local_rank}] add_halo_features_to_global done')
        # storage_server.add_halo_features_to_local('forward0', halo_node_feats)
    print(f'[{local_rank}] Preparing...')
    torch.cuda.synchronize()
    # dist.barrier()
    torch.cuda.empty_cache()
    # Start training according to the configuration
    time_record = trainer.train(local_rank)
    # Save training records
    trainer.save(time_record, suffix='csv')
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='distributed full graph training')
    parser.add_argument('--dataset', type=str, help='training dataset')
    parser.add_argument('--model_name', type=str, default='sage', help='model for training, gcn or sage')
    parser.add_argument('--backend', type=str, default='gloo', help='backend for distributed training')
    parser.add_argument('--init_method', type=str, default='env://', help='init method for distributed training')
    parser.add_argument('--mode', type=str, default='Vanilla')
    parser.add_argument('--assign_scheme', type=str, default='uniform')
    parser.add_argument('--logger_level', type=str, default='INFO', help='logger level')
    parser.add_argument('--gcache_size', type=int, default=5000, help='')
    parser.add_argument('--lcache_size', type=int, default=5000, help='')
    parser.add_argument("-s", "--server_num", type=int, help="Set the number of servers.")
    parser.add_argument("-i", "--server_id", type=int, default=0, help="Which server, the same as node_rank")
    parser.add_argument("-d", "--dataset_index", type=int, help="Set the dataset index.")
    parser.add_argument("-g", "--gpus_index", type=int, help="Set the dataset index.")
    parser.add_argument("-n", "--num_parts", type=str, help="Set the number of partitions of each server. E.g., 4 or 2,3")
    args = parser.parse_args()

    # ------------Parameter Main Definition Area---------------- #
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
        # dataset                 , feature dimension
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
        ('tolokers'               , 100),   # 11
        ('ogbn-papers100M'        , 128),   # 12
        ('friendster'             , 256),   # 13
        ('wikidata5M'             , 128),   # 14
    ]
    policy_map          = {
        'adaqp'    : {'our_cache': False, 'our_partition': False},
        'cache'    : {'our_cache': True , 'our_partition': False},
        'parti'    : {'our_cache': False, 'our_partition': True },
        'all'      : {'our_cache': True , 'our_partition': True },
    }
    # --------------------------------------- #
    # python main.py --dataset_index=6 --num_parts=4 --gpus_index=0
    # python main.py -d=6  -n=4    -g=0  -s=1
    # python main.py -d=4  -n=2,2  -g=0  -s=2  -i=0
    # ------------Parameter main modification area---------------- #
    dataset_index       = args.dataset_index    or 6
    args.num_parts      = args.num_parts        or "2,2"
    args.server_num     = args.server_num       or 2
    args.gpus_index     = args.gpus_index       if args.gpus_index is not None else 0
    args.num_epoches    = 200                                        # Total number of training rounds
    args.learning_rate  = 0.01                                       # Learning rate. Can be used with the scheduler.
    args.model_name     = ['gcn', 'sage'][0]                         # Which model to use
    policy              = 3                                          # Which strategy to use
    our_policy          = ['adaqp', 'cache', 'parti', 'all'][policy] # Which strategy to use
    args.cache_alg      = ['jaca', 'lru', 'fifo', 'rand', ][0]       # Which caching method to use
    args.use_pipeline   = [False, True][1]                           # Whether to use a pipeline
    args.eval           = [False, True][1]                           # Whether to verify at the end of each round. Note that it will lead to an increase in timing.
    args.scaler         = [False, True][1]                           # Whether to use precision scaling
    args.pretrain       = [False, True][1]                           # Whether to perform pre-training

    args.usecast        = [False, True][1]                           # Whether to use mixed precision
    args.reducer        = [False, True][0]                           # Whether to use asynchronous gradient synchronization
    args.cvt_fmts       = [False, True][1]                           # Whether to generate all sparse matrix formats
    args.do_reorder     = [False, True][1]                           # Whether to use subgraph rearrangement. It is recommended to keep it as True.
    args.swanlab        = [False, True][0]                           # Whether to upload records to swanlab. Valid only when eval=True.
    args.scheduler      = [False, True][0]                           # Whether to use automatic learning rate. It is only effective when eval=True.
    args.enable_back    = [False, True][0]                           # Does the backpropagation phase also participate (not used)
    # --------------------------------------- #
    # ------------Parameter Auto Configuration Zone---------------- #
    args.num_parts      = [int(x) for x in args.num_parts.split(',')]
    # args.port           = f'29{dataset_index%10}{args.num_parts[args.server_id]}{policy}'
    args.port           = f'29500'
    args.swanlab        = args.swanlab and args.eval
    args.our_cache      = policy_map[our_policy]['our_cache']
    args.our_partition  = policy_map[our_policy]['our_partition']
    args.cache_alg      = CACHEMAP[args.cache_alg]
    dims                = [dataset_groups[dataset_index][1], 256, 256]
    args.n_layers       = len(dims)
    gpu.init_gpus(args.server_id)
    if args.server_num > 1:
        args.gpus       = gpu_groups[gpu.all_servers[args.server_id]][args.num_parts[args.server_id]][args.gpus_index]
    else:
        args.gpus       = gpu_groups[tools.outer_ip][args.num_parts[args.server_id]][args.gpus_index]
    args.gpus_list      = list(map(int, args.gpus.split(',')))
    args.dataset        = dataset_groups[dataset_index][0]
    args.cache_topk     = -1
    args.experiment_name = f'model={args.model_name}|policy={our_policy}|dataset={dataset_index}|parts={args.num_parts[args.server_id]}|cache={args.gcache_size}/{args.lcache_size}|pipeline={"T" if args.use_pipeline else "F"}|eval={"T" if args.eval else "F"}|pretrain={"T" if args.pretrain else "F"}'
    # --------------------------------------- #
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    gpu.cal_gpus_capability(args.gpus_list)
    # set_random_seeds(42)
    print("-" * 50)
    print(f'>> Use GPUs: [{args.gpus_index}] => {args.gpus}')
    print(f'>> Dataset: {args.dataset}')
    print(f'>> Feature dimensions: {dims}')
    print(f'>> Use verification: {args.eval}')
    print(f'>> Use node reorder: {args.do_reorder}')
    print(f'>> Pre-convert format: {args.cvt_fmts}')
    print(f'>> √Use caching strategy: {args.our_cache}')
    print(f'>> √Use our partition: {args.our_partition}')
    print(f'>> √Use pipeline: {args.use_pipeline}')
    print(f'>> Use model: {args.model_name}')
    if args.our_cache:
        # 5000,10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,
        # for cache_size in [10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,]:
        #     args.lcache_size = cache_size  # int(0.2*args.lcache_size)
        #     args.gcache_size = cache_size  # int(0.2*args.gcache_size)
        args.gcache_size, args.lcache_size = StorageServer.cal_capacity(args=args, f_dims=dims, part_dir='/mnt/disk/sxf/data/part_data')
        print('>> CPU cache capacity: ', args.gcache_size)
        print('>> GPU cache capacity: ', args.lcache_size)
    print("-" * 50)

    manager = Manager()
    t1 = time.time()
    storage_server = StorageServer(manager, gpus_num=args.num_parts[args.server_id], gsize=args.gcache_size, dims=dims, cache_alg=args.cache_alg)
    print('>> storage server: ', time.time() - t1)
    torch.multiprocessing.spawn(main, (args, storage_server), args.num_parts[args.server_id], join=True, daemon=True, start_method='spawn')
    print('=' * 50)

    del manager, storage_server
    time.sleep(2)
    print('\n\nAll finished.!')

