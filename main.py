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

# 一个简单的前向传播
def warmup(device):
    # 定义一个网络
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
    # 关闭异常检测和性能分析，以减少训练时的开销。
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
            print(f'>> 最大的子图节点数: {max_subg_size}')
            storage_server.add_halo_features_to_global('forward0', halo_node_feats)
        # storage_server.add_halo_features_to_local('forward0', halo_node_feats)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    # 根据配置开始训练
    time_record = trainer.train(rank)
    # 保存训练记录
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

    # ------------参数主要定义区---------------- #
    '''
        228: 
        GPU索引: 0, GPU名称: NVIDIA GeForce RTX 3090, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processnt=82)
        GPU索引: 1, GPU名称: NVIDIA GeForce RTX 3060, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=12044MB, multi_processnt=28)
        GPU索引: 2, GPU名称: NVIDIA GeForce RTX 3090, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24259MB, multi_processnt=82)
        GPU索引: 3, GPU名称: NVIDIA GeForce RTX 3060, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce RTX 3060', major=8, minor=6, total_memory=12044MB, multi_processnt=28)
        GPU索引: 4, GPU名称: NVIDIA A40, GPU属性: _CudaDeviceProperties(name='NVIDIA A40', major=8, minor=6, total_memory=45416MB, multi_processor_count=84)
        GPU索引: 5, GPU名称: NVIDIA GeForce GTX 1660 Ti, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce GTX 1660 Ti', major=7, minor=5, total_memory=5936MB, multi_prr_count=24)
        GPU索引: 6, GPU名称: NVIDIA A40, GPU属性: _CudaDeviceProperties(name='NVIDIA A40', major=8, minor=6, total_memory=45416MB, multi_processor_count=84)
        GPU索引: 7, GPU名称: NVIDIA GeForce GTX 1660 Ti, GPU属性: _CudaDeviceProperties(name='NVIDIA GeForce GTX 1660 Ti', major=7, minor=5, total_memory=5936MB, multi_prr_count=24)
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
        # 数据集                   , 特征维度
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
    '''
    前期测试时用，已经替换为使用StorageServer.cal_capacity自动计算出缓存容量
    cache_max_size数据构成说明：
    cache_size = cache_max_size[tools.outer_ip][args.dataset][args.our_partition][args.num_parts][gpus_index]
                                  [服务器IP]      [数据集名称]     [是否使用分区]         [分区数]      [GPU组索引]  <= 缓存容量
                                    '228'  : 'coauthorPhysics': {   True :       {      2:    {       1:          8862   }, }  },
    '''
    cache_max_size      = {
        '228': {
            'coauthorPhysics': {
                True : {2: {0: 5095, }, 3: {0: 6432, }, 4: {0: 4562, }, 5: {0: 5130, }, 6: {0: 8011, }, 7: {0: 1584, }, 8: {0: 3407, } },
                False: {2: {0: 9430, }, 3: {0: 15913, }, 4: {0: 18676, }, 5: {0: 19904, }, }
            },
            'flickr': {
                True : {2: {0: 56258, }, 3: {0: 62523, }, 4: {0: 51327, }, 5: {0: 63877, }, 6: {0: 35127, }, 7: {0: 56232, }, 8: {0: 67891, }, },
                False: {2: {0: 56601, }, 3: {0: 66386, }, 4: {0: 69169, }, 5: {0: 72724, }, }
            },
            'reddit': {
                True : {2: {0: 68124, }, 3: {0: 81148, }, 4: {0: 137946, }, 5: {0: 91832, }, 6: {0: 137971, }, 7: {0: 84486, }, 8: {0: 73175, }, },
                False: {2: {0: 143842,}, 3: {0: 183030,}, 4: {0: 184796, }, 5: {0: 193032,}, 6: {0: 214983, }, 7: {0: 206183,}, 8: {0: 208836, }}
            },
            'yelp': {
                True : {2: {0: 138011, }, 3: {0: 146456, }, 4: {0: 163150, }, 5: {0: 101171, }, 6: {0: 158378, }, 7: {0: 257776, }, 8: {0: 253573, }, },
                False: {2: {0: 256104, }, 3: {0: 329851, }, 4: {0: 351357, }, 5: {0: 377335, }, }
            },
            'coraFull': {
                True : {2: {0: 835, }, 3: {0: 2845, }, 4: {0: 1615, }, 5: {0: 3656, }, 6: {0: 3256, }, 7: {0: 3447, }, 8: {0: 3983, } },
                False: {2: {0: 1685, }, 3: {0: 3832, }, 4: {0: 4728, }, 5: {0: 5253, }, }
            },
            'ogbn-products': { #                                                                                  80                110
                True : {2: {0: 234907, }, 3: {0: 179163, }, 4: {0: 182500, }, 5: {0: 307121, }, 6: {0: 607750, }, 7: {0: 59602, }, 8: {0: 142295, }, },
                False: {2: {0: 444209, }, 3: {0: 677482, }, 4: {0: 825510, }, 5: {0: 877016, }, }
            },
            'amazonProducts': { #                                                                                                   500...
                True : {2: {0: 271693, }, 3: {0: 156015, }, 4: {0: 135114, }, 5: {0: 259760, }, 6: {0: 164521, }, 7: {0: 259760, }, 8: {0: 0, }, },
                False: {2: {0: 496490, }, 3: {0: 582151, }, 4: {0: 677847, }, 5: {0: 668084, }, }
            },
            'amazonCoBuyComputer': {
                True : {2: {0: 2598, }, 3: {0: 4046, }, 4: {0: 2149, }, 5: {0: 2718, }, },
                False: {2: {0: 4888, }, 3: {0: 7009, }, 4: {0: 8458, }, 5: {0: 8157, }, }
            },
        },
        '229': {
            'flickr'            : {True : {2: {0: 56367  }, 3: {0: 60722  }, 4: {0: 53593   }},   # 使用分区
                                   False: {2: {0: 56601  }, 3: {0: 66386  }, 4: {0: 69169   }}},  # 不使用分区
            'reddit'            : {True : {2: {0: 98507 }, 3: {0: 73788  }, 4: {0: 116524  }},   # 使用分区
                                   False: {2: {0: 181683 }, 3: {0: 172413 }, 4: {0: 185846  }}},  # 不使用分区
            'coraFull'          : {True : {2: {0: 881    }, 3: {0: 1681   }, 4: {0: 1792    }},   # 使用分区
                                   False: {2: {0: 1840   }, 3: {0: 3832   }, 4: {0: 4677    }}},  # 不使用分区
            'coauthorPhysics'   : {True : {2: {0: 5095   }, 3: {0: 6476   }, 4: {0: 5057    }} ,  # 使用分区
                                   False: {2: {0: 9430   }, 3: {0: 15913  }, 4: {0: 18676   }}},  # 不使用分区
            'ogbn-products'     : {True : {2: {0: 234907 }, 3: {0: 167303 }, 4: {0: 187753  }},   # 使用分区
                                   False: {2: {0: 444209 }, 3: {0: 677482 }, 4: {0: 825510  }}},  # 不使用分区
            'amazonProducts'    : {True : {2: {0: 271693 }, 3: {0: 241294 }, 4: {0: 223406  }},   # 使用分区
                                   False: {2: {0: 496400 }, 3: {0: 547328 }, 4: {0: 626519  }}},  # 不使用分区. acc is low
            'yelp'              : {True : {2: {0: 256104 }, 3: {0: 329851 }, 4: {0: 351357  }},   # 使用分区
                                   False: {2: {0: 141304 }, 3: {0: 171324 }, 4: {0: 164538  }}},   # 不使用分区
        }
    }
    policy_map          = {
        'adaqp'    : {'our_cache': False, 'our_partition': False},
        'cache'    : {'our_cache': True , 'our_partition': False},
        'parti'    : {'our_cache': False, 'our_partition': True },
        'all'      : {'our_cache': True , 'our_partition': True },
    }
    # --------------------------------------- #
    # python main.py --dataset_index=6 --part_num=4 --gpus_index=0
    # python main.py -d=6              -n=4         -g=0
    # ------------参数主要修改区---------------- #
    dataset_index       = args.dataset_index    or 6
    partition_num       = args.part_num         or 6
    gpus_index          = args.gpus_index       if args.gpus_index is not None else 0
    args.num_epoches    = 200                                        # 总训练轮数
    args.learning_rate  = 0.01                                       # 学习率. 可以搭配scheduler使用
    args.model_name     = ['gcn', 'sage'][0]                         # 使用哪种模型
    policy              = 3                                          # 执行哪种策略
    our_policy          = ['adaqp', 'cache', 'parti', 'all'][policy] # 执行哪种策略
    args.cache_alg      = ['jaca', 'lru', 'fifo', 'rand', ][0]       # 使用哪种缓存方法
    args.use_pipeline   = [False, True][1]                           # 是否使用流水线
    args.eval           = [False, True][0]                           # 是否每轮结束进行验证. 注意会导致计时增加
    args.scaler         = [False, True][1]                           # 是否使用精度缩放
    args.pretrain       = [False, True][1]                           # 是否进行预训练

    args.usecast        = [False, True][1]                           # 是否使用mix精度
    args.reducer        = [False, True][0]                           # 是否使用异步梯度同步
    args.cvt_fmts       = [False, True][1]                           # 是否生成所有稀疏矩阵格式
    args.do_reorder     = [False, True][1]                           # 是否使用子图重排. 建议保持为True
    args.swanlab        = [False, True][0]                           # 是否上传记录到swanlab. eval=True时才有效
    args.scheduler      = [False, True][0]                           # 是否使用自动学习率. eval=True时才有效
    args.enable_back    = [False, True][0]                           # 是否反向传播阶段也参与（未使用）
    # --------------------------------------- #
    # ------------参数自动配置区---------------- #
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
    print(f'>> 所用GPU列表: [{gpus_index}] => {args.gpus}')
    print(f'>> 数据集的名称: {args.dataset}')
    print(f'>> 各层特征维度: {dims}')
    print(f'>> 进行验证测试: {args.eval}')
    print(f'>> 使用节点重排: {args.do_reorder}')
    print(f'>> 提前转换格式: {args.cvt_fmts}')
    print(f'>> √使用缓存策略: {args.our_cache}')
    print(f'>> √使用分区方法: {args.our_partition}')
    print(f'>> √使用流水线: {args.use_pipeline}')
    print(f'>> 使用的模型: {args.model_name}')
    if args.our_cache:
        # args.gcache_size = cache_max_size[tools.outer_ip][args.dataset][args.our_partition][args.num_parts][gpus_index] if args.our_cache else 0
        # args.lcache_size = cache_max_size[tools.outer_ip][args.dataset][args.our_partition][args.num_parts][gpus_index] if args.our_cache else 0
        # 5000,10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,
        # for cache_size in [10000,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000,220000,240000,260000,]:
        #     args.lcache_size = cache_size  # int(0.2*args.lcache_size)
        #     args.gcache_size = cache_size  # int(0.2*args.gcache_size)
        args.gcache_size, args.lcache_size = StorageServer.cal_capacity(part_size=args.num_parts, dataset=args.dataset, gpus_list=args.gpus_list, f_dims=dims, part_dir='/mnt/disk/sxf/data/part_data', our_partition=args.our_partition)
        print('>> CPU缓存容量: ', args.gcache_size)
        print('>> GPU缓存容量: ', args.lcache_size)
    print("-" * 50)

    manager = Manager()
    storage_server = StorageServer(manager, gpus_num=partition_num, gsize=args.gcache_size, dims=dims, cache_alg=args.cache_alg)
    torch.multiprocessing.spawn(main, (args, storage_server), args.num_parts, join=True, daemon=True, start_method='spawn')
    print('=' * 50)

    del manager, storage_server
    time.sleep(2)
    print('\n\n全部结束!')

