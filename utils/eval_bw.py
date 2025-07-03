import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from env import DistEnv
from util import *


def measure_bandwidth(env, sz_tag, size, repeat, mode='IDT'):
    env.logger.log(f'Size: {sz_tag: <5} Repeat: {repeat}', rank=0)
    h_data = torch.randn(size, dtype=torch.float32, device='cpu', pin_memory=False)
    d_data = torch.randn(size, dtype=torch.float32, device=env.device)

    if mode == 'IDT':
        d2_data = torch.randn(size, dtype=torch.float32, device=env.device)
        for _ in range(5): d2_data.copy_(d_data)
        torch.cuda.synchronize()
        
        for i in range(repeat):
            d_data = torch.randn(size, dtype=torch.float32, device=env.device)
            torch.cuda.synchronize()
            with env.timer.timing_cuda('IDT-' + sz_tag):
                d2_data.copy_(d_data)
            torch.cuda.synchronize()
            del d_data
            torch.cuda.empty_cache()
            env.logger.log(f'IDT {sz_tag} {i + 1}/{repeat}', oneline=True, rank=0, show=True)
        env.logger.log(f'', rank=0, show=True)
    elif mode == 'H2D':
        for _ in range(5): d_data.copy_(h_data)
        torch.cuda.synchronize()
        
        for i in range(repeat):
            h_data = torch.randn(size, dtype=torch.float32, device='cpu', pin_memory=False)
            torch.cuda.synchronize()
            with env.timer.timing_cuda('HtoD-' + sz_tag):
                d_data.copy_(h_data)
            torch.cuda.synchronize()
            del h_data
            torch.cuda.empty_cache()
            env.logger.log(f'HtoD {sz_tag} {i + 1}/{repeat}', oneline=True, rank=0, show=True)
        env.logger.log(f'', rank=0, show=True)
    elif mode == 'D2H':
        for _ in range(5): h_data.copy_(d_data) # 预热5次
        torch.cuda.synchronize()

        for i in range(repeat):
            d_data = torch.randn(size, dtype=torch.float32, device=env.device)
            torch.cuda.synchronize()
            with env.timer.timing_cuda('DtoH-' + sz_tag):
                h_data.copy_(d_data)
            torch.cuda.synchronize()
            del d_data
            torch.cuda.empty_cache()
            env.logger.log(f'DtoH {sz_tag} {i + 1}/{repeat}', oneline=True, rank=0, show=True)
        env.logger.log(f'', rank=0, show=True)
    else:
        print('a??')

def eval_bandwidth(env):
    sizes = [
        #(1000, '1K', (16, 16)),
        #(1000, '10K', (51, 51)),
        #(1000, '100K', (160, 160)),
        #(1000, '1M', (512, 512)),
        #(1000, '10M', (1620, 1620)),
        #(100, '100M', (5119, 5119)),
        (50, '512M', (11585, 11585)),
        # (10, '1G', (16384, 16384)),
        #(10, '2G', (2, 16384, 16384)),
        # (10, '3G', (3, 16384, 16384)),
        # (10, '4G', (4, 16384, 16384)),
        # (10, '5G', (5, 16384, 16384)),
        # (10, '6G', (6, 16384, 16384)),
        # (10, '7G', (7, 16384, 16384)),
        # (10, '8G', (8, 16384, 16384)),
        # (10, '9G', (9, 16384, 16384)),
        # (10, '10G', (10, 16384, 16384)),
    ]
    for repeat, tag, size in sizes:
        measure_bandwidth(env, tag, size, repeat, mode='H2D')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    for repeat, tag, size in sizes:
        measure_bandwidth(env, tag, size, repeat, mode='D2H')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    for repeat, tag, size in sizes:
        measure_bandwidth(env, tag, size, repeat, mode='IDT')
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def evaluate(env):
    warm_up_gpus(env.device)
    warm_up_pcie(env)
    torch.cuda.synchronize()
    with env.timer.timing('total'):
        eval_bandwidth(env)
    rank_summary, rank_summary_str = env.timer.summary()
    env.logger.log(rank_summary_str, prefix=('\n' + '*' * 10 + '\n'), suffix='*' * 10, show=True)
    env.logger.save_summary_to_csv(rank_summary, f'rank.csv')

if __name__ == "__main__":
    #set_cpu_affinity(0, start_core_index=40, num_cores_per_gpu=20)
    set_random_seeds()
    gpu_index = 0
    print('>> gpu group index:', gpu_index)
    # 228: 3090  3060  3090  3060  A40   1660  A40   1660
    # 229: 1650  1650  3090  3090  3090  2060  2060  3090
    gpus = ['0',  '1',  '2',  '3', '4',  '5',  '6', '7'][gpu_index]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    check_gpu_temperatures(gpus, temp_threshold=60)
    env = DistEnv(0, 1, 'nccl', log_prefix='eval_bw', gpus_note=gpu_index, force_single_gpu=True)
    env.logger.log(f"Rank {0}: {torch.cuda.get_device_name(0)}", show=True)
    evaluate(env)

