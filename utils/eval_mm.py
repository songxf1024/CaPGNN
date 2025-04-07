import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from env import DistEnv
from util import *
import time

def sparse_tensor_memory_usage(m):
    bytes_per_index = 8
    bytes_per_value = m.element_size()
    total_bytes = (m.ndim * bytes_per_index + bytes_per_value) * m._nnz()
    memory_usage_mb = total_bytes / (1024 ** 2)
    return memory_usage_mb


def batch_matmul(env, sz_tag, size, repeat, sparsity=0.5, sparse=False):
    env.logger.log(f'Size: {sz_tag: <5} Repeat: {repeat}',  rank=0)
    do_mm = torch.sparse.mm if sparse else torch.mm
    for i in range(repeat):
        # with env.timer.timing_cuda('load'):
        A = create_adj_sparse_matrix(size, sparsity=sparsity, device=env.device) if sparse else create_dense_matrix(size, device=env.device)
        B = create_dense_matrix(size, device=env.device)
        torch.cuda.synchronize()
        with env.timer.timing_cuda('matmul-'+sz_tag):
            do_mm(A, B)
        torch.cuda.synchronize()
        del A, B
        torch.cuda.empty_cache()
        env.logger.log(f'{sz_tag} {i+1}/{repeat}', oneline=True, rank=0, show=True)
    env.logger.log('', rank=0, show=True)


def eval_matmul(env, sparsity=0.5, sparse=False):
    sizes = [
         # (1000, '1K', (16, 16)),
         # (1000, '10K', (51, 51)),
         # (1000, '100K', (160, 160)),
         # (1000, '1M', (512, 512)),
         # (1000, '10M', (1620, 1620)),
         # (100, '100M', (5119, 5119)),
         (50, '512M', (11585, 11585)),
         # (10, '1G', (16384, 16384)),
         ]
    for repeat, tag, size in sizes:
        batch_matmul(env, tag, size, repeat, sparsity=sparsity, sparse=sparse)
        torch.cuda.empty_cache()


def evaluate(env, sparsity=0.5, sparse=False):
    print(">> spmm" if sparse else ">> mm")
    warm_up_gpus(env.device)
    torch.cuda.synchronize()
    with env.timer.timing('total'):
        eval_matmul(env, sparsity=sparsity, sparse=sparse)
    torch.cuda.synchronize()
    rank_summary, rank_summary_str = env.timer.summary()
    env.logger.log(rank_summary_str, prefix=('\n' + '*' * 10 + '\n'), suffix='*' * 10, show=True)
    env.logger.save_summary_to_csv(rank_summary, f'rank-{"spmm" if sparse else "mm"}.csv')
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # set_cpu_affinity(rank=0, start_core_index=40, num_cores_per_gpu=20)
    set_random_seeds()

    gpu_index = 1
    sparsity = 0.996
    for sparse in [False, True]:
        print('>> gpu group index:', gpu_index)
        # 228: 3090  3060  3090  3060  A40   1660  A40   1660
        # 229: 1650  1650  3090  3090  3090  2060  2060  3090
        gpus = ['0', '1',   '2',  '3',  '4',  '5',  '6',  '7'][gpu_index]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        check_gpu_temperatures(gpus, temp_threshold=60)
        env = DistEnv(0, 1, 'nccl', log_prefix='eval_mm', gpus_note=gpu_index, force_single_gpu=True)
        env.logger.log(f"Rank {0}: {torch.cuda.get_device_name(0)}", show=True)
        evaluate(env, sparsity=sparsity, sparse=sparse)
        del env
        torch.cuda.empty_cache()
        print('\n', '#' * 50)
        time.sleep(2)

