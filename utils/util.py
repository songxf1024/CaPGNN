import os
import random
import subprocess
import time
import matplotlib.pyplot as plt
import dgl
import numpy as np
import psutil
import torch
import dgl.sparse as dglsp
import scipy.sparse as sp


def plot_sparse_matrix_structure(sparse, title='Sparse Matrix Structure Visualization'):
    if isinstance(sparse, dgl.DGLGraph):
        src, dst = sparse.edges()
        num_nodes = sparse.number_of_nodes()
        sparse_matrix = np.zeros((num_nodes, num_nodes))
        sparse_matrix[src.numpy(), dst.numpy()] = 1
    else:
        sparse_matrix = sparse.to_dense().cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.spy(sparse_matrix, marker='.', markersize=1)
    plt.title(title)
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def sparse_tensor_to_dglgraph(sparse_tensor, directed=False):
    coalesced_tensor = sparse_tensor.coalesce()
    graph = dgl.DGLGraph()
    graph.add_nodes(sparse_tensor.size(0))
    indices = coalesced_tensor.indices()
    vals = coalesced_tensor.values()
    if directed:
        # If directed, add edges with specified weights
        src = indices[0].cpu()
        dst = indices[1].cpu()
        weights = vals.cpu()
        graph.add_edges(src, dst, {'weight': weights})
    else:
        # If undirected, add edges bidirectionally with specified weights
        rows, cols = indices
        mask = rows <= cols  # Mask for upper triangle
        upper_tri_indices = indices[:, mask]
        upper_tri_vals = vals[mask]
        src_dst = upper_tri_indices.cpu()
        weights = upper_tri_vals.cpu()
        graph.add_edges(src_dst[0], src_dst[1], {'weight': weights})
    return graph

def create_dense_matrix(size, device='cpu'):
    return torch.randn(size, dtype=torch.float32, device=device)

def create_adj_sparse_matrix_old(size, sparsity=0.5, device='cuda'):
    rows, cols = size
    total_elements = rows * cols
    num_non_zeros = int(total_elements * (1 - sparsity) / 2)  # Divide by 2 for symmetric matrix
    # Generate a random sparse matrix
    sparse_matrix = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    # Randomly set num_non_zeros elements to 1 in the upper triangle (excluding diagonal)
    upper_indices = torch.triu_indices(rows, cols, offset=1, device=device)
    indices = torch.randperm(upper_indices.size(1), device=device)[:num_non_zeros]
    selected_indices = (upper_indices[0][indices], upper_indices[1][indices])
    sparse_matrix[selected_indices] = 1.0
    # Mirror the upper triangle to the lower triangle
    sparse_matrix = sparse_matrix + sparse_matrix.t()
    # Ensure no diagonal entries (self-loops)
    # sparse_matrix.fill_diagonal_(0)
    # Convert sparse matrix to COO format sparse tensor
    coo_matrix = sparse_matrix.to_sparse_coo().coalesce()
    # Calculate and print the actual sparsity
    # num_nonzero = coo_matrix._nnz()
    # actual_sparsity = 1.0 - (num_nonzero / total_elements)
    # print(f"Expected sparsity: {sparsity}")
    # print(f"Actual sparsity: {actual_sparsity}")
    return coo_matrix


def create_adj_sparse_matrix(size, sparsity=0.5, device='cuda'):
    rows, cols = size
    total_elements = rows * cols
    num_non_zeros = int(total_elements * (1 - sparsity) / 2)  # Divide by 2 for symmetric matrix
    
    # Generate a random sparse matrix
    sparse_matrix = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    
    # Randomly set num_non_zeros elements to 1 in the upper triangle (excluding diagonal)
    upper_indices = torch.triu_indices(rows, cols, offset=1, device=device)
    
    # 选择非零元素的随机索引，避免内存问题
    indices = torch.randint(0, upper_indices.size(1), (num_non_zeros,), device=device)
    selected_indices = (upper_indices[0][indices], upper_indices[1][indices])
    
    # 将选择的索引位置的元素设置为 1
    sparse_matrix[selected_indices] = 1.0
    
    # Mirror the upper triangle to the lower triangle
    sparse_matrix = sparse_matrix + sparse_matrix.t()
    
    # Ensure no diagonal entries (self-loops)
    # sparse_matrix.fill_diagonal_(0)  # 你注释掉的这行可以选择保留或取消注释
    
    # Convert sparse matrix to COO format sparse tensor
    coo_matrix = sparse_matrix.to_sparse_coo().coalesce()
    
    # Calculate and print the actual sparsity
    # num_nonzero = coo_matrix._nnz()
    # actual_sparsity = 1.0 - (num_nonzero / total_elements)
    # print(f"Expected sparsity: {sparsity}")
    # print(f"Actual sparsity: {actual_sparsity}")
    
    # Return the COO sparse matrix
    return coo_matrix

    

def calculate_sparsity(graph, directed=False):
    """
    稀疏度的指标，表示图中未连接的节点对的比例.
    稀疏度越高，图中的连接关系越少，反之则越密集.
    """
    if isinstance(graph, torch.Tensor):
        graph = sparse_tensor_to_dglgraph(graph, directed)
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    possible_edges = num_nodes * (num_nodes - 1) / (1 if directed else 2)
    if possible_edges <= 0: return 1
    density = num_edges / possible_edges
    sparsity = 1 - density
    return sparsity

def create_sparse_matrix2(size, sparsity=0.5, device='cuda'):
    dense_matrix = torch.randn(size, dtype=torch.float32, device=device)
    mask = (torch.rand(size, device=device) > sparsity).float()
    sparse_matrix = dense_matrix * mask
    coo_matrix = sparse_matrix.to_sparse_coo()
    return coo_matrix

def calculate_sparsity2(matrix):
    total_elements = matrix.numel()
    # 以 COO 格式获取非零元素的个数
    non_zero_elements = matrix._values().numel()
    sparsity = 1.0 - (non_zero_elements / total_elements)
    return sparsity


def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def warm_up_gpus(device):
    x = torch.randn(100, 100).to(device)
    for _ in range(10): torch.matmul(x, x)
    torch.cuda.synchronize()
    print('warm up gpu done')

def warm_up_pcie(env):
    env.broadcast(tensor=torch.ones(100).to(env.device), src=0)
    if env.world_size > 1: env.barrier_all()
    torch.cuda.synchronize()
    print('warm up pcie done')



def check_gpu_temperatures(gpu_ids, temp_threshold=40, timeout=None):
    gpu_ids_list = gpu_ids.split(',')
    start_time = time.time()
    while True:
        temperatures = []
        all_below_threshold = True
        for gpu_id in gpu_ids_list:
            result = subprocess.run(['nvidia-smi', '-i', gpu_id, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
            temp = int(result.stdout.decode('utf-8').strip())
            temperatures.append(f'GPU {gpu_id}: {temp}°')
            if temp > temp_threshold: all_below_threshold = False
        if all_below_threshold:
            print('>> 当前GPU温度: ' + ' | '.join(temperatures))
            break
        print(f'>> 为防止GPU高温导致性能限制，等待降温中({temp_threshold}°): ' + ' | '.join(temperatures), end='\r')
        if timeout and (time.time() - start_time) > timeout:
            print('\n>>已达超时，不在等待 GPU 温度下降。')
            break
        time.sleep(1)
    print()


def set_cpu_affinity(rank, num_cores_per_gpu=4, start_core_index=0):
    num_cores = psutil.cpu_count(logical=True)
    core_ids = list(range(num_cores))
    # 计算起始和结束核的索引
    start_core = start_core_index + rank * num_cores_per_gpu
    end_core = start_core + num_cores_per_gpu
    # 获取要绑定的CPU核列表
    cpu_affinity = core_ids[start_core:end_core]
    # 设置当前进程的CPU核绑定
    p = psutil.Process(os.getpid())
    p.cpu_affinity(cpu_affinity)
    print(f">> GPU {rank} is bound to CPU cores {cpu_affinity}")


def set_high_priority(priority=-10):
    if os.name == 'nt':  # Windows
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print(f">> Set process priority to high on Windows")
    else:  # Unix-like systems
        pid = os.getpid()
        os.nice(priority)  # -20 是最高优先级，19 是最低优先级
        print(f">> Set process priority to high on Linux for PID {pid}")


def high_precision_sleep(seconds):
    end_time = time.perf_counter() + seconds
    while time.perf_counter() < end_time:
        pass




