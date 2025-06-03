import tools

if tools.outer_ip == '229':
    ## 229
    gpu_info = {
        # id:  memory,  spmm,    mm,    H2D,    D2H,    D2D
        2: (24 * 1024, 0.1069, 0.1389, 0.1198, 0.1207, 0.0014),  # RTX 3090
        3: (24 * 1024, 0.1060, 0.1372, 0.1188, 0.1198, 0.0014),  # RTX 3090
        4: (24 * 1024, 0.1072, 0.1393, 0.1224, 0.1242, 0.0014),  # RTX 3090
        7: (24 * 1024, 0.1058, 0.1385, 0.1194, 0.1208, 0.0014),  # RTX 3090
        5: (8  * 1024, 0.2934, 0.4942, 0.1188, 0.1186, 0.0033),  # RTX 2060
        6: (8  * 1024, 0.2975, 0.5002, 0.1195, 0.1204, 0.0033),  # RTX 2060
    }
    gpu_capability = {
     # id: memory, (spmm/max(spmm), mm/max(mm), H2D/max, D2H/max, D2D/max)
        2: [24*1024, None, None, None, None, None],  # RTX 3090
        3: [24*1024, None, None, None, None, None],  # RTX 3090
        4: [24*1024, None, None, None, None, None],  # RTX 3090
        7: [24*1024, None, None, None, None, None],  # RTX 3090
        5: [ 8*1024, None, None, None, None, None],  # RTX 2060
        6: [ 8*1024, None, None, None, None, None],  # RTX 2060
    }
elif tools.outer_ip == '228':
    ## 228
    gpu_info = {
        # id:  memory,  spmm,    mm,    H2D,    D2H,    D2D
        0: (24 * 1024, 0.1067, 0.1409, 0.1184, 0.1217, 0.0014),  # RTX 3090
        1: (12 * 1024, 0.1948, 0.3393, 0.1223, 0.1240, 0.0038),  # RTX 3060
        2: (24 * 1024, 0.1054, 0.1351, 0.1196, 0.1207, 0.0014),  # RTX 3090
        3: (12 * 1024, 0.1975, 0.3485, 0.1217, 0.1232, 0.0038),  # RTX 3060
        4: (48 * 1024, 0.1195, 0.1416, 0.1175, 0.1184, 0.0021),  # Tesla A40
        5: ( 6 * 1024, 0.3370, 0.9791, 0.1220, 0.1232, 0.0057),  # GTX 1660 Ti
        6: (48 * 1024, 0.1201, 0.1426, 0.1198, 0.1193, 0.0021),  # Tesla A40
        7: ( 6 * 1024, 0.3447, 1.0084, 0.1255, 0.1256, 0.0057),  # GTX 1660Ti
    }
    gpu_capability = {
     # id: memory(MB), (spmm/max(spmm), mm/max(mm), H2D/max, D2H/max, D2D/max)
        0: [24*1024, None, None, None, None, None],  # RTX 3090
        1: [12*1024, None, None, None, None, None],  # RTX 3060
        2: [24*1024, None, None, None, None, None],  # RTX 3090
        3: [12*1024, None, None, None, None, None],  # RTX 3060
        4: [48*1024, None, None, None, None, None],  # Tesla A40
        5: [ 6*1024, None, None, None, None, None],  # GTX 1660 Ti
        6: [48*1024, None, None, None, None, None],  # Tesla A40
        7: [ 6*1024, None, None, None, None, None],  # GTX 1660Ti
    }

def cal_gpus_capability(all_gpus_list):
    global gpu_capability
    spmm_list = []
    mm_list = []
    h2d_list = []
    d2h_list = []
    d2d_list = []
    for gpu_id in all_gpus_list:
        spmm_list.append(gpu_info[gpu_id][1])
        mm_list.append(gpu_info[gpu_id][2])
        h2d_list.append(gpu_info[gpu_id][3])
        d2h_list.append(gpu_info[gpu_id][4])
        d2d_list.append(gpu_info[gpu_id][5])
    for gpu_id in all_gpus_list:
        gpu_capability[gpu_id][1] = gpu_info[gpu_id][1] / max(spmm_list)
        gpu_capability[gpu_id][2] = gpu_info[gpu_id][2] / max(mm_list)
        gpu_capability[gpu_id][3] = gpu_info[gpu_id][3] / max(h2d_list)
        gpu_capability[gpu_id][4] = gpu_info[gpu_id][4] / max(d2h_list)
        gpu_capability[gpu_id][5] = gpu_info[gpu_id][5] / max(d2d_list)
    return gpu_capability


# GPU computing power. Embodied by spmm and mm
def get_gpu_capability(gpus_list, alpha=0.5, reverse=True):
    # Calculate the spmm/max(spmm) + mm/max(mm) and store it in the dict
    indicators = {gpu_id: gpu_capability[gpu_id][1]*alpha+gpu_capability[gpu_id][1]*(1-alpha) for gpu_id in gpus_list}
    # Sort by indices value descending order and get the corresponding GPU device key. 
    # The larger the value, the greater the calculation cost and the weaker the GPU.
    sorted_gpu_ids = sorted(gpus_list, key=lambda x: indicators[x], reverse=reverse)
    return sorted_gpu_ids

def get_gpu_memory(num_nodes, num_edges, feats, beta=0):
    return 1*(num_nodes * 4 + num_edges * 4 * 2 + num_nodes * feats.shape[1] * 4 + beta * 4) / 1024 / 1024

def gpu_memory_enough(gpu_id, num_nodes, num_edges, feats, beta=0):
    return gpu_capability[gpu_id][0] > get_gpu_memory(num_nodes, num_edges, feats, 0)


