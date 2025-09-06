# Optimizing Parallel Graph Neural Network Training with Joint Caching and Resource-Aware Graph Partitioning

This is the toy example of distributed version for CaPGNN.

## Usage

### Partition the Graph
Before conducting training, run `cspart.py` to partition the coressponding graph into several subgraphs:
```bash
 python cspart.py --our_partition=1 --server_num=2 --partition_num="2,2" --dataset_index=6 --gpus_index=0
```

### Train the Model
Run `main.py`:
```bash
## machine 1:
python main.py --dataset_index=6 --num_parts="2,2" --server_num=2 --server_id=0 --gpus_index=0
## machine 2:
python main.py --dataset_index=6 --num_parts="2,2" --server_num=2 --server_id=1 --gpus_index=0
```


## License
Copyright (c) 2025 xianfeng song. All rights reserved.

Licensed under the MIT License.
