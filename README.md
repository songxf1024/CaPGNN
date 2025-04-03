# Optimizing Parallel Graph Neural Network Training with Joint Caching and Resource-Aware Graph Partitioning

## Directory Hierarchy

```
|-- CaPGNN
|   `-- assigner
|   `-- communicator
|   `-- config            # offline configurations of experiments
|   `-- helper
|   `-- manager
|   `-- model             # customized PyTorch modules
|   `-- trainer
|   `-- util
|-- exp                   # experiment results
```

## Setup
#### Software Dependencies
- Ubuntu 20.04.6 LTS
- Python 3.9.15
- CUDA 12.1
- PyTorch 2.3.0
- DGL 2.3.0

#### Hardware Dependencies
- CPU: dual-core Intel® Xeon® Gold 6230
- RAM: 768GB 
- GPUs: 2 NVIDIA Tesla A40, 2 NVIDIA RTX 3090, 2 NVIDIA RTX 3060, 2 NVIDIA GTX 1660Ti.
- PCIe 3.0 x16

### Dataset
- CoraFull
- Flickr
- CoauthorPhysics
- Reddit
- Yelp
- AmazonProducts
- ogbn-products

## Usage

### Partition the Graph

Before conducting training, run `cspart.py` to partition the coressponding graph into several subgraphs:

```bash
python cspart.py --dataset_index=6 --part_num=4 --our_partition=1 --gpus_index=0
```

### Train the Model

Run `main.py`:

```bash
python main.py --dataset_index=6 --part_num=4 --gpus_index=0
```

### Experiment Customization

Adjust configurations in `CaPGNN/config/*yaml` to customize dataset, model, training hyperparameter, bit-width assignment settings or add new configurations.

## License
Copyright (c) 2025 xianfeng song. All rights reserved.

Licensed under the MIT License.
