# Optimizing Parallel Graph Neural Network Training with Joint Caching and Resource-Aware Graph Partitioning

<div align="center"><img src="https://github.com/songxf1024/CaPGNN/blob/master/images/cover.png?raw=true"/></div>   

## Directory Hierarchy
```
|-- CaPGNN
|   `-- communicator
|   `-- config            # offline configurations of experiments
|   `-- helper
|   `-- manager
|   `-- model             # customized PyTorch modules
|   `-- trainer
|   `-- util
|-- exp                   # experiment results
|-- utils
```
> **News:**  
> - CaPGNN is easily extensible to **distributed systems**, and we have released a demo of the distributed version: [branch/dist](https://anonymous.4open.science/r/CaPGNN-1024Aa-dist/).
> - The architecture of the distributed version is:  
> <div align="center"><img src="https://github.com/songxf1024/CaPGNN/blob/master/images/demo%20of%20distributed%20version.png?raw=true" width="600px" /></div>   

## Setup
#### Software Dependencies
- Ubuntu 20.04.6 LTS
- Python 3.9.15
- CUDA 12.1
- PyTorch 2.3.0
- DGL 2.3.0

#### Hardware Dependencies
> The following configurations are recommended and not mandatory.

- CPU: dual-core Intel® Xeon® Gold 6230
- RAM: 768GB 
- GPUs: 2x NVIDIA Tesla A40, 2x NVIDIA RTX 3090, 2x NVIDIA RTX 3060, 2x NVIDIA GTX 1660Ti.
- PCIe 3.0 x16

### Installation
Running the following commands will create the virtual environment and install the dependencies. `conda` can be downloaded from [anaconda](https://www.anaconda.com/download/success). 
```bash
conda clean --all -y
pip cache purge
conda create -n capgnn python=3.9.12
conda activate capgnn

pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html \
  -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
  
```

> Note that the version of DGL must be compatible with the corresponding [PyTorch version](https://www.dgl.ai/pages/start.html) and [Python version](https://www.dgl.ai/dgl_docs/install/index.html).  
> Additionally, if you have a GPU, the PyTorch version must also match your [CUDA version](https://pytorch.org/get-started/previous-versions/).


### Datasets
> Datasets will be downloaded automatically if they are missing.

| Dataset             | \|V\|    | \|E\|      | f\_dim | Classes |
| ------------------- | :------: | :--------: | :----: | :-----: |
| Yelp                | 716847   | 13954819   | 300    | 100     |
| Reddit              | 232965   | 114615892  | 602    | 41      |
| Flickr              | 89250    | 899756     | 500    | 7       |
| CoraFull            | 19793    | 126842     | 8710   | 7       |
| AmazonProducts      | 1569960  | 264339468  | 200    | 107     |
| CoauthorPhysics     | 34493    | 495924     | 8415   | 5       |
| ogbn-products       | 2449029  | 61859140   | 100    | 47      |



## Usage

### GPU performance
GPU communication capabilities (H2D, D2H, IDT):
```bash
python utils/eval_bw.py
```
> The output in the console will be like:
> ```bash
> 07:29:28.949017 [0] Rank 0: NVIDIA GeForce RTX 3090
> 07:30:05.132368 [0] Size: 512M  Repeat: 50
> 07:31:10.345209 [0] HtoD 512M 50/50
> 07:31:10.346265 [0] Size: 512M  Repeat: 50
> 07:31:17.811319 [0] DtoH 512M 50/50
> 07:31:17.859560 [0] Size: 512M  Repeat: 50
> 07:31:19.343551 [0] IDT 512M 50/50
> Timer Summary:
> Key             Total        Ave        Std      Count
> --------------------------------------------------
> HtoD-512M      3.8012     0.0760     0.0194         50
> DtoH-512M      5.3154     0.1063     0.0050         50
> IDT-512M       0.0686     0.0014     0.0000         50
> total         74.2611    74.2611     0.0000          1
> ```

GPU computation capabilities (SpMM and MM):
```bash
python utils/eval_mm.py
```
> The output in the console will be like:
> ```bash
> 07:35:59.305771 [0] Rank 0: NVIDIA GeForce RTX 3060
> >> spmm
> 07:35:59.307674 [0] Size: 512M  Repeat: 50
> 07:36:12.598796 [0] 512M 50/50
> **********
> 07:36:12.599498 [0] 
> Timer Summary:
> Key             Total        Ave        Std      Count
> --------------------------------------------------
> matmul-512M     9.7713     0.1954     0.0078         50
> total          13.2912    13.2912     0.0000          1
> ```

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
The results are in the `exp` directory.


### Experiment Customization
Adjust configurations in `CaPGNN/config/*yaml` to customize dataset, model, training hyperparameter or add new configurations.


## License
Copyright (c) 2025 . All rights reserved.

Licensed under the MIT License.
