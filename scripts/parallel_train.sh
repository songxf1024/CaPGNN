#!/usr/bin/env bash

# ===== 主机 & 密码 =====
A_HOST="sxf@202.38.247.228"
B_HOST="sxf@202.38.247.229"
PWA="passwd_for_A"
PWB="passwd_for_B"

# ===== 远端路径 & Python 可执行文件 =====
DIR_A="/mnt/disk/storage/sxf/gnn/myGNN/CaPGNN"
DIR_B="/mnt/disk/sxf/gnn/CaPGNN"
PY_A="/mnt/disk/software/anaconda3/envs/dglv2/bin/python"
PY_B="/opt/software/anaconda3/envs/dglv2/bin/python"

# ===== 训练参数（都可改）=====
DATASET_INDEX=6
NUM_PARTS="2,2"
SERVER_NUM=2
SERVER_ID_A=0
SERVER_ID_B=1
GPUS_INDEX=0

# ===== 本地日志文件 =====
LA="serverA.log"
LB="serverB.log"

# ===== 并行执行 =====
sshpass -p"$PWA" ssh -o StrictHostKeyChecking=accept-new "$A_HOST" \
"bash -lc 'cd \"$DIR_A\" && \"$PY_A\" main.py \
  --dataset_index=$DATASET_INDEX \
  --num_parts=\"$NUM_PARTS\" \
  --server_num=$SERVER_NUM \
  --server_id=$SERVER_ID_A \
  --gpus_index=$GPUS_INDEX'" >"$LA" 2>&1 & PA=$!

sshpass -p"$PWB" ssh -o StrictHostKeyChecking=accept-new "$B_HOST" \
"bash -lc 'cd \"$DIR_B\" && \"$PY_B\" main.py \
  --dataset_index=$DATASET_INDEX \
  --num_parts=\"$NUM_PARTS\" \
  --server_num=$SERVER_NUM \
  --server_id=$SERVER_ID_B \
  --gpus_index=$GPUS_INDEX'" >"$LB" 2>&1 & PB=$!

wait "$PA" "$PB"
echo "done. logs: $LA  $LB"
unset PWA PWB
