#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ldm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2 $3
export exp=$1
MASTER_PORT=25678 torchrun --nnodes=1 --nproc-per-node=$2 --master-port 25678 main.py --base ./configs/latent-diffusion/brats2021_3d_ldm_ft.yaml --gpus $3 --name $exp -t > ./runs/$exp/out.txt 2>&1
