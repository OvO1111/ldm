#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ldm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2 $3
export exp=$1
torchrun --nnodes=1 --nproc-per-node=$2 main.py --base ./configs/latent-diffusion/ruijin_3d_ldm.yaml --gpus $3 --name $exp -t > ./runs/$exp/out.txt 2>&1
