#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ldm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2 $3
torchrun --nnodes=1 --nproc-per-node=$2 main.py --base ./configs/autoencoder/ruijin_3d_ae_kl.yaml --gpus $3 --name $1 -t > ./runs/$1/out.txt 2>&1
