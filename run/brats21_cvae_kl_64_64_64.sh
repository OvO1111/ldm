#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ldm
wandb offline

export PYTHONUNBUFFERED=1

echo $1
export exp=$1
torchrun --nproc-per-node=4 main.py --base ./configs/autoencoder/brats2021_3d_ae_kl.yaml --gpus 0,1,2,3 --name $exp -t > ./runs/$exp/out.txt 2>&1
