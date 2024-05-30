#!/bin/sh

module load anaconda
module load cuda/12.1
source activate ldm
wandb offline

export PYTHONUNBUFFERED=1

echo $1 $2 $3
torchrun --nnodes=1 --nproc-per-node=$2 main.py --base ./configs/categorical-diffusion/ruijin_3d_cdm.yaml --gpus $3 --name $1 -t > ./runs/$1/out.txt 2>&1
