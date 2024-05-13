#!/bin/sh

exp="brats21_cvae_kl_64_64_64"
mkdir -p ./runs/$exp
sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:4 -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_cvae_kl_64_64_64.sh $exp