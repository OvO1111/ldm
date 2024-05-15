#!/bin/sh

ngpu=4

i=0
igpus=""
for (( ; i<ngpu; i++ )); do
  if [ $i -ne 0 ]; then
    igpus+=","
  fi
  igpus+="$i"
done

# exp="brats21_cvae_kl_64_64_64"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:4 -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_cvae_kl_64_64_64.sh $exp $ngpu $igpus

exp="ruijin_vae_kl_64_128_128"
mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:4 -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ruijin_vae_kl_64_128_128.sh $exp $ngpu $igpus