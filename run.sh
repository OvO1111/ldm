#!/bin/sh

ngpu=$1
# if [ -f ./dataset.tar ]; then
#   date
#   tar -xf datasets.tar -C /dev/shm
#   date
# fi

i=0
igpus=""
for (( ; i<ngpu; i++ )); do
  if [ $i -ne 0 ]; then
    igpus+=","
  fi
  igpus+="$i"
done
igpus+=","

# exp="brats21_cvae_kl_64_64_64"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_cvae_kl.sh $exp $ngpu $igpus
# exp="ruijin_mask_vae_kl_64_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ruijin_vae_kl.sh $exp $ngpu $igpus
# exp="brats21_cvq_128_128_128"; mkdir -p ./runs/$exp; sbatch -J $exp -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/run_template.sh $exp $ngpu $igpus train configs/autoencoder/brats2021_3d_cvq.yaml
# exp="ruijin_mask_ldm_vae_kl_64_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/run_template.sh $exp $ngpu $igpus train ./configs/latent-diffusion/ruijin_3d_ldm_mask.yaml
# exp="brats21_ldm_cvae_kl_128_128_128"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_ldm_cvae_kl.sh $exp $ngpu $igpus
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n 8 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_ldm_cvae_kl_test.sh $exp $ngpu $igpus
# exp="ruijin_ldm_vae_kl_64_128_128"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt -J $exp ./run/ruijin_ldm_vae_kl_64_128_128.sh $exp $ngpu $igpus
# exp="ruijin_cldm_vae_kl_64_128_128"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt -J $exp ./run/ruijin_cldm_vae_kl.sh $exp $ngpu $igpus
# exp="brats21_transfer_t1000_ldm_cvae_kl_64_64_64"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt -J $exp ./run/brats21_transfer_ldm_cvae_kl_64_64_64.sh $exp $ngpu $igpus
# exp="ruijin_cdm_96_192_192_x0_v4"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -N 1 -n 24 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt -J $exp ./run/ruijin_cdm.sh $exp $ngpu $igpus
# exp="brats21_ddpm_128_128_128"
# mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n 8 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_ddpm.sh $exp $ngpu $igpus
# exp="brats21_ddpm_128_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n 8 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/brats21_ddpm_test.sh $exp $ngpu $igpus
# exp="brats21_ddpm_128_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n 8 --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/run_template.sh $exp $ngpu $igpus test ./configs/latent-diffusion/brats2021_3d_ddpm_test.yaml
# exp="ensemble_cdm_64_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n $((6*$ngpu)) --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/ensemble_cdm.sh $exp $ngpu $igpus
exp="ensemble_vq_128_128_128"; mkdir -p ./runs/$exp; sbatch -D /ailab/user/dailinrui/code/latentdiffusion -J $exp -N 1 -n $((6*$ngpu)) --gres=gpu:$ngpu -p smart_health_02 -A smart_health_02 -o ./runs/$exp/slurm_out.txt ./run/run_template.sh $exp $ngpu $igpus train ./configs/autoencoder/ensemble_vq.yaml