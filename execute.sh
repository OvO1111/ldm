#!/bin/sh

ngpu=$1
# if [ -f ./dataset.tar ]; then
#   date
#   tar -xf datasets.tar -C /dev/shm
#   date
# fi

# export exp="ensemble_cdm_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*6)) --gpus=$ngpu --mem=$(($ngpu*64))G ./run/run_template.sh ${@:2}
# export exp="ensemble_ldm_vq_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*4)) --gpus=$ngpu --mem=$(($ngpu*48))G ./run/run_template.sh ${@:2}
# export exp="ruijin_cdm_baseline_64_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*4)) --gpus=$ngpu --mem=$(($ngpu*48))G ./run/run_template.sh ${@:2}
# export exp="brats21_ldm_cvq_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*6)) --gpus=$ngpu --mem=$(($ngpu*64))G ./run/run_template.sh ${@:2}
export exp="brats21_ldm_cvq_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*6)) --gpus=$ngpu --mem=$(($ngpu*64))G ./run/run_template.sh ${@:2}