#!/bin/sh

export ngpus=$1
# export exp=$2
# if [ -f ./dataset.tar ]; then
#   date
#   tar -xf datasets.tar -C /dev/shm
#   date
# fi

# export exp="ensemble_cdm_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out_$2.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*6)) --gpus=$ngpu --mem=$(($ngpu*64))G ./run/run_template.sh ${@:2}
# export exp="ruijin_cdm_baseline_64_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out_$2.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*4)) --gpus=$ngpu --mem=$(($ngpu*48))G ./run/run_template.sh ${@:2}
# export exp="brats21_ldm_cvq_128_128_128"; export ngpus=$ngpu; mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out_$2.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpu*4)) --gpus=$ngpu --mem=$(($ngpu*48))G ./run/run_template.sh ${@:2}
# export exp="brats21fg_cvq_64_64_64"
# export exp="ruijin_2d_vq_ldm_(128)_512_512"
export exp="ensemblev2_classifier_128_128_128"
# export exp="brats21_ldm_cvq_128_128_128"
# export exp="ensemble_multiwinnorm_ldm_vq_128_128_128"
# export exp='ensemble_ddpm_128_128_128'
# export exp="brats21_subclass/866f/1100"
# export exp="test"

mkdir -p ./runs/$exp; sbatch -D $(pwd) -J $exp -o ./runs/$exp/slurm_out_$2.txt -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpus*4)) --gpus=$ngpus --mem=$(($ngpus*64))G ./run/run_template.sh ${@:2}

# export organ=$2
# sbatch -D $(pwd) -J seg_msd_$organ -p smart_health_02 -N 1 -n 1 --cpus-per-task=$(($ngpus*8)) --gpus=$ngpus --mem=$(($ngpus*128))G ./run/run_seg.sh ${@:2}