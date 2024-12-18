#!/bin/bash

#SBATCH --job-name=DL_SYS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000


# Singularity path
ext3_path=/scratch/$USER/CNN_attention/environment/overlay-25GB-500K.ext3
sif_path=/scratch/$USER//CNN_attention/environment/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv --overlay ${ext3_path}:ro ${sif_path} /bin/bash -c "
conda activate /ext3/envs/vgg16_env


python /scratch/hy2611/CNN_attention/main/simulation/vgg16_OrispatAttn.py
"

