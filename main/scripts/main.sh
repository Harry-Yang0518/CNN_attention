#!/bin/bash

#SBATCH --job-name=DL_SYS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# Set fixed experiment parameters
IMTYPE=1
BATCH_SIZE=1
MAX_IMAGES=2
ATTENTION_TYPE="TCs"

# Define categories to iterate
CATEGORIES=(1 5 9 13 17)

# Create experiment name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_BASE="attention_analysis_${TIMESTAMP}"
LOG_BASE_DIR="/scratch/$USER/CNN_attention/logs/${EXPERIMENT_BASE}"
mkdir -p ${LOG_BASE_DIR}

# Save base experiment parameters
echo "Base Experiment Parameters:" > ${LOG_BASE_DIR}/experiment_params.txt
echo "Image Type: ${IMTYPE}" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Batch Size: ${BATCH_SIZE}" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Max Images: ${MAX_IMAGES}" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Attention Type: ${ATTENTION_TYPE}" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Categories: ${CATEGORIES[*]}" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Layers: 0-12" >> ${LOG_BASE_DIR}/experiment_params.txt
echo "Timestamp: ${TIMESTAMP}" >> ${LOG_BASE_DIR}/experiment_params.txt

# Set paths
ext3_path=/scratch/$USER/CNN_attention/environment/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/CNN_attention/environment/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# Iterate through categories
for CATEGORY in "${CATEGORIES[@]}"
do
    echo "Starting experiments for category ${CATEGORY}"
    
    # Create category-specific directory
    CAT_DIR="${LOG_BASE_DIR}/category${CATEGORY}"
    mkdir -p ${CAT_DIR}
    
    # Iterate through layers 0-12
    for LAYER in {0..12}
    do
        echo "Running experiment for category ${CATEGORY}, layer ${LAYER}"
        
        # Create layer-specific log directory
        LAYER_LOG_DIR="${CAT_DIR}/layer${LAYER}"
        mkdir -p ${LAYER_LOG_DIR}
        
        singularity exec --nv --overlay ${ext3_path}:ro ${sif_path} /bin/bash -c "
        source ~/.bashrc
        conda activate /ext3/envs/vgg16_env
        
        echo '==================================================='
        echo 'Starting experiment:'
        echo 'Category: ${CATEGORY}'
        echo 'Layer: ${LAYER}'
        echo 'Logs will be saved to: ${LAYER_LOG_DIR}'
        echo '==================================================='
        
        python /scratch/$USER/CNN_attention/main/main.py \
            --category ${CATEGORY} \
            --imtype ${IMTYPE} \
            --batch_size ${BATCH_SIZE} \
            --max_images ${MAX_IMAGES} \
            --attention_type ${ATTENTION_TYPE} \
            --layer ${LAYER} \
            2>&1 | tee ${LAYER_LOG_DIR}/experiment.log
        "
        
        # Copy layer-specific slurm output
        cp slurm-${SLURM_JOB_ID}.out ${LAYER_LOG_DIR}/slurm.out
        
        echo "Completed experiment for category ${CATEGORY}, layer ${LAYER}"
        echo "----------------------------------------"
    done
    
    echo "Completed all layer experiments for category ${CATEGORY}"
    echo "==========================================="
done

echo "All experiments completed. Results saved in ${LOG_BASE_DIR}"