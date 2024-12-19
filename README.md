# CNN_attention
Simulation and analysis code for Lindsay and Miller 2018
## 1. Environment
# NYU Greene Environment Setup Guide

This guide provides instructions for setting up a Python environment with TensorFlow on NYU Greene using Singularity containers.

## Initial Setup

First, navigate to your scratch directory and copy the required Singularity container and overlay filesystem:

```bash
cd /scratch/$USER
cp /scratch/work/public/singularity/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif .
cp /scratch/work/public/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip -vvv overlay-25GB-500K.ext3.gz
```

## Conda Environment Setup

1. Navigate to the ext3 directory and download Miniconda:
```bash
cd /ext3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

2. Install Miniconda:
```bash
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda
```

3. Initialize Conda:
```bash
source /ext3/miniconda/bin/activate
conda init
```

4. Create and activate a Python 2.7 environment:
```bash
conda create -p /ext3/envs/vgg16_env python=2.7 -y
conda activate /ext3/envs/vgg16_env
```

## Package Versions

The following packages will be installed in the environment:

| Package | Version |
|---------|----------|
| tensorflow | 1.15.0 |
| numpy | 1.16.6 |
| scipy | 1.2.3 |
| matplotlib | 2.2.5 |
| scikit-learn | 0.20.4 |
| scikit-image | 0.14.2 |
| Pillow | 6.2.1 |
| h5py | 2.10.0 |

Additional dependencies and their versions:
- absl-py (0.15.0)
- tensorboard (1.15.0)
- tensorflow-estimator (1.15.1)
- Keras-Applications (1.0.8)
- Keras-Preprocessing (1.1.2)
- protobuf (3.17.3)
- grpcio (1.41.1)
- Markdown (3.1.1)
- Werkzeug (1.0.1)

## Note

This environment is specifically configured for running VGG16 and other deep learning models that require Python 2.7 and TensorFlow 1.15.0. Make sure to activate the environment (`conda activate /ext3/envs/vgg16_env`) before running your code.
## 2. Load the Data
## 3. How to Run
