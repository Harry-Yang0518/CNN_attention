# Comparing Saliency Maps and Attention Masks in Neural Networks for Enhanced Interpretability and Performance

# 1. NYU Greene Environment Setup Guide

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

## Package Installation

After activating the environment, install all required packages using pip:

```bash
pip install tensorflow==1.15.0 numpy==1.16.6 scipy==1.2.3 matplotlib==2.2.5 scikit-learn==0.20.4 scikit-image==0.14.2 Pillow==6.2.1 h5py==2.10.0 absl-py==0.15.0 astor==0.8.1 backports-abc==0.5 backports.functools-lru-cache==1.6.6 backports.weakref==1.0.post1 certifi==2020.6.20 cloudpickle==1.2.2 cycler==0.10.0 cytoolz==0.10.1 dask==1.2.2 decorator==5.1.0 enum34==1.1.10 funcsigs==1.0.2 functools32==3.2.3.post2 futures==3.4.0 gast==0.2.2 google-pasta==0.2.0 grpcio==1.41.1 imageio==2.6.1 Keras-Applications==1.0.8 Keras-Preprocessing==1.1.2 kiwisolver==1.1.0 Markdown==3.1.1 mkl-fft==1.0.15 mkl-random==1.1.0 mkl-service==2.3.0 mock==3.0.5 networkx==2.2 olefile==0.46 opt-einsum==2.3.2 protobuf==3.17.3 pyparsing==2.4.7 python-dateutil==2.9.0.post0 pytz==2024.2 PyWavelets==1.0.3 singledispatch==3.7.0 six==1.16.0 subprocess32==3.5.4 tensorboard==1.15.0 tensorflow-estimator==1.15.1 termcolor==1.1.0 toolz==0.10.0 tornado==5.1.1 Werkzeug==1.0.1 wheel==0.37.1 wrapt==1.15.0
```

This command will install all packages with their specific versions to ensure compatibility.

# 2. Dataset Structure

Download the dataset here  `https://datadryad.org/stash/dataset/doi:10.5061/dryad.jc14081#usage`

Base path: `/scratch/$USER/CNN_attention/Data/VGG16`

## Main Directories
```
VGG16/
├── images/                     # Image data directory
├── catbins/                    # Category bins data
├── ori_catbins/               # Orientation category bins
├── object_GradsTCs/           # Object gradients TCs
├── objperf/                   # Object performance metrics
├── oriperf/                   # Orientation performance metrics
├── ori_TCGrads/              # Orientation TC gradients
```

## Model Weights
```
VGG16/
└── vgg16_weights.npz         # VGG16 model weights
```

## Stimulus and Object Files
```
VGG16/
├── Stim2Constr540.npz                             # Stimulus construction data
├── ObjectAttn_aTCs_0c0l_a1bd1_im1Nact.npz        # Object attention negative activation
├── ObjectAttn_aTCs_0c0l_a1bd1_im1Pact.npz        # Object attention positive activation
├── ObjectAttn_aTCs_0c0l_a1bd1_im1perf.npz        # Object attention performance
```

## Orientation Files
```
VGG16/
├── OriAttn_aTCs_40o12lNact.npz                   # Orientation attention negative activation
├── OriAttn_aTCs_40o12lPact.npz                   # Orientation attention positive activation
├── OriAttn_aTCs_40o12lperf.npz                   # Orientation attention performance
```

Note: The `attention_results_*` files are excluded from this structure as they are not part of the dataset.

## Required Files for Project

1. Dataset Files:
   - `images/`
   - `catbins/`
   - `ori_catbins/`
   - `object_GradsTCs/`
   - `objperf/`
   - `oriperf/`
   - `ori_TCGrads/`

2. Model Files:
   - `vgg16_weights.npz`

3. Stimulus and Analysis Files:
   - `Stim2Constr540.npz`
   - Object attention files (ObjectAttn_*.npz)
   - Orientation attention files (OriAttn_*.npz)

# 3. How to Run

Text the command in terminal:
```
sbatch /scratch/$USER/CNN_attention/main/scripts/main.sh
```

# 4. Results
The numeric output are logged under the `/logs` directory, and you can check the visual output here: https://drive.google.com/drive/folders/1LUptdZMMl4rTT63bABkDo3AArDtgU4pr?usp=sharing
