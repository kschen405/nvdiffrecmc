#!/bin/bash
export PATH="/home/kc81/setup/anaconda3/bin:$PATH"
export PATH="/software/gcc-8.2.0/bin:$PATH" 
# export PATH="/software/gcc-9.2.0/bin:$PATH" 
export TCNN_CUDA_ARCHITECTURES=75 # RTX 6000: 75, A40: 86
source /etc/profile
module load cuda-toolkit/11.6
module load gcc/8.2.0
echo $CUDA_HOME
echo $PATH
echo $LD_LIBRARY_PATH
nvidia-smi
source ~/.bashrc
conda activate dmodel3
gcc --version
# nvcc --version
python -c "import torch; print('torch version = ', torch.__version__); print('torch.cuda.is_available() = ', torch.cuda.is_available() ) ; print('torch.cuda.device_count() = ', torch.cuda.device_count())"
python -c "import torchvision; print('torchvision.__version__ = ', torchvision.__version__);"
python -c "import scipy;"
# pip uninstall tinycudann -y
# pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
# python data/download_datasets.py
python train.py --config configs/nerf_hotdog01.json
# python test.py
# python train.py --config configs/waymo.json
# python train.py --outdir=finetune-pretrain --resume "diffusion-stylegan2-lsun-bedroom.pkl" --data="/home/kc81/datasets/lsun_bedroom200k.zip" --gpus=2 --cfg paper256 --kimg 50712 --target 0.6 --ts_dist priority
# python converter.py ../../datasets/wod ../../datasets/wod_kitti_test --prefix 0 --num_proc 8
# python converter.py /home/kc81/datasets/wod /home/kc81/datasets/wod_kitti --prefix 0 --num_proc 8



