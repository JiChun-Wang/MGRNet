# MGRNet: Multiple geometry representations for 6D object pose estimation in occluded or truncated scenes
This repository contains authors' implementation of MGRNet: 6D Object Pose Estimation under Multiple Geometry Representations. Our implementation is based on [FFB6D](https://github.com/ethnhe/FFB6D) and [HybridPose](https://github.com/chensong1995/HybridPose). We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.
## Introduction
Deep learning-based 6D object pose estimation methods from a single RGBD image have recently received increasing attention because of their powerful representation learning capabilities. These methods, however, cannot handle severe occlusion and truncation. In this paper, we present a novel 6D object
pose estimation method based on multiple geometry representations. Specifically, we introduce a network to fuse the appearance and geometry features
extracted from input color and depth images. Then, we utilize these per-point fusion features to estimate keypoint offsets, edge vectors, and dense
symmetry correspondences in the canonical coordinate system. Finally, a two-stage pose regression module is applied to compute the 6D pose of an object. Relative to the unitary 3D keypoint-based strategy, such combinationof multiple geometry representations provides sufficient and diverse information, especially for occluded or truncated scenes.
![Pipeline](https://github.com/JiChun-Wang/MGRNet/blob/main/assert/pipeline.png)
## Installation
+ Set up the python environment:
```shell
conda create -n mgrnet python=3.7
conda activate mgrnet

# install torch 1.8.1 built from cuda 11.1
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    
pip install -r requirement.txt
```
+ Install [apex](https://github.com/NVIDIA/apex):
```shell    
git clone https://github.com/NVIDIA/apex
export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5"  # set the target architecture manually, suggested in issue https://github.com/NVIDIA/apex/issues/605#issuecomment-554453001
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```
+ Install [normalSpeed](https://github.com/hfutcgncas/normalSpeed), a fast and light-weight normal map estimator:
    ```shell
    git clone https://github.com/hfutcgncas/normalSpeed.git
    cd normalSpeed/normalSpeed
    python setup.py install --user
    cd ..
    ```
+ Install tkinter through `sudo apt get install python3-tk `
+ Compile [RandLA-Net](https://github.com/qiqihaer/RandLA-Net-pytorch) operators:
```shell
cd models/RandLA/
sh compile_op.sh
```
+ Compile the pose regressor
```shell
cd lib/regressor
make
```
## Dataset set-up
s
