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
+ LineMOD: Download the preprocessed LineMOD dataset from [onedrive link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhebk_connect_ust_hk/ETW6iYHDbo1OsIbNJbyNBkABF7uJsuerB6c0pAiiIv6AHw?e=eXM1UE) or [google drive link](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7). Unzip it and link the unzipped `Linemod_preprocessed/` to `datasets/linemod/Linemod_preprocessed`:
    ```shell
    ln -s path_to_unzipped_Linemod_preprocessed dataset/linemod/
    ```
  Generate rendered and fused data following [raster_triangle](https://github.com/ethnhe/raster_triangle).
  Generate sym_cor labels with [label_real.py](), [label_render.py]() and [label_fuse.py]() for each object.
+ Occlusion LineMOD: Download the Occlusion LineMOD from [Google Drive](https://drive.google.com/file/d/1PItmDj7Go0OBnC1Lkvagz3RRB9qdJUIG/view?usp=sharing) or [Tencent Weiyun](https://share.weiyun.com/50i7KTb). Unzip it and link the unzipped `OcclusionChallengeICCV2015` to `datasets/linemod/Occusion_Linemod`:
    ```shell
    ln -s path_to_unzipped_Occlusion_Linemod dataset/linemod/Occlusion_Linemod
    ```
+ Truncation LineMOD: Download our regenerated Truncation LineMOD dataset from 
+ BOP LineMOD: Download the BOP-version LineMOD dataset from [here](https://bop.felk.cvut.cz/datasets/) mainly including lm_train_pbr.zip, lm_models.zip and lmo_test_all.zip. Unzip them and link to `datasets/lmo/`:
    ```shell
    ln -s path_to_unzipped_train_pbr dataset/lmo/train_pbr
    ln -s path_to_unzipped_models dataset/lmo/models_eval
    ln -s path_to_unzipped_lmo_test dataset/lmo/test
    ```
    Generate sym_cor labels with [label_sym.py]() for all training and testing images.
## Training and evaluating
### Training on the LineMOD dataset
+ Train the model for the target object. Take object ape for example, you should change variable $cls$ to 'ape' and execute the following script in command line:
    ```shell
    bash train_lm.sh
    ```
### Evaluating on the Occlusion and Truncation LineMOD datasets
### Training on the BOP LineMOD dataset
### Evaluating on the BOP LM-O dataset
