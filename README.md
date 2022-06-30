# MGRNet: Multiple geometry representations for 6D object pose estimation in occluded or truncated scenes
This repository contains authors' implementation of MGRNet: 6D Object Pose Estimation under Multiple Geometry Representations. Our implementation is based on [FFB6D](https://github.com/ethnhe/FFB6D) and [HybridPose](https://github.com/chensong1995/HybridPose). We warmly welcome any discussions related to our implementation and our paper. Please feel free to open an issue.
## Introduction
Deep learning-based 6D object pose estimation methods from a single RGBD image have recently received increasing attention because of their powerful representation learning capabilities. These methods, however, cannot handle severe occlusion and truncation. In this paper, we present a novel 6D object
pose estimation method based on multiple geometry representations. Specifically, we introduce a network to fuse the appearance and geometry features
extracted from input color and depth images. Then, we utilize these per-point fusion features to estimate keypoint offsets, edge vectors, and dense
symmetry correspondences in the canonical coordinate system. Finally, a two-stage pose regression module is applied to compute the 6D pose of an object. Relative to the unitary 3D keypoint-based strategy, such combinationof multiple geometry representations provides sufficient and diverse information, especially for occluded or truncated scenes.
![Pipeline](https://github.com/JiChun-Wang/MGRNet/blob/main/assert/pipeline.png)
