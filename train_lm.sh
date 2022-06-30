#!/bin/bash
n_gpu=2
cls='phone'
opt_level='O0'
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_lm.py --gpus=$n_gpu --cls=$cls --opt_level=$opt_level
