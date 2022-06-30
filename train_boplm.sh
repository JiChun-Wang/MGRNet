#!/bin/bash
n_gpu=2  # number of gpu to use
opt_level='O0'
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_boplm.py --gpus=$n_gpu --opt_level=$opt_level
