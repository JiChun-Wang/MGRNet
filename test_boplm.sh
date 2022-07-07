#!/bin/bash
tst_mdl=train_log/boplm/checkpoints/MGRNet_best.pth.tar  # checkpoint to test.
python3 -m torch.distributed.launch --nproc_per_node=1 train_boplm.py --gpu '0' -eval_net -checkpoint $tst_mdl -test -test_pose # -debug
