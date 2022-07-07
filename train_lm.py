import os
import time
import math
import tqdm
import shutil
import argparse
import resource
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import pickle as pkl
from collections import namedtuple
from cv2 import imshow, waitKey

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from common import Config, ConfigRandLA
import models.pytorch_utils as pt_utils
from models.ffb6d import FFB6D
from models.loss import OFLoss, FocalLoss
import datasets.linemod.linemod_dataset as dataset_desc
import datasets.linemod.occlinemod_dataset as occ_dataset_desc
import datasets.linemod.trunlinemod_dataset as trunc_dataset_desc
from lib.coretrainer_lm import Trainer

from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
from apex.multi_tensor_apply import multi_tensor_applier

def parse_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument("-weight_decay", type=float, default=0, help="L2 regularization coeff [default: 0.0]")
    parser.add_argument("-lr", type=float, default=1e-3, help="Initial learning rate [default: 1e-2]")
    parser.add_argument("-lr_decay", type=float, default=1e-5, help="Learning rate decay gamma [default: 0.5]")
    parser.add_argument("-decay_step", type=float, default=2e5, help="Learning rate decay step [default: 20]")
    parser.add_argument("-bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9]")
    parser.add_argument("-bn_decay", type=float, default=0.5, help="Batch norm momentum decay gamma [default: 0.5]")
    parser.add_argument("-checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument("-epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("-eval_net", action='store_true', help="whether is to eval net.")
    parser.add_argument('--cls', type=str, default="ape",
                        help="Target object. (ape, benchvise, cam, can, cat, driller, duck, eggbox, glue, " +
                             "holepuncher, iron, lamp, phone)")
    parser.add_argument('--test_occ', action="store_true", help="To eval occlusion linemod or not.")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-test_pose", action="store_true")
    parser.add_argument("-test_gt", action="store_true")
    parser.add_argument("-cal_metrics", action="store_true")
    parser.add_argument("-view_dpt", action="store_true")
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=8, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--gpu', type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--keep_batchnorm_fp32', default=True)
    parser.add_argument('--opt_level', default="O0", type=str, help='opt level of apex mix presision trainig.')
    args = parser.parse_args()
    return args

# color_lst = [(0, 0, 0)]
# for i in range(config.n_objects):
#     col_mul = (255 * 255 * 255) // (i+1)
#     color = (col_mul//(255*255), (col_mul//255) % 255, col_mul % 255)
#     color_lst.append(color)


lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# def view_labels(rgb_chw, cld_cn, labels, K=config.intrinsic_matrix['linemod']):
#     rgb_hwc = np.transpose(rgb_chw[0].numpy(), (1, 2, 0)).astype(np.uint8).copy()
#     cld_nc = np.transpose(cld_cn.numpy(), (1, 0)).copy()
#     p2ds = bs_utils.project_p3d(cld_nc, 1.0, K).astype(np.int32)
#     labels = labels.squeeze().contiguous().cpu().numpy()
#     colors = []
#     h, w = rgb_hwc.shape[0], rgb_hwc.shape[1]
#     rgb_hwc = np.zeros((h, w, 3), "uint8")
#     for lb in labels:
#         if int(lb) == 0:
#             c = (0, 0, 0)
#         else:
#             c = color_lst[int(lb)]
#         colors.append(c)
#     show = bs_utils.draw_p2ds(rgb_hwc, p2ds, 3, colors)
#     return show


def train():
    args = parse_args()
    args.world_size = args.gpus * args.nodes
    print("local_rank:", args.local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = Config(ds_name='linemod', cls_type=args.cls)

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
    torch.manual_seed(0)

    # multi-proc setup
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    # dataset
    if not args.eval_net:
        train_ds = dataset_desc.Dataset('train', cls_type=args.cls)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.mini_batch_size, shuffle=False,
            drop_last=True, num_workers=4, sampler=train_sampler, pin_memory=True
        )

        test_ds = dataset_desc.Dataset('test', cls_type=args.cls)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.val_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=test_sampler
        )
    else:
        val_ds = dataset_desc.Dataset('trainval', cls_type=args.cls)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.test_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=val_sampler
        )

        test_ds = occ_dataset_desc.Dataset(cls_type=args.cls)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=test_sampler
        )

    # model, load status from checkpoint
    rndla_cfg = ConfigRandLA
    model = FFB6D(n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg, n_kps=config.n_keypoints)
    it = -1
    best_loss = 1e10
    start_epoch = 0
    if args.checkpoint is not None:
        filename = "{}.pth.tar".format(args.checkpoint)
        ck = torch.load(filename)
        start_epoch = ck.get("epoch", 0)
        it = ck.get("it", 0.0)
        best_loss = ck.get("best_prec", None)
        if ck["model_state"] is not None:
            ck_st = ck['model_state']
            if 'module' in list(ck_st.keys())[0]:
                tmp_ck_st = {}
                for k, v in ck_st.items():
                    tmp_ck_st[k.replace("module.", "")] = v
                ck_st = tmp_ck_st
            model.load_state_dict(ck_st)
    model = convert_syncbn_model(model)
    device = torch.device('cuda:{}'.format(args.local_rank))
    print('local_rank:', args.local_rank)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    if args.checkpoint is not None:
        if ck["optimizer_state"] is not None:
            optimizer.load_state_dict(ck["optimizer_state"])
    if args.eval_net:
        criterion, criterion_of = FocalLoss(gamma=2), OFLoss()
    else:
        criterion, criterion_of = FocalLoss(gamma=2).to(device), OFLoss().to(device)

    # multi-gpu setup
    opt_level = args.opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    if args.checkpoint is not None:
        if ck.get("amp", None) is not None:
            amp.load_state_dict(ck["amp"])

    if not args.eval_net:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True
        )
        # model = DistributedDataParallel(model, delay_allreduce=True)
        clr_div = 2
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-3,
            cycle_momentum=False,
            step_size_up=config.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus,
            step_size_down=config.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus,
            mode='triangular'
        )
        # lf = lambda x: (args.lr - args.lr_decay) / 2  * math.cos(x * math.pi / config.n_total_epoch) \
        #                + (args.lr + args.lr_decay) / 2
        # lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)
    else:
        lr_scheduler = None

    bnm_lmbd = lambda it: max(
        args.bn_momentum
        * args.bn_decay ** (int(it * config.mini_batch_size / args.decay_step)),
        bnm_clip,
    )
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bnm_lmbd, last_epoch=it
    )

    it = max(it, 0)  # for the initialize value of `trainer.train`

    checkpoint_fd = config.log_model_dir

    trainer = Trainer(
        model,
        criterion,
        criterion_of,
        optimizer,
        checkpoint_name=os.path.join(checkpoint_fd, "MGRNet_%s" % args.cls),
        best_name=os.path.join(checkpoint_fd, "MGRNet_%s_best" % args.cls),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        config=config,
        args=args
    )

    if args.eval_net:
        # start = time.time()
        # val_loss, res = trainer.eval_epoch(
        #     test_loader, is_test=True, test_pose=args.test_pose
        # )
        # end = time.time()
        # if args.local_rank == 0:
        #     print("\nUse time: ", end - start, 's')
        trainer.generate_data(test_loader, val_loader)
        trainer.compute_add_score()
        # trainer.compute_pose_score()
    else:
        trainer.train(
            it, start_epoch, config.n_total_epoch, train_loader, None,
            test_loader, best_loss=best_loss,
            tot_iter=config.n_total_epoch * train_ds.minibatch_per_epoch // args.gpus,
            clr_div=clr_div
        )


if __name__ == "__main__":
    train()
