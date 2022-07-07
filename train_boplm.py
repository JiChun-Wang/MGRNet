import os
import time
import argparse
import resource
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import torch.backends.cudnn as cudnn

from common import Config, ConfigRandLA
import datasets.lmo.boplm_dataset as dataset_desc

import models.pytorch_utils as pt_utils
from models.ffb6d import FFB6D
from models.loss import OFLoss, FocalLoss
from lib.coretrainer_boplm import Trainer

from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier


def parse_args():
    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument("-weight_decay", type=float, default=0, help="L2 regularization coeff [default: 0.0]")
    parser.add_argument("-lr", type=float, default=1e-2, help="Initial learning rate [default: 1e-2]")
    parser.add_argument("-lr_decay", type=float, default=0.5, help="Learning rate decay gamma [default: 0.5]")
    parser.add_argument("-decay_step", type=float, default=2e5, help="Learning rate decay step [default: 20]")
    parser.add_argument("-bn_momentum", type=float, default=0.9, help="Initial batch norm momentum [default: 0.9]")
    parser.add_argument("-bn_decay", type=float, default=0.5, help="Batch norm momentum decay gamma [default: 0.5]")
    parser.add_argument("-checkpoint", type=str, default=None, help="Checkpoint to start from")
    parser.add_argument("-epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("-eval_net", action='store_true', help="whether is to eval net.")
    parser.add_argument("-test", action="store_true")
    parser.add_argument("-test_pose", action="store_true")
    parser.add_argument("-test_gt", action="store_true")
    parser.add_argument("-cal_metrics", action="store_true")
    parser.add_argument("-view_dpt", action="store_true")
    parser.add_argument('-debug', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu_id', type=list, default=[0, 1])
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--gpu', type=str, default="0,1")
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--keep_batchnorm_fp32', default=True)
    parser.add_argument('--opt_level', default="O0", type=str, help='opt level of apex mix presision trainig.')

    args = parser.parse_args()
    return args


lr_clip = 1e-5
bnm_clip = 1e-2


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def train():
    args = parse_args()
    args.world_size = args.gpus * args.nodes
    print("local_rank:", args.local_rank)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = Config(ds_name='boplm')

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (30000, rlimit[1]))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)
    torch.manual_seed(0)

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
    )

    if not args.eval_net:
        train_ds = dataset_desc.Dataset('train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=config.mini_batch_size, shuffle=False,
            drop_last=True, num_workers=4, sampler=train_sampler, pin_memory=True
        )

        test_ds = dataset_desc.Dataset('test')
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.val_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=test_sampler
        )
    else:
        val_ds = dataset_desc.Dataset('trainval')
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=config.test_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=val_sampler
        )

        test_ds = dataset_desc.Dataset('test')
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.val_mini_batch_size, shuffle=False,
            drop_last=False, num_workers=4, sampler=test_sampler
        )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )

    # default value
    it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
    best_loss = 1e10
    start_epoch = 1

    # load status from checkpoint
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            ck = torch.load(args.checkpoint)
            start_epoch = ck.get('epoch', 0)
            it = ck.get('it', 0)
            best_prec = ck.get("best_prec", None)
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

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.checkpoint is not None:
        optimizer.load_state_dict(ck["optimizer_state"])
    if args.eval_net:
        criterion, criterion_of = FocalLoss(gamma=2), OFLoss()
    else:
        criterion, criterion_of = FocalLoss(gamma=2).to(device), OFLoss().to(device)

    opt_level = args.opt_level
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)
    if args.checkpoint is not None:
        amp.load_state_dict(ck["amp"])

    if not args.eval_net:
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[args.local_rank], output_device=args.local_rank,
        #     find_unused_parameters=True
        # )
        model = DistributedDataParallel(model, delay_allreduce=True)
        clr_div = 6
        lr_scheduler = CyclicLR(
            optimizer, base_lr=1e-5, max_lr=1e-3,
            cycle_momentum=False,
            step_size_up=config.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus,
            step_size_down=config.n_total_epoch * train_ds.minibatch_per_epoch // clr_div // args.gpus,
            mode='triangular'
        )
    else:
        lr_scheduler = None

    bnm_lmbd = lambda it: max(
        args.bn_momentum * args.bn_decay ** (int(it * config.mini_batch_size / args.decay_step)),
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
        checkpoint_name=os.path.join(checkpoint_fd, "MGRNet"),
        best_name=os.path.join(checkpoint_fd, "MGRNet"),
        lr_scheduler=lr_scheduler,
        bnm_scheduler=bnm_scheduler,
        config=config,
        args=args
    )

    if args.eval_net:
        # trainer.generate_data(test_loader, val_loader)
        trainer.compute_auc()
    else:
        trainer.train(
            it, start_epoch, config.n_total_epoch, train_loader, None, test_loader,
            best_loss=best_loss, tot_iter=config.n_total_epoch * train_ds.minibatch_per_epoch // args.gpus
        )



if __name__ == "__main__":
    train()
