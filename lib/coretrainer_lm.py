import os
import time

import tqdm
import shutil
import math

import numpy as np
import cv2
from cv2 import imshow, waitKey
from sklearn.neighbors import KDTree
from scipy.linalg import logm
from numpy import linalg as LA
import torch
from torch.utils.tensorboard import SummaryWriter
from apex import amp

from utils.basic_utils import Basic_Utils
from lib.regressor.regressor import load_wrapper, get_2d_ctypes
from utils.pvn3d_eval_utils_kpls import best_fit_transform


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def compute_add_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):
    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred
    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        distance = np.linalg.norm(pts_xformed_gt - pts_xformed_pred, axis=0)
        mean_distances[i] = np.mean(distance)

    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score


def compute_adds_score(pts3d, diameter, pose_gt, pose_pred, percentage=0.1):

    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    mean_distances = np.zeros((count,), dtype=np.float32)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            mean_distances[i] = np.inf
            continue
        pts_xformed_gt = R_gt[i] * pts3d.transpose() + t_gt[i]
        pts_xformed_pred = R_pred[i] * pts3d.transpose() + t_pred[i]
        kdt = KDTree(pts_xformed_gt.transpose(), metric='euclidean')
        distance, _ = kdt.query(pts_xformed_pred.transpose(), k=1)
        mean_distances[i] = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum() / count
    return score


def compute_pose_error(diameter, pose_gt, pose_pred):

    R_gt, t_gt = pose_gt
    R_pred, t_pred = pose_pred

    count = R_gt.shape[0]
    R_err = np.zeros(count)
    t_err = np.zeros(count)
    for i in range(count):
        if np.isnan(np.sum(t_pred[i])):
            continue
        r_err = logm(np.dot(R_pred[i].transpose(), R_gt[i])) / 2
        R_err[i] = LA.norm(r_err, 'fro')
        t_err[i] = LA.norm(t_pred[i] - t_gt[i])
    return np.median(R_err) * 180 / np.pi, np.median(t_err) / diameter


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel) or \
                isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "amp": amp.state_dict(),
    }


def save_checkpoint(
        state, is_best, filename="checkpoint", bestname="model_best",
        bestname_pure='ffb6d_best'
):
    filename = "{}.pth.tar".format(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "{}.pth.tar".format(bestname))
        shutil.copyfile(filename, "{}.pth.tar".format(bestname_pure))


class Trainer(object):
    r"""
        Reasonably generic trainer for pytorch models

    Parameters
    ----------
    model : pytorch model
        Model to be trained
    model_fn : function (model, inputs, labels) -> preds, loss, accuracy
    optimizer : torch.optim
        Optimizer for model
    checkpoint_name : str
        Name of file to save checkpoints to
    best_name : str
        Name of file to save best model to
    lr_scheduler : torch.optim.lr_scheduler
        Learning rate scheduler.  .step() will be called at the start of every epoch
    bnm_scheduler : BNMomentumScheduler
        Batchnorm momentum scheduler.  .step() will be called at the start of every epoch
    """

    def __init__(
        self,
        model,
        criterion,
        criterion_of,
        optimizer,
        checkpoint_name="ckpt",
        best_name="best",
        lr_scheduler=None,
        bnm_scheduler=None,
        args=None,
        config=None,
        viz=None,
    ):
        self.model, self.criterion, self.criterion_of, self.optimizer, self.lr_scheduler, self.bnm_scheduler, \
        self.args, self.config = (
            model,
            criterion,
            criterion_of,
            optimizer,
            lr_scheduler,
            bnm_scheduler,
            args,
            config
        )

        self.bs_utils = Basic_Utils(self.config)

        self.checkpoint_name, self.best_name = checkpoint_name, best_name

        self.writer = SummaryWriter(log_dir=config.log_traininfo_dir)

        self.training_best, self.eval_best = {}, {}
        self.viz = viz

    def model_fn(self, data, it=0, epoch=0, is_eval=False, is_test=False):
        if is_eval:
            self.model.eval()
        with torch.set_grad_enabled(not is_eval):
            cu_dt = {}
            for key in data.keys():
                if data[key].dtype in [np.float32, np.uint8]:
                    cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
                elif data[key].dtype in [np.int32, np.uint32]:
                    cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
                elif data[key].dtype in [torch.uint8, torch.float32]:
                    cu_dt[key] = data[key].float().cuda()
                elif data[key].dtype in [torch.int32, torch.int16]:
                    cu_dt[key] = data[key].long().cuda()

            end_points = self.model(cu_dt)

            if not is_test:
                labels = cu_dt['labels']
                loss_rgbd_seg = self.criterion(
                    end_points['pred_rgbd_segs'], labels.view(-1)
                ).sum()
                loss_kp_of = self.criterion_of(
                    end_points['pred_kp_ofs'], cu_dt['kp_targ_ofst'], labels
                ).sum()
                loss_graph = self.criterion_of(
                    end_points['pred_graph'], cu_dt['graph_targ'], labels
                ).sum()
                loss_sym_cor = self.criterion_of(
                    end_points['pred_sym_cor'], cu_dt['sym_cor_targ'], labels
                ).sum()
                loss_ctr_of = self.criterion_of(
                    end_points['pred_ctr_ofs'], cu_dt['ctr_targ_ofst'], labels
                ).sum()

                loss_lst = [
                    (loss_rgbd_seg, 2.0),
                    (loss_kp_of, 1.0),
                    (loss_graph, 0.1),
                    (loss_sym_cor, 1),
                    (loss_ctr_of, 1.0),
                ]
                loss = sum([ls * w for ls, w in loss_lst])

                _, cls_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)
                acc_rgbd = (cls_rgbd == labels).float().sum() / labels.numel()

                loss_dict = {
                    'loss_rgbd_seg': loss_rgbd_seg.item(),
                    'loss_kp_of': loss_kp_of.item(),
                    'loss_graph': loss_graph.item(),
                    'loss_sym_cor': loss_sym_cor.item(),
                    'loss_ctr_of': loss_ctr_of.item(),
                    'loss_all': loss.item()
                }
                acc_dict = {
                    'acc_rgbd': acc_rgbd.item(),
                }
                info_dict = loss_dict.copy()
                info_dict.update(acc_dict)

                if not is_eval:
                    if self.args.local_rank == 0:
                        self.writer.add_scalars('loss', loss_dict, it)
                        self.writer.add_scalars('train_acc', acc_dict, it)

                return (cu_dt, end_points, loss, info_dict)
            else:
                return (cu_dt, end_points)

    def meanshift(self, A, bandwidth=0.05, max_iter=300, ret_mid_res=False):        # params: A: [N, 3]
        stop_thresh = bandwidth * 1e-3

        def gaussian_kernel(distance, bandwidth):
            return torch.exp(-0.5 * ((distance / bandwidth)) ** 2) \
                   / (bandwidth * math.sqrt(2 * np.pi))

        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            dis = torch.norm(C.reshape(1, N, c) - C.reshape(N, 1, c), dim=2)
            w = gaussian_kernel(dis, bandwidth).reshape(N, N, 1)
            new_C = torch.sum(w * C, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Cdis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Cdis) < stop_thresh or it > max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        dis = torch.norm(C.view(N, 1, c) - C.view(1, N, c), dim=2)
        num_in = torch.sum(dis < bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < bandwidth
        if not ret_mid_res:
            return C[max_idx, :], labels
        else:
            return C, dis

    def eval_epoch(self, d_loader, it=0):
        self.model.eval()

        eval_dict = {}
        total_loss = 0.0
        count = 1
        for i, data in tqdm.tqdm(
            enumerate(d_loader), leave=False, desc="val"
        ):
            count += 1
            self.optimizer.zero_grad()

            _, _, loss, eval_res = self.model_fn(data, is_eval=True)

            if 'loss_target' in eval_res.keys():
                total_loss += eval_res['loss_target']
            else:
                total_loss += loss.item()

            for k, v in eval_res.items():
                if v is not None:
                    eval_dict[k] = eval_dict.get(k, []) + [v]

        mean_eval_dict = {}
        acc_dict = {}
        for k, v in eval_res.items():
            per = 100 if 'acc' in k else 1
            mean_eval_dict[k] = np.array(v).mean() * per
            if 'acc' in k:
                acc_dict[k] = v
        for k, v in mean_eval_dict.items():
            print(k, v)

        if self.args.local_rank == 0:
            self.writer.add_scalars('val_acc', acc_dict, it)

        return total_loss / count, eval_dict

    def train(self, start_it, start_epoch, n_epochs, train_loader, train_sampler, test_loader=None, best_loss=0.0,
              log_epoch_f=None, tot_iter=1, clr_div=6):

        print("Totally train %d iters per gpu." % tot_iter)

        def is_to_eval(epoch, it):
            if it == 100:
                return True, 1
            wid = tot_iter // clr_div
            if (it // wid) % 2 == 1:
                eval_frequency = wid // 15
            else:
                eval_frequency = wid // 6
            to_eval = (it % eval_frequency) == 0
            return to_eval, eval_frequency

        it = start_it
        _, eval_frequency = is_to_eval(0, it)
        bs_per_epoch = train_loader.dataset.minibatch_per_epoch // self.args.gpus

        # with tqdm.tqdm(range(self.config.n_total_epoch), desc="%s_epochs" % self.config.cls_type) as tbar, \
        #      tqdm.tqdm(total=eval_frequency, leave=False, desc="train") as pbar:
        for epoch in range(start_epoch, self.config.n_total_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            # Reset numpy seed.
            # REF: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed()
            if log_epoch_f is not None:
                os.system("echo {} > {}".format(epoch, log_epoch_f))
            for i_batch, batch in enumerate(train_loader):
                self.model.train()

                self.optimizer.zero_grad()
                _, _, loss, res = self.model_fn(batch, it=it, is_eval=False)

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                lr = get_lr(self.optimizer)
                if self.args.local_rank == 0:
                    self.writer.add_scalar('lr/lr', lr, it)

                self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(it)

                if self.bnm_scheduler is not None:
                    self.bnm_scheduler.step(it)

                it += 1

                if self.args.local_rank == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Mask: {mask:.4f}\t'
                          'Kpts: {kpts:.4f}\t'
                          'Graph: {graph:.4f}\t'
                          'Sym: {sym:.4f}\t'
                          'Ctr: {ctr:.4f}\t'
                          'Total: {total:.4f}'.format(epoch, i_batch, bs_per_epoch,
                                                        mask=res['loss_rgbd_seg'],
                                                        kpts=res['loss_kp_of'],
                                                        graph=res['loss_graph'],
                                                        sym=res['loss_sym_cor'],
                                                        ctr=res['loss_ctr_of'],
                                                        total=res['loss_all']))

                eval_flag, eval_frequency = is_to_eval(epoch, it)
                if eval_flag:
                    if test_loader is not None:
                        val_loss, res = self.eval_epoch(test_loader, it=it)
                        print("val_loss", val_loss)

                        is_best = val_loss < best_loss
                        best_loss = min(best_loss, val_loss)
                        if self.args.local_rank == 0:
                            save_checkpoint(
                                checkpoint_state(
                                    self.model, self.optimizer, val_loss, epoch, it
                                ),
                                is_best,
                                filename=self.checkpoint_name,
                                bestname=self.best_name+'_%.4f' % val_loss,
                                bestname_pure=self.best_name
                            )
                            info_p = self.checkpoint_name.replace(
                                '.pth.tar', '_epoch.txt'
                            )
                            os.system(
                                'echo {} {} >> {}'.format(
                                    it, val_loss, info_p
                                )
                            )

            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step()
        if self.args.local_rank == 0:
            self.writer.close()
        return best_loss

    def filter_symmetry(self, vecs_pred, sigma=0.01, min_count=100, n_neighbors=100):
        if len(vecs_pred) < min_count:
            qs1_cross_qs2 = np.zeros((0, 3), dtype=np.float32)
            symmetry_weight = np.zeros((0,), dtype=np.float32)
            return qs1_cross_qs2, symmetry_weight
        vecs_pred /= np.sqrt(np.sum(vecs_pred**2, axis=1)).reshape((-1, 1))
        kdt = KDTree(vecs_pred, leaf_size=40, metric='euclidean') # following matlab default values
        dis, _ = kdt.query(vecs_pred, k=n_neighbors)
        saliency = np.mean(dis * dis, axis=1, dtype=np.float32)
        order = np.argsort(saliency)
        seeds = np.zeros((2, order.shape[0]), dtype=np.uint32)
        seeds[0][0] = order[0]
        seeds[1][0] = 1
        seeds_size = 1
        flags = np.zeros((order.shape[0],), dtype=np.uint32)
        flags[order[0]] = 0
        for i in range(1, order.shape[0]):
            vec = vecs_pred[order[i]]
            candidates = vecs_pred[seeds[0]]
            dif = candidates - vec
            norm = np.linalg.norm(dif, axis=1)
            closest_seed_i = norm.argmin()
            min_dis = norm[closest_seed_i]
            if min_dis < sigma:
                flags[order[i]] = closest_seed_i
                seeds[1][closest_seed_i] = seeds[1][closest_seed_i] + 1
            else:
                seeds[0, seeds_size] = order[i]
                seeds[1, seeds_size] = 1
                flags[order[i]] = seeds_size
                seeds_size += 1
        seeds = seeds[:, :seeds_size]
        valid_is = np.argwhere(seeds[1] > (np.max(seeds[1]) / 3)).transpose()[0]
        seeds = seeds[:, valid_is]
        n_symmetry = seeds.shape[1]
        qs1_cross_qs2 = np.zeros((n_symmetry, 3), dtype=np.float32)
        for i in range(n_symmetry):
            row_is = np.argwhere(flags == valid_is[i]).transpose()[0]
            qs1_cross_qs2[i] = np.mean(vecs_pred[row_is], axis=0)
            qs1_cross_qs2[i] /= np.linalg.norm(qs1_cross_qs2[i])
        symmetry_weight = np.float32(seeds[1])
        symmetry_weight /= np.max(symmetry_weight)
        return qs1_cross_qs2, symmetry_weight

    def vote_keypoints(self, pcld, mask, ctr_ofs_pred, kps_ofs_pred, radius=0.04, use_ctr=False, use_ctr_clus_flter=True):
        n_kps, n_pts, _ = kps_ofs_pred.size()

        pred_ctr = pcld - ctr_ofs_pred[0]
        pred_kps = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - kps_ofs_pred

        cls_id = 1
        cls_msk = mask == cls_id
        cls_voted_ctr = pred_ctr[cls_msk, :]
        ctr, ctr_labels = self.meanshift(cls_voted_ctr, bandwidth=radius)
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1

        pred3D_cam = torch.zeros(n_kps, 3).cuda()
        point_inv_half_var = torch.zeros(n_kps, 3, 3).cuda()
        cls_voted_kps = pred_kps[:, cls_msk, :]
        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps
        for i, kpt_map in enumerate(in_pred_kp):
            pred3D_cam[i], _ = self.meanshift(kpt_map, bandwidth=radius)
            point_inv_half_var[i] = self.bs_utils.calc_cov_cuda(kpt_map)
        if use_ctr:
            pred3D_cam = torch.cat((pred3D_cam, ctr.unsqueeze(0)), 0)
        return pred3D_cam, point_inv_half_var

    def vote_graph(self, pcld, mask, ctr_ofs_pred, graph_pred, radius=0.03, use_ctr_clus_flter=True):
        n_edges, n_pts, _ = graph_pred.size()
        pred_ctr = pcld - ctr_ofs_pred[0]
        cls_id = 1
        cls_msk = mask == cls_id
        cls_voted_ctr = pred_ctr[cls_msk, :]
        ctr, ctr_labels = self.meanshift(cls_voted_ctr, bandwidth=radius)
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1

        vec_pred = torch.zeros(n_edges, 3).cuda()
        edge_inv_half_var = torch.zeros(n_edges, 3, 3).cuda()
        cls_voted_edges = graph_pred[:, cls_msk, :]
        if use_ctr_clus_flter:
            in_pred_edges = cls_voted_edges[:, ctr_labels, :]
        else:
            in_pred_edges = cls_voted_edges
        for i, edge_map in enumerate(in_pred_edges):
            vec_pred[i], _ = self.meanshift(edge_map, bandwidth=radius)
            edge_inv_half_var[i] = self.bs_utils.calc_cov_cuda(edge_map)
        return vec_pred, edge_inv_half_var

    def fill_intermediate_predictions(self,
                                      regressor,
                                      predictions,
                                      pcld,
                                      mask,
                                      kpts_gt_obj,
                                      kpts_pred_loc,
                                      kpts_pred_var,
                                      graph_pred_loc,
                                      graph_pred_var,
                                      # graph_pred,
                                      sym_cor_pred,
                                      normal_gt):
        n_keypts = self.config.n_keypoints
        n_edges = n_keypts * (n_keypts - 1) // 2
        cls_id = 1
        cls_msk = mask == cls_id
        # point3D_obj
        regressor.set_point3D_obj(predictions, get_2d_ctypes(kpts_gt_obj), n_keypts)
        # point3D_cam
        regressor.set_point3D_cam(predictions, get_2d_ctypes(kpts_pred_loc), n_keypts)
        # point_inv_half_var
        point_inv_half_var = np.zeros((n_keypts, 3, 3), dtype=np.float32)
        for i in range(n_keypts):
            try:
                cov = np.matrix(kpts_pred_var[i])
                cov = (cov + cov.transpose()) / 2
                v, u = np.linalg.eig(cov)
                v = np.matrix(np.diag(1. / np.sqrt(v)))
                point_inv_half_var[i] = u * v * u.transpose()
            except:
                point_inv_half_var[i] = np.eye(3)
        point_inv_half_var = point_inv_half_var.reshape((n_keypts, 9))
        regressor.set_point_inv_half_var(predictions, get_2d_ctypes(point_inv_half_var), n_keypts)
        # normal_gt
        regressor.set_normal_gt(predictions, normal_gt.ctypes)
        # vec_pred and edge_inv_half_var : graph_pred: [n_edges, n_pts, 3]
        # vec_pred = np.zeros((n_edges, 3), dtype=np.float32)
        edge_inv_half_var = np.zeros((n_edges, 3, 3), dtype=np.float32)
        # cls_voted_graph = graph_pred[:, cls_msk, :]
        for i in range(n_edges):
            # xs = cls_voted_graph[i][:, 0]
            # ys = cls_voted_graph[i][:, 1]
            # zs = cls_voted_graph[i][:, 2]
            # vec_pred[i] = [xs.mean(), ys.mean(), zs.mean()]
            try:
                # cov = np.cov(np.vstack((xs, ys, zs)))
                cov = np.matrix(graph_pred_var[i])
                cov = (cov + cov.transpose()) / 2
                v, u = np.linalg.eig(cov)
                v = np.matrix(np.diag(1. / np.sqrt(v)))
                edge_inv_half_var[i] = u * v * u.transpose()
            except:
                edge_inv_half_var[i] = np.eye(3)
            # edge_inv_half_var[i] = np.eye(3)
        edge_inv_half_var = edge_inv_half_var.reshape((n_edges, 9))
        # regressor.set_vec_pred(predictions, get_2d_ctypes(vec_pred), n_edges)
        regressor.set_vec_pred(predictions, get_2d_ctypes(graph_pred_loc), n_edges)
        regressor.set_edge_inv_half_var(predictions, get_2d_ctypes(edge_inv_half_var), n_edges)
        # qs1_cross_qs2 and symmetry weight
        cls_voted_sym_cor = sym_cor_pred[cls_msk, :]
        cls_voted_pcld = pcld[cls_msk, :]
        flat = np.zeros((cls_voted_sym_cor.shape[0], 2, 3), dtype=np.float32)
        for i in range(flat.shape[0]):
            x, y, z = cls_voted_pcld[i]
            x_cor, y_cor, z_cor = cls_voted_sym_cor[i]
            flat[i, 0] = [x, y, z]
            flat[i, 1] = [x+x_cor, y+y_cor, z+z_cor]
        qs1_cross_qs2_all = np.zeros((flat.shape[0], 3), dtype=np.float32)
        for i in range(flat.shape[0]):
            qs1 = flat[i][0]
            qs2 = flat[i][1]
            qs1_cross_qs2_all[i] = np.cross(qs1, qs2)
        qs1_cross_qs2_filtered, symmetry_weight = self.filter_symmetry(qs1_cross_qs2_all)
        n_symmetry = qs1_cross_qs2_filtered.shape[0]
        regressor.set_qs1_cross_qs2(predictions, get_2d_ctypes(qs1_cross_qs2_filtered), n_symmetry)
        regressor.set_symmetry_weight(predictions, symmetry_weight.ctypes, n_symmetry)
        # set initial pose calculated by SVD for ablation study
        R, t = best_fit_transform(kpts_gt_obj,
                           kpts_pred_loc)
        pose_init = np.zeros((4, 3), dtype=np.float32)
        pose_init[0] = t
        pose_init[1:] = R.transpose()
        regressor.set_pose(predictions, get_2d_ctypes(pose_init))
        return R, t.reshape((3, 1))

    def search_para(self, regressor, predictions_para, poses_para, normal_gt, diameter, val_set):
        para_id = 0
        cls_id = 1
        for data_id in range(len(val_set['pclds'])):
            cls_msk = val_set['mask_pred'][data_id] == cls_id
            if cls_msk.sum() < 1:
                continue
            predictions = regressor.get_prediction_container(predictions_para, para_id)
            _, _ = self.fill_intermediate_predictions(regressor,
                                               predictions,
                                               val_set['pclds'][data_id],
                                               val_set['mask_pred'][data_id],
                                               val_set['kpts_gt_obj'][data_id],
                                               val_set['kpts_pred_loc'][data_id],
                                               val_set['kpts_pred_var'][data_id],
                                               val_set['edges_pred_loc'][data_id],
                                               val_set['edges_pred_var'][data_id],
                                               # val_set['graph_pred'][data_id],
                                               val_set['sym_cor_pred'][data_id],
                                               normal_gt)

            pose_gt = np.zeros((4, 3), dtype=np.float32)
            r = val_set['RTs'][data_id][:, :3]
            tvec = val_set['RTs'][data_id][:, 3]
            pose_gt[0] = tvec
            pose_gt[1:] = r.transpose()
            regressor.set_pose_gt(poses_para, para_id, get_2d_ctypes(pose_gt))
            para_id += 1
        pi_para = regressor.search_pose_initial(predictions_para, poses_para, para_id, diameter)
        pr_para = regressor.search_pose_refine(predictions_para, poses_para, para_id, diameter)
        return pi_para, pr_para

    def regress_pose(self,
                     regressor,
                     predictions,
                     pi_para,
                     pr_para,
                     pcld,
                     mask,
                     kpts_gt_obj,
                     kpts_pred_loc,
                     kpts_pred_var,
                     graph_pred_loc,
                     graph_pred_var,
                     # graph_pred,
                     sym_cor_pred,
                     normal_gt):
        cls_id = 1
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            R = np.eye(3, dtype=np.float32)
            t = np.zeros((3, 1), dtype=np.float32)
            return R, t
        R_init, t_init = self.fill_intermediate_predictions(regressor,
                                           predictions,
                                           pcld,
                                           mask,
                                           kpts_gt_obj,
                                           kpts_pred_loc,
                                           kpts_pred_var,
                                           # graph_pred,
                                           graph_pred_loc,
                                           graph_pred_var,
                                           sym_cor_pred,
                                           normal_gt)
        # initialize pose: pose initial submodule is closed because ablation study uses ICP(SVD) as pose initial
        # predictions = regressor.initialize_pose_ablation(predictions, pi_para, 1, 1, 1)
        # pose_init = np.zeros((4, 3), dtype=np.float32)
        # regressor.get_pose(predictions, get_2d_ctypes(pose_init))
        # R_init = pose_init[1:].transpose()
        # t_init = pose_init[0].reshape((3, 1))
        # refine pose
        predictions = regressor.refine_pose_ablation(predictions, pr_para, 1, 1, 0)
        pose_final = np.zeros((4, 3), dtype=np.float32)
        regressor.get_pose(predictions, get_2d_ctypes(pose_final))
        R_final = pose_final[1:].transpose()
        t_final = pose_final[0].reshape((3, 1))
        return R_init, t_init, R_final, t_final

    def generate_data(self, test_loader, val_loader, val_size=50):
        self.model.eval()
        normal = val_loader.dataset.normal
        obj_id = self.config.lm_obj_dict[self.args.cls]
        diameter = self.config.lm_r_lst[obj_id]['diameter'] / 1000.0

        val_set = {
            'pclds': [],
            'mask_pred': [],
            'kpts_gt_obj': [],
            'kpts_pred_loc': [],
            'kpts_pred_var': [],
            'edges_pred_loc': [],
            'edges_pred_var': [],
            # 'graph_pred': [],
            'sym_cor_pred': [],
            'RTs': []
        }
        n_examples = len(test_loader.dataset)
        test_set = {
            'R_init': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_init': np.zeros((n_examples, 3, 1), dtype=np.float32),
            'R_pred': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_pred': np.zeros((n_examples, 3, 1), dtype=np.float32),
            'R_gt': np.zeros((n_examples, 3, 3), dtype=np.float32),
            't_gt': np.zeros((n_examples, 3, 1), dtype=np.float32),
            'img_id': np.zeros((n_examples, ), dtype=np.int32)
        }
        regressor = load_wrapper()
        predictions = regressor.new_container()
        predictions_para = regressor.new_container_para()
        poses_para = regressor.new_container_pose()
        with torch.no_grad():
            # search parameters
            keep_searching = True
            for i_batch, batch in enumerate(val_loader):
                if not keep_searching:
                    break
                base_idx = self.config.test_mini_batch_size * i_batch
                cu_dt, end_points, _, _ = self.model_fn(batch, is_eval=True)
                _, cls_rgbds = torch.max(end_points['pred_rgbd_segs'], 1)                # [bs, n_pts]
                pclds = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()     # [bs, n_pts, 3]
                ctr_ofs_pred = end_points['pred_ctr_ofs']                                # [bs, 1, n_pts, 3]
                kps_ofs_pred = end_points['pred_kp_ofs']                                 # [bs, n_kps, n_pts, 3]
                graph_pred = end_points['pred_graph']                                    # [bs, n_edges, n_pts, 3]
                sym_cor_pred = end_points['pred_sym_cor']                                # [bs, 1, n_pts, 3]
                RTs = cu_dt['RTs']
                col = list(zip(pclds, cls_rgbds, ctr_ofs_pred, kps_ofs_pred, graph_pred, sym_cor_pred, RTs))
                for i in range(len(col)):
                    pcld, cls_rgbds_, ctr_ofs_pred_, kps_ofs_pred_, graph_pred_, sym_cor_pred_, RT = col[i]
                    if (base_idx + i) < val_size:
                        # kpts_pred_loc: [n_kps, 3], kpts_pred_var: [n_kps, 3, 3]
                        kpts_pred_loc, kpts_pred_var = self.vote_keypoints(pcld, cls_rgbds_, ctr_ofs_pred_, kps_ofs_pred_)
                        # edges_pred_loc: [n_edges, 3], edges_pred_var: [n_edges, 3, 3]
                        edges_pred_loc, edges_pred_var = self.vote_graph(pcld, cls_rgbds_, ctr_ofs_pred_, graph_pred_)
                        if self.config.n_keypoints == 8:
                            kp_type = 'farthest'
                        else:
                            kp_type = 'farthest{}'.format(self.config.n_keypoints)
                        pred3D_obj = self.bs_utils.get_kps(self.args.cls, kp_type=kp_type, ds_type='linemod')
                        val_set['pclds'].append(pcld.cpu().numpy())
                        val_set['mask_pred'].append(cls_rgbds_.detach().cpu().numpy())
                        val_set['kpts_gt_obj'].append(pred3D_obj)
                        val_set['kpts_pred_loc'].append(kpts_pred_loc.detach().cpu().numpy())
                        val_set['kpts_pred_var'].append(kpts_pred_var.detach().cpu().numpy())
                        val_set['edges_pred_loc'].append(edges_pred_loc.detach().cpu().numpy())
                        val_set['edges_pred_var'].append(edges_pred_var.detach().cpu().numpy())
                        # val_set['graph_pred'].append(graph_pred_.detach().cpu().numpy())
                        val_set['sym_cor_pred'].append(sym_cor_pred_[0].detach().cpu().numpy())
                        val_set['RTs'].append(RT[0].cpu().numpy())
                    elif (base_idx + i) == val_size:
                        # search hyper-parameters of regressor module
                        pi_para, pr_para = self.search_para(regressor,
                                                           predictions_para,
                                                           poses_para,
                                                           normal,
                                                           diameter,
                                                           val_set)
                        keep_searching = False
                        break
            regressor.delete_container1(predictions_para, poses_para)

            # prediction
            time_record1 = AverageMeter()
            time_record2 = AverageMeter()
            for i_batch, batch in enumerate(test_loader):
                base_idx = self.config.test_mini_batch_size * i_batch
                start_time = time.time()
                cu_dt, end_points = self.model_fn(batch, is_eval=True, is_test=True)
                time_record1.update(time.time() - start_time)
                _, cls_rgbds = torch.max(end_points['pred_rgbd_segs'], 1)
                pclds = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
                ctr_ofs_pred = end_points['pred_ctr_ofs']
                kps_ofs_pred = end_points['pred_kp_ofs']
                graph_pred = end_points['pred_graph']
                sym_cor_pred = end_points['pred_sym_cor']
                RTs = cu_dt['RTs']
                img_id = cu_dt['img_id']
                col = list(zip(pclds, cls_rgbds, ctr_ofs_pred, kps_ofs_pred, graph_pred, sym_cor_pred, RTs, img_id))
                for i in range(len(col)):
                    pcld, cls_rgbds_, ctr_ofs_pred_, kps_ofs_pred_, graph_pred_, sym_cor_pred_, RT, img_id_ = col[i]
                    cls_id = 1
                    cls_msk = cls_rgbds_ == cls_id
                    start_time = time.time()
                    if cls_msk.sum() < 1:
                        test_set['R_gt'][base_idx + i] = RT[0].cpu().numpy()[:, :3]
                        test_set['t_gt'][base_idx + i] = RT[0].cpu().numpy()[:, 3].reshape(3, 1)
                        test_set['R_pred'][base_idx + i] = np.eye(3, dtype=np.float32)
                        test_set['t_pred'][base_idx + i] = np.zeros((3, 1), dtype=np.float32)
                        test_set['img_id'][base_idx + i] = img_id_[0]
                    else:
                        kpts_pred_loc, kpts_pred_var = self.vote_keypoints(pcld, cls_rgbds_, ctr_ofs_pred_,
                                                                           kps_ofs_pred_)
                        edges_pred_loc, edges_pred_var = self.vote_graph(pcld, cls_rgbds_, ctr_ofs_pred_, graph_pred_)
                        if self.config.n_keypoints == 8:
                            kp_type = 'farthest'
                        else:
                            kp_type = 'farthest{}'.format(self.config.n_keypoints)
                        pred3D_obj = self.bs_utils.get_kps(self.args.cls, kp_type=kp_type, ds_type='linemod')
                        R_init, t_init, R_pred, t_pred = self.regress_pose(regressor,
                                                                           predictions,
                                                                           pi_para,
                                                                           pr_para,
                                                                           pcld.cpu().numpy(),
                                                                           cls_rgbds_.detach().cpu().numpy(),
                                                                           pred3D_obj,
                                                                           kpts_pred_loc.detach().cpu().numpy(),
                                                                           kpts_pred_var.detach().cpu().numpy(),
                                                                           edges_pred_loc.detach().cpu().numpy(),
                                                                           edges_pred_var.detach().cpu().numpy(),
                                                                           # graph_pred_.detach().cpu().numpy(),
                                                                           sym_cor_pred_[0].detach().cpu().numpy(),
                                                                           normal)
                        test_set['R_gt'][base_idx + i] = RT[0].cpu().numpy()[:, :3]
                        test_set['t_gt'][base_idx + i] = RT[0].cpu().numpy()[:, 3].reshape(3, 1)
                        test_set['R_pred'][base_idx + i] = R_pred
                        test_set['t_pred'][base_idx + i] = t_pred
                        test_set['R_init'][base_idx + i] = R_init
                        test_set['t_init'][base_idx + i] = t_init
                        test_set['img_id'][base_idx + i] = img_id_[0]
                    time_record2.update(time.time()- start_time)
            np.save(os.path.join(self.config.log_eval_dir, 'test_set_pose_trunc.npy'), test_set)
            print('Time1: {time.avg:.3f}'.format(time=time_record1))
            print('Time2: {time.avg:.3f}'.format(time=time_record2))
            print('saved')
            regressor.delete_container2(pi_para, pr_para)
            regressor.delete_container3(predictions)

    def compute_add_score(self):
        record = np.load(os.path.join(self.config.log_eval_dir, 'test_set_pose_trunc.npy'), allow_pickle=True).item()
        obj_id = self.config.lm_obj_dict[self.args.cls]
        pts3d = self.bs_utils.get_pointxyz(obj_id, ds_type="linemod").copy()
        diameter = self.config.lm_r_lst[obj_id]['diameter'] / 1000.0
        if self.args.cls in ['eggbox', 'glue']:
            compute_score = compute_adds_score
        else:
            compute_score = compute_add_score
        score = compute_score(np.matrix(pts3d),
                              diameter,
                              (record['R_gt'], record['t_gt']),
                              (record['R_init'], record['t_init']))
        if self.args.local_rank == 0:
            print("Init: ADD(-S) score of {} is: {}".format(self.args.cls, score))
        score = compute_score(np.matrix(pts3d),
                              diameter,
                              (record['R_gt'], record['t_gt']),
                              (record['R_pred'], record['t_pred']))
        if self.args.local_rank == 0:
            print("Refine: ADD(-S) score of {} is: {}".format(self.args.cls, score))

    def compute_pose_score(self):
        record = np.load(os.path.join(self.config.log_eval_dir, 'test_set_pose_trun.npy'), allow_pickle=True).item()
        obj_id = self.config.lm_obj_dict[self.args.cls]
        diameter = self.config.lm_r_lst[obj_id]['diameter'] / 1000.0
        R_err, t_err = compute_pose_error(diameter,
                                          (record['R_gt'], record['t_gt']),
                                          (record['R_init'], record['t_init']))
        if self.args.local_rank == 0:
            print(self.args.cls + ' prediction rotation error is: {} translation error is : {}'.format(R_err, t_err))
