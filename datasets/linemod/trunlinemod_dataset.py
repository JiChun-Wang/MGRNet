#!/usr/bin/env python3
import os
import pickle
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
from utils.basic_utils import PoseTransformer
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
from termcolor import colored
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


class Dataset():
    def __init__(self, cls_type="duck", DEBUG=False):
        self.DEBUG = DEBUG
        self.config = Config(ds_name='linemod', cls_type=cls_type)
        self.bs_utils = Basic_Utils(self.config)

        self.xmap = np.array([[j for i in range(256)] for j in range(256)])
        self.ymap = np.array([[i for i in range(256)] for j in range(256)])

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.cls_type = cls_type
        self.obj_dict = self.config.lm_obj_dict
        self.cls_id = self.obj_dict[cls_type]
        self.cls_root = os.path.join(self.config.lm_root, 'Truncation_linemod/{}'.format(cls_type))
        self.real_lst = list(filter(lambda x: x.endswith('pkl'), os.listdir(self.cls_root)))
        self.rng = np.random
        print("Truncation linemod dataset size: {}".format(len(self.real_lst)))

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item):
        sample_id = int(item.split('_')[0])
        # image
        image_name = os.path.join(self.cls_root, '{:04d}_rgb.png'.format(sample_id))
        with Image.open(image_name) as ri:
            rgb = np.array(ri)[:, :, :3]
        # depth
        with Image.open(os.path.join(self.cls_root, '{:04d}_dpt.png'.format(sample_id))) as di:
            dpt_mm = np.array(di)
        # mask
        with Image.open(os.path.join(self.cls_root, '{:04d}_msk.png'.format(sample_id))) as li:
            labels = np.array(li)
            labels = (labels > 0).astype(np.uint8)
        # RT
        with open(os.path.join(self.cls_root, '{:04d}_info.pkl'.format(sample_id)), 'rb') as f:
            info = pickle.load(f)
        RT = info[0]
        # K
        K = info[1]
        cam_scale = 1000.0
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        rgb_labels = labels.copy()

        dpt_mm = dpt_mm.copy().astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )
        if self.DEBUG:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            imshow("nrm_map", show_nrm_map)

        dpt_m = dpt_mm.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0

        msk_dp = dpt_mm > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > self.config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.config.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, RT
        )

        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            cld = sub_pts

        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()
            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
        if self.DEBUG:
            for ip, xyz in enumerate(xyz_lst):
                pcld = xyz.reshape(3, -1).transpose(1, 0)
                p2ds = self.bs_utils.project_p3d(pcld, cam_scale, K)
                srgb = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                # imshow("rz_pcld_%d" % ip, srgb)
                p2ds = self.bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
                srgb1 = self.bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                # imshow("rz_pcld_%d_rnd" % ip, srgb1)
        # print(
        #     "kp3ds:", kp3ds.shape, kp3ds, "\n",
        #     "kp3ds.mean:", np.mean(kp3ds, axis=0), "\n",
        #     "ctr3ds:", ctr3ds.shape, ctr3ds, "\n",
        #     "cls_ids:", cls_ids, "\n",
        #     "labels.unique:", np.unique(labels),
        # )

        item_dict = dict(
            img_id=np.array([sample_id], dtype=np.int32),
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
        )
        item_dict.update(inputs)
        if self.DEBUG:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, RT):
        RTs = np.zeros((self.config.n_objects, 3, 4))
        kp3ds = np.zeros((self.config.n_objects, self.config.n_keypoints, 3))
        ctr3ds = np.zeros((self.config.n_objects, 3))
        cls_ids = np.zeros((self.config.n_objects, 1))
        kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3))
        for i, cls_id in enumerate([1]):
            RTs[i] = RT
            r = RT[:, :3]
            t = RT[:, 3]

            ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="linemod")[:, None]
            ctr = np.dot(ctr.T, r.T) + t
            ctr3ds[i, :] = ctr[0]
            msk_idx = np.where(labels == cls_id)[0]

            target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]
            cls_ids[i, :] = np.array([1])

            if self.config.n_keypoints == 8:
                kp_type = 'farthest'
            else:
                kp_type = 'farthest{}'.format(self.config.n_keypoints)
            kps = self.bs_utils.get_kps(
                self.cls_type, kp_type=kp_type, ds_type='linemod'
            )
            kps = np.dot(kps, r.T) + t
            kp3ds[i] = kps

            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0*kp))
            target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.real_lst)

    def __getitem__(self, idx):
        item = self.real_lst[idx]
        return self.get_item(item)


def main():
    # config.mini_batch_size = 1
    cls = 'ape'
    ds = Dataset(cls, DEBUG=True)
    idx = 0
    while True:
        datum = ds.__getitem__(idx)
        idx += 1
        K = datum['K']
        cam_scale = datum['cam_scale']
        rgb = datum['rgb'].transpose(1, 2, 0)[..., ::-1].copy()  # [...,::-1].copy()
        for i in range(22):
            pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
            p2ds = ds.bs_utils.project_p3d(pcld, cam_scale, K)
            # rgb = ds[cat].bs_utils.draw_p2ds(rgb, p2ds)
            kp3d = datum['kp_3ds'][i]
            if kp3d.sum() < 1e-6:
                break
            kp_2ds = ds.bs_utils.project_p3d(kp3d, cam_scale, K)
            rgb = ds.bs_utils.draw_p2ds(
                rgb, kp_2ds, 3, ds.bs_utils.get_label_color(datum['cls_ids'][i][0], mode=1)
            )
            ctr3d = datum['ctr_3ds'][i]
            ctr_2ds = ds.bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
            rgb = ds.bs_utils.draw_p2ds(
                rgb, ctr_2ds, 4, (0, 0, 255)
            )
        imshow('rgb', rgb)
        cmd = waitKey(0)
        if cmd == ord('q'):
            exit()
        else:
            continue


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
