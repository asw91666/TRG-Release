from pdb import set_trace
import cv2
import socket
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
import time
import os
from models.op import euler_to_rotation_matrix, rotation_matrix_to_euler_angles, uniform_sampled_data, rotation_matrix_z
from util.random import sample_uniform_distribution
from util.vis import plotting_points2d_on_img_cv
from util.pkl import read_pkl, write_pkl
from util.metric import degree2radian, radian2degree
from data.preprocessing import update_after_crop, update_after_resize, gen_trans_from_patch_cv
from scipy.spatial.transform import Rotation
import random
use_jpeg4py = False

data_root = Path('./dataset/ARKitFace')

npz_path = 'npy/kpt_ind.npy'
kpt_ind = np.load(npz_path)

npy_path = 'npy/uv_coords_std_202109.npy'
uv_coords_std = np.load(npy_path)   # (1279, 2)，[0, 1]

npy_path = 'npy/tris_2500x3_202110.npy'
tris_full = np.load(npy_path)
npy_path = 'npy/tris_2388x3.npy'
tris_mask = np.load(npy_path)
txt_path = 'npy/projection_matrix.txt'
M_proj = np.loadtxt(txt_path, dtype=np.float32)

class ARKitDataset(BaseDataset):
    def __init__(self, opt):
        print(f'Load ARKitFace')
        self.data_root = data_root
        self.opt = opt
        self.physical_bbox_size = 0.2
        self.is_train = opt.isTrain
        self.img_size = opt.img_size
        self.n_pts = opt.n_pts
        if self.is_train:
            self.df = pd.read_csv(opt.csv_path_train, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=2721 if opt.debug else None)

            # data sampling
            sampling_rate = 1
            print(f'ARKitFace train data sampling rate: {sampling_rate}')
            self.df = self.df[::sampling_rate] # sampling
            self.df = self.df.reset_index(drop=True)

            self.rnd=np.random.permutation(len(self.df))

        else:
            self.df = pd.read_csv(opt.csv_path_test, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=1394 if opt.debug else None)

        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])
        self.tfm_train = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])
        self.tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        self.kpt_ind = kpt_ind
        self.dst_pts = np.float32([
            [0, 0],
            [0, opt.img_size - 1],
            [opt.img_size - 1, 0]
        ])
        self.dst_pts_mask = np.float32([
            [0, 0],
            [0, opt.img_size // 4 - 1],
            [opt.img_size // 4 - 1, 0]
        ])

        self.faces=np.load('./data/triangles.npy')
        flip_index_path = 'dataset/ARKitFace/flip_index.npy'
        self.flip_index = np.load(flip_index_path)

        # subsample matrix
        self.subsample = read_pkl('data/arkit_subsample.pkl')

        ##############################################################################################
        # rotate vtx world, roll 180
        ##############################################################################################
        roll = np.pi
        pitch = 0
        yaw = 0
        delta_rotmat = euler_to_rotation_matrix(roll, pitch, yaw)  # [3,3]
        rotmat = np.zeros([4, 4])
        rotmat[:3, :3] = delta_rotmat.T
        rotmat[3, 3] = 1
        self.rot_roll_180 = rotmat

        self.rolls = None
        self.pitchs = None
        self.yaws = None
        ################################################################################
        # Use mask patch
        ################################################################################
        # self.do_mask_patch = opt.do_mask_patch
        # self.do_mask_patch = False
        self.mask_prob = 0.15
        self.mask_size = self.img_size // 5
        self.rot_prob = 0.5
        self.rot_aug_angle = 30 # degree
        self.flip_prob = 0.5
        if self.mask_prob > 0:
            print(f'self.mask_size: {self.mask_size}')
        if self.rot_prob > 0:
            print(f'self.rot_aug_angle: {self.rot_aug_angle}')
        # print(f'self.do_mask_patch: {self.do_mask_patch}, self.mask_size: {self.mask_size}')

    def get_item(self, index):

        if self.is_train:
            split = 'train'
        else:
            split = 'test'
        subject_id = str(self.df['subject_id'][index])
        facial_action = str(self.df['facial_action'][index])
        img_id = str(self.df['img_id'][index])
        img_path = os.path.join(self.data_root, f'ARKitFace_image_{split}', 'image', subject_id, facial_action, f'{img_id}_ar.jpg')
        npz_path = os.path.join(self.data_root, f'ARKitFace_info_{split}', 'info', subject_id, facial_action, f'{img_id}_info.npz')
        img_path = str(img_path)

        ##############################################################################################
        # Image Augmentation Config
        ##############################################################################################
        if self.is_train:
            do_rot = True if np.random.random_sample() < self.rot_prob else False
            do_mask_patch = True if np.random.random_sample() < self.mask_prob else False
            do_flip = True if np.random.random_sample() < self.flip_prob else False
        else:
            do_rot = False
            do_mask_patch = False
            do_flip = False

        npz_path = str(npz_path) # dataset/ARKitFace/ARKitFace_info_train/info/225579/09_LeftToRight_Frown/1616212233525_info.npz
        M = np.load(npz_path)
        ###3d model and mean shape
        model = M['verts'].copy()
        R_t = M['R_t'].copy()

        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = img_raw.shape
        ###render for coord and mask
        ones = np.ones([model.shape[0], 1])
        # verts_homo = np.concatenate([model, ones], axis=1) # [1220,4]

        ##############################################################################################
        # ARKitFace format --> BIWI format
        ##############################################################################################
        vtx_world_homo = np.concatenate([model, ones], axis=1)  # [1220,4]
        # rotate world vertices
        vtx_world_homo = vtx_world_homo @ self.rot_roll_180 # [1220,4]
        # Rotation
        # R_t = M['R_t'].copy()
        yaw, pitch, roll = rotation_matrix_to_euler_angles(R_t[:3, :3].T)
        rotmat_transformed = euler_to_rotation_matrix(roll, -pitch, -yaw)
        R_t[:3, :3] = rotmat_transformed.T
        # translation
        R_t[3, 1] = -R_t[3, 1]  # ty 부호 전환
        R_t[3, 2] = -R_t[3, 2]  # tz 부호 전환

        ##############################################################################################
        # Flip augmentation
        ##############################################################################################
        if do_flip:
            # flip world space coordinates
            vtx_world_homo[:, 0] = -vtx_world_homo[:, 0]
            vtx_world_homo = vtx_world_homo[self.flip_index, :] # [1220,3]

            # flip head rotation
            head_rotmat_tmp = R_t[:3, :3].T
            yaw, pitch, roll = Rotation.from_matrix(head_rotmat_tmp).as_euler('yxz', degrees=False)
            head_rot = euler_to_rotation_matrix(pitch, -yaw, -roll)  # [3,3]
            R_t_flip = R_t.copy()
            R_t_flip[:3, :3] = head_rot.T  # [4,4]

            # flip head translation
            R_t_flip[3, 0] = -R_t_flip[3, 0]
            R_t = R_t_flip.copy()

            # flip uncropped image
            img_raw = img_raw[:, ::-1, :]  # [h,w,c]

        ##############################################################################################
        # world space -> cam space
        ##############################################################################################
        # get camera coordi
        vtx_cam_homo = vtx_world_homo @ R_t # [1220,4]

        # availability 3d annot
        has_3d_vtx = np.ones(1, dtype=np.float32)
        ##############################################################################################
        # Get intrinsic, and full-size image coordi
        ##############################################################################################
        M1 = np.array([
            [img_w / 2, 0, 0, 0],
            [0, img_h / 2, 0, 0],
            [0, 0, 1, 0],
            [img_w / 2, img_h / 2, 0, 1]
        ])

        K_org = M_proj @ M1 # [4,4]
        # K_img: [4,3]
        K_img = np.array([
            [K_org[0, 0], 0, 0],
            [0, K_org[1, 1], 0],
            [-K_org[2, 0], -K_org[2, 1], 1],
            [0, 0, 0],
        ])

        # get full-size image coordi
        image_vertices = vtx_cam_homo @ K_img # [1220,3]
        image_vertices[:, :2] = image_vertices[:, :2] / image_vertices[:, [2]] # divide by z

        ##############################################################################################
        # Get Original Bounding Box
        ##############################################################################################
        roi_cnt = 0.5 * (
                    np.max(image_vertices[self.kpt_ind, :], 0)[0:2] + np.min(image_vertices[self.kpt_ind, :], 0)[0:2])
        size = np.max(np.max(image_vertices[self.kpt_ind, :], 0)[0:2] - np.min(image_vertices[self.kpt_ind, :], 0)[0:2])
        x_center = roi_cnt[0]
        y_center = roi_cnt[1]
        ss = np.array([0.75, 0.75, 0.75, 0.75])

        # center
        if self.is_train:
            rnd_size = 0.15 * size # 0.15
            # rnd_size = 0.25 * size # 0.15
            dx = np.random.uniform(-rnd_size, rnd_size)
            dy = np.random.uniform(-rnd_size, rnd_size)
            x_center, y_center = x_center + dx, y_center + dy
            ss *= np.random.uniform(0.95, 1.05)
            # ss *= np.random.uniform(0.85, 1.25)

        left = int(x_center - ss[0] * size)
        right = int(x_center + ss[1] * size)
        top = int(y_center - ss[2] * size)
        bottom = int(y_center + ss[3] * size)

        ##################################################################
        # Rotate augmentation
        ##################################################################
        if do_rot:
            # Get head center image coordinates
            head_center_cam = R_t[3, :3]  # [3]
            head_center_img = head_center_cam[None, :] @ K_img[:3, :3]  # [1,3]
            head_center_img = head_center_img / head_center_img[:, [2]]  # divide by depth
            head_center_img = head_center_img[:, :2]  # slicing [1,2]

            # Align bbox center to head center
            bbox_size = right - left
            bbox_center = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
            bbox_offset = head_center_img - bbox_center[None, :]
            bbox_offset = bbox_offset[0] # [x_offset, y_offset]

            left = left + bbox_offset[0]
            top = top + bbox_offset[1]
            right = right + bbox_offset[0]
            bottom = bottom + bbox_offset[1]

            # Get affine transform matrix
            rot_aug_angle = np.random.uniform(-self.rot_aug_angle, self.rot_aug_angle)
            tform = gen_trans_from_patch_cv((left + right) / 2, (top + bottom) / 2, bbox_size, bbox_size, self.img_size,
                                            self.img_size, 1, rot_aug_angle, inv=False)
            tform_inv = gen_trans_from_patch_cv((left + right) / 2, (top + bottom) / 2, bbox_size, bbox_size, self.img_size,
                                            self.img_size, 1, rot_aug_angle, inv=True)

            # Update head rotation
            rotmat_aug = rotation_matrix_z(np.deg2rad(-rot_aug_angle)).astype(np.float32)
            # R_t_rot = R_t.copy()
            head_rot = R_t[:3, :3].T  # [3,3]
            head_rot_aug = np.dot(rotmat_aug, head_rot)

            R_t[:3, :3] = head_rot_aug.T  # [3,3]

            # Update verts cam coord
            vtx_cam_homo = vtx_world_homo @ R_t # [1220,4]

            # Update verts img coord
            image_vertices = vtx_cam_homo @ K_img  # [1220,3]
            image_vertices[:, :2] = image_vertices[:, :2] / image_vertices[:, [2]]  # divide by z

        else:
            ##############################################################################################
            # Get transform matrix: Full size image -> cropped image
            ##############################################################################################
            src_pts = np.float32([
                [left, top],
                [left, bottom],
                [right, top]
            ])

            tform = cv2.getAffineTransform(src_pts, self.dst_pts)
            tform_inv = cv2.getAffineTransform(self.dst_pts, src_pts)

        ##############################################################################################
        # Get 2D coordinate of 68 landmark on heatmap scale
        ##############################################################################################
        # get lmk
        lmk68 = image_vertices[self.kpt_ind, :2]  # [1220,3] -> [68,2]
        lmk68_full = lmk68.copy()

        # crop bbox
        lmk68_crop = lmk68  # [B, nv, 2]
        lmk68_crop[:, 0] = lmk68[:, 0] - left  # x
        lmk68_crop[:, 1] = lmk68[:, 1] - top  # y

        # resize img
        bbox_size = right - left
        resize = (self.img_size / bbox_size)  # [B]
        lmk68_crop = lmk68_crop * resize
        lmk68_crop_norm = lmk68_crop / self.img_size * 2 - 1 # [0~191]->[-1~1]
        lmk68_crop_homo = np.ones([lmk68_crop_norm.shape[0], lmk68_crop_norm.shape[1]+1], dtype=np.float32)
        lmk68_crop_homo[:,:2] = lmk68_crop_norm

        ##############################################################################################
        # Original image -> cropped image
        ##############################################################################################
        # tform = cv2.getAffineTransform(src_pts, self.dst_pts)
        img = cv2.warpAffine(img_raw, tform, (self.img_size,) * 2)

        # do_mask_patch = True
        if do_mask_patch:
            patch_size = self.mask_size
            patch_x = int(np.random.uniform(0, self.img_size-patch_size-1))
            patch_y = int(np.random.uniform(0, self.img_size - patch_size - 1))

            random_color = np.random.randint(0, 256, size=3)

            img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size,:] = random_color

        if self.is_train:
            img_cropped_raw = img.copy()
            img = self.tfm_train(Image.fromarray(img))
            # img_raw = self.tfm_train(Image.fromarray(img_raw))
        else:
            img_cropped_raw = img.copy()
            img = self.tfm_test(Image.fromarray(img))
            # img_raw = self.tfm_test(Image.fromarray(img_raw))

        ##############################################################################################
        # Build bbox info for original data
        ##############################################################################################
        focal_length = K_img[0, 0].copy()
        bbox_info = np.array([left, top, right, bottom, focal_length, img_h, img_w], dtype=np.float32)

        ##############################################################################################
        # Update Intrinsics
        ##############################################################################################
        bbox_size = right - left

        updated_K = K_img.copy()
        K_tmp = update_after_crop(K_img[:3, :3].T, bbox_info[:4]) # [3,3]
        K_tmp = update_after_resize(K_tmp, [bbox_size, bbox_size], [self.img_size, self.img_size]) # [3,3]
        updated_K[:3, :3] = K_tmp.T

        ##############################################################################################
        # GT vtx img
        ##############################################################################################
        vtx_img_homo = image_vertices.copy() # [1220,3]
        vtx_img_homo[:,-1] = 1

        #############################################################################
        # Landmark 2D
        #############################################################################
        gt_lmk68_full = image_vertices[self.kpt_ind, :2]
        gt_lmk68_crop = cv2.transform(gt_lmk68_full[None, :, :], tform)[0]  # [68,2], [0~191]
        ones = np.ones([len(gt_lmk68_crop), 1], dtype=gt_lmk68_crop.dtype) # [68,1]
        gt_lmk68_crop = np.concatenate([gt_lmk68_crop, ones], axis=1) # [68,3]

        # end loading my data
        sample = {
            'img': img,
            'img_raw': img_cropped_raw,
            'img_path': str(img_path),
            'lmk68_full': lmk68_full,
            'lmk68_crop': lmk68_crop_homo,
            'R_t': R_t,
            'K': K_img.astype(np.float32),
            'vtx_img': vtx_img_homo.astype(np.float32),
            'vtx_cam': vtx_cam_homo.astype(np.float32),
            'vtx_world': vtx_world_homo.astype(np.float32),  # [1220,4]
            'has_3d_vtx': has_3d_vtx, # True
            'has_rot': np.ones(1, dtype=np.float32), # True
            'bbox_info': bbox_info, # [left, top, right, bottom, focal_length, img_h, img_w]
            'dataset': 'arkit',
            'K_crop': updated_K.astype(np.float32),
        }

        # if hasattr(self.opt, 'eval'):
        #     sample['subjectid'] = str(subject_id)
        #     sample['imgid'] = str(img_id)
        #     sample['facial_action'] = str(facial_action)

        return sample

    def __getitem__(self, idx):
        data = self.get_item(idx)
        while data is None:
            idx = np.random.randint(0, len(self.df))
            data = self.get_item(idx)
        return data

    def __len__(self):
        return len(self.df)

    def compute_metrics_(self, inference_data):
        d = {}
        d['3DRecon_mean'] = np.mean(inference_data['3DRecon'])
        d['3DRecon_median'] = np.median(inference_data['3DRecon'])
        d['face_size_error'] = np.mean(inference_data['face_size_error'])
        d['ADD'] = np.mean(inference_data['ADD'])
        d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
        d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
        d['roll_mae'] = np.mean(inference_data['roll_mae'])
        d['tx_mae'] = np.mean(inference_data['tx_mae'])
        d['ty_mae'] = np.mean(inference_data['ty_mae'])
        d['tz_mae'] = np.mean(inference_data['tz_mae'])

        try:
            d['ge_err'] = np.mean(inference_data['ge_err'])
            d['ge_err'] = '%.2f °' % d['ge_err']
        except KeyError:
            pass

        d['mae_r'] = '%.2f °' % ((d['roll_mae'] + d['yaw_mae'] + d['pitch_mae']) / 3)
        d['mae_t'] = '%.2f mm' % ((d['tz_mae'] + d['tx_mae'] + d['ty_mae']) / 3 * 1000)
        d['3DRecon_mean'] = '%.2f mm' % (d['3DRecon_mean'] * 1000)
        d['3DRecon_median'] = '%.2f mm' % (d['3DRecon_median'] * 1000)
        d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
        d['pitch_mae'] = '%.2f °' % d['pitch_mae']
        d['yaw_mae'] = '%.2f °' % d['yaw_mae']
        d['roll_mae'] = '%.2f °' % d['roll_mae']
        d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
        d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
        d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)

        d['face_size_error'] = '%.4f mm^2' % (d['face_size_error'])
        return d
