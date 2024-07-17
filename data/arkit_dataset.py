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
from data.augmentation import EulerAugmentor, HorizontalFlipAugmentor
from util.util import landmarks106to68
from lib import mesh
from lib import mesh_p
from lib.eyemouth_index import vert_index, face_em
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
        self.debug = True

        print(f'Load ARKitFace')
        self.data_root = data_root
        self.opt = opt
        self.physical_bbox_size = 0.2
        self.is_train = opt.isTrain
        self.img_size = opt.img_size
        self.n_pts = opt.n_pts
        self.do_random_sample = False
        self.do_uniform_sample = False
        if self.is_train:
            # 둘 중에 하나만 true 가 나와야 한다.
            # self.do_random_sample = opt.random_sample
            # self.do_uniform_sample = opt.uniform_sample
            print(f'self.do_random_sample: {self.do_random_sample}')
            print(f'self.do_uniform_sample: {self.do_uniform_sample}')
            if self.do_random_sample and self.do_uniform_sample:
                assert False
        if self.is_train:
            self.df = pd.read_csv(opt.csv_path_train, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=2721 if opt.debug else None)
            self.rnd=np.random.permutation(len(self.df))
        else:
            self.df = pd.read_csv(opt.csv_path_test, dtype={'subject_id': str, 'facial_action': str, 'img_id': str},
                nrows=1394 if opt.debug else None)

        #################################################################
        # Data augmentation config
        #################################################################
        self.mask_prob = 0.15
        self.mask_size = int(self.img_size / 3.5)
        self.rot_prob = 0.5
        self.rot_aug_angle = 35  # degree
        self.flip_prob = 0.5  # 0.5
        self.blur_prob = -1
        if self.mask_prob > 0:
            print(f'self.mask_size: {self.mask_size}')
        if self.rot_prob > 0:
            print(f'self.rot_aug_angle: {self.rot_aug_angle}')
        # print(f'self.do_mask_patch: {self.do_mask_patch}, self.mask_size: {self.mask_size}')

        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])

        self.tfm_train = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1.0))(x) if random.random() < self.blur_prob else x),
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

        ############ uv map ############
        self.uv_size = self.img_size
        uv_coords = uv_coords_std * (self.uv_size - 1)
        zeros = np.zeros([uv_coords.shape[0], 1])
        self.uv_coords_extend = np.concatenate([uv_coords, zeros], axis=1)
        self.tris_full = tris_full
        self.contour_ind = [
                20, 853, 783, 580, 659, 660, 765, 661, 616, 579,
                489, 888, 966, 807, 730, 1213, 1214, 1215, 1216, 822,
                906, 907, 908, 909, 910, 911, 912, 913, 1047, 914,
                915, 916, 917, 918, 919, 920, 921, 392, 462, 905,
                904, 208, 295, 376, 57, 467, 39, 130, 167, 213,
                330, 212, 211, 131, 352, 425
                ]
        self.mouth_ind = [24, 691, 690, 689, 688, 687, 686, 685, 823, 684, 834, 740, 683, 682, 710, 725, 709, 700,
                25, 265, 274, 290, 275, 247, 248, 305, 404, 249, 393, 250, 251, 252, 253, 254, 255, 256]
        self.eye1_ind = [1101, 1100, 1099, 1098, 1097, 1096, 1095, 1094, 1093, 1092, 1091, 1090,
                1089, 1088, 1087, 1086, 1085, 1108, 1107, 1106, 1105, 1104, 1103, 1102]
        self.eye2_ind = [1081, 1080, 1079, 1078, 1077, 1076, 1075, 1074, 1073, 1072, 1071, 1070,
                1069, 1068, 1067, 1066, 1065, 1064, 1063, 1062, 1061, 1084, 1083, 1082]
        self.tris_mask = tris_mask

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

        ##############################################################################################
        # Load arkit train R_t
        ##############################################################################################
        self.fan_pred = None
        if self.do_uniform_sample:
            # load R t
            arkit_rt_path = './dataset/ARKitFace/train_R_t.npz'
            arkit_rt_data = np.load(arkit_rt_path)
            arkit_rt = arkit_rt_data['R_t'] # [T,4,4], np
            arkit_rt = torch.FloatTensor(arkit_rt) # tensor
            arkit_R = (arkit_rt[:, :3, :3].transpose(1,2)).cpu().numpy()  # [T,3,3]

            # Rotmat -> Euler
            roll, pitch, yaw = self.get_euler_angle_from_rotmat(arkit_R) # [T], [T], [T]
            self.rolls = roll
            self.pitchs = pitch
            self.yaws = yaw

            # Load FAN prediction
            fan_path = 'dataset/ARKitFace/arkit_fan_kpt_error.pkl'
            self.fan_pred = read_pkl(fan_path)

        else:
            self.rolls = None
            self.pitchs = None
            self.yaws = None

        ################################################################################
        # For TTA
        ################################################################################
        if self.is_train:
            fan_path = "./dataset/ARKitFace/fan_kpt_arkit_traindata.pkl"
            self.annot = read_pkl(fan_path)
        else:
            fan_path = "./dataset/ARKitFace/fan_kpt_arkit_testdata.pkl"
            self.annot = read_pkl(fan_path)

    def get_euler_angle_from_rotmat(self, rotmat):
        '''
        rotmat: [T,3,3], np.ndarray

        return
        rolls: [T], np.ndarray
        pitchs: [T], np.ndarray
        yaws: [T], np.ndarray
        '''
        total_frame = len(rotmat)
        yaws, pitchs, rolls = [], [], []
        for frame_i in range(total_frame):
            y, p, r = rotation_matrix_to_euler_angles(rotmat[frame_i])  # np
            yaws.append(y)
            pitchs.append(p)
            rolls.append(r)

        yaws = radian2degree(np.stack(yaws))
        pitchs = radian2degree(np.stack(pitchs))
        rolls = radian2degree(np.stack(rolls))
        # angles = [rolls, pitchs, yaws]
        return rolls, pitchs, yaws


    def generate_uv_position_map(self, verts):
        temp1 = verts[self.contour_ind] * 1.1
        temp2 = verts[self.mouth_ind].mean(axis=0, keepdims=True)
        temp3 = verts[self.eye1_ind].mean(axis=0, keepdims=True)
        temp4 = verts[self.eye2_ind].mean(axis=0, keepdims=True)
        verts_ = np.vstack([verts, temp1, temp2, temp3, temp4])  # (1279, 3)
        uv_map = mesh.render.render_colors(self.uv_coords_extend, self.tris_full, verts_, h=self.uv_size, w=self.uv_size, c=3)   #[0, 1]
        uv_map = np.clip(uv_map, 0, 1)
        return uv_map

    def get_item(self, index):

        if self.is_train:
            if self.do_random_sample:
                index = self.rnd[index]
            split = 'train'

        else:
            split = 'test'
        subject_id = str(self.df['subject_id'][index])
        facial_action = str(self.df['facial_action'][index])
        img_id = str(self.df['img_id'][index])
        img_path = os.path.join(self.data_root, f'ARKitFace_image_{split}', 'image', subject_id, facial_action, f'{img_id}_ar.jpg')
        npz_path = os.path.join(self.data_root, f'ARKitFace_info_{split}', 'info', subject_id, facial_action, f'{img_id}_info.npz')
        img_path = str(img_path)

        fan_key = os.path.join(f'ARKitFace_image_{split}', 'image', subject_id, facial_action, f'{img_id}_ar.jpg')
        fan_key = str(fan_key)

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

        # 기존 intrinsic 이 이상했다.
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
        resize = resize
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
            # patch_size = int(np.random.uniform(self.mask_size//2, self.mask_size))
            patch_x = int(np.random.uniform(0, self.img_size-patch_size-1))
            patch_y = int(np.random.uniform(0, self.img_size - patch_size - 1))

            img[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size,:] = 0

        if self.is_train:
            img_cropped_raw = img.copy()
            img = self.tfm_train(Image.fromarray(img))
            # img_raw = self.tfm_train(Image.fromarray(img_raw))
        else:
            img_cropped_raw = img.copy()
            img = self.tfm_test(Image.fromarray(img))
            # img_raw = self.tfm_test(Image.fromarray(img_raw))

        # ##############################################################################################
        # # Get inverse affine transform
        # ##############################################################################################
        # # tform_inv
        # dst_pts2 = np.float32([
        #     [0, 0],
        #     [0, self.img_size * 2 - 1],
        #     [self.img_size * 2 - 1, 0]
        # ])
        # tform_inv = cv2.getAffineTransform(dst_pts2, src_pts)
        # W_inv = tform_inv.T[:2, :2].astype(np.float32)
        # b_inv = tform_inv.T[2].astype(np.float32)

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

        if self.debug:
            # inverse transform
            # tform_inv = cv2.getAffineTransform(self.dst_pts, src_pts)
            sample['tform_inv'] = tform_inv
            sample['tform'] = tform
            # FAN
            # sample['pred_lmk68_full_homo'] = pred_lmk68_full_homo # [68,3]
            # sample['pred_lmk68_crop_norm'] = pred_lmk68_crop_norm # [68,2], [-1~1]
            # sample['updated_K'] = updated_K # [68,2], [-1~1]

        if hasattr(self.opt, 'eval'):
            sample['subjectid'] = str(subject_id)
            sample['imgid'] = str(img_id)
            sample['facial_action'] = str(facial_action)

        return sample

    def __getitem__(self, idx):
        if self.do_uniform_sample and False:
            data = None
            # uniform sample, Fine-tune setting
            while data is None:
                random = np.random.uniform(0, 3)
                if random < 1:
                    # roll
                    sampled_index, _ = uniform_sampled_data(self.rolls, num_sample=1, percentage=0.8)
                elif random < 2:
                    # pitch
                    sampled_index, _ = uniform_sampled_data(self.pitchs, num_sample=1, percentage=0.8)
                else:
                    # yaw
                    sampled_index, _ = uniform_sampled_data(self.yaws, num_sample=1, percentage=0.8)

                idx = sampled_index.item()
                data = self.get_item(idx)
        else:
            data = self.get_item(idx)
            while data is None:
                idx = np.random.randint(0, len(self.df))
                data = self.get_item(idx)
        return data

    def __len__(self):
        return len(self.df)

    def compute_metrics_(self, inference_data):
        """
         modified by jsh
        """
        d = {}
        d['3DRecon'] = np.mean(inference_data['3DRecon'])
        d['3DRecon_median'] = np.median(inference_data['3DRecon'])
        # d['3DRecon_median'] = np.mean(inference_data['3DRecon_median'])
        # print(np.std(inference_data['3DRecon']))
        # print(np.median(inference_data['3DRecon']))
        d['ADD'] = np.mean(inference_data['ADD'])
        d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
        d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
        d['roll_mae'] = np.mean(inference_data['roll_mae'])
        d['tx_mae'] = np.mean(inference_data['tx_mae'])
        d['ty_mae'] = np.mean(inference_data['ty_mae'])
        d['tz_mae'] = np.mean(inference_data['tz_mae'])
        # d['5°5cm'] = inference_data['strict_success'] / inference_data['total_count']
        # d['5°10cm'] = inference_data['easy_success'] / inference_data['total_count']
        # d['mean_IoU'] = np.mean(inference_data['IoU'])

        try:
            d['angular_distance'] = np.mean(inference_data['angular_distance'])
            d['angular_distance'] = '%.2f °' % d['angular_distance']
        except KeyError:
            pass

        try:
            d['area_mean'] = inference_data['area_mean']
            d['area_std'] = inference_data['area_std']
        except KeyError:
            pass

        d['mae_r'] = '%.2f °' % ((d['roll_mae'] + d['yaw_mae'] + d['pitch_mae']) / 3)
        d['mae_t'] = '%.2f mm' % ((d['tz_mae'] + d['tx_mae'] + d['ty_mae']) / 3 * 1000)
        d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
        d['3DRecon_median'] = '%.2f mm' % (d['3DRecon_median'] * 1000)
        d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
        d['pitch_mae'] = '%.2f °' % d['pitch_mae']
        d['yaw_mae'] = '%.2f °' % d['yaw_mae']
        d['roll_mae'] = '%.2f °' % d['roll_mae']
        d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
        d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
        d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)
        # d['tz_duli_mae'] = '%.2f mm' % (d['tz_duli_mae'] * 1000)
        # d['5°5cm'] = '%.2f ' % (d['5°5cm'] * 100)
        # d['5°10cm'] = '%.2f' % (d['5°10cm'] * 100)
        # d['mean_IoU'] = '%.4f' % d['mean_IoU']

        return d

    def compute_metrics(self, inference_data):
        bs_list = np.array(inference_data['batch_size'])
        loss1 = np.array(inference_data['loss_total'])
        loss2 = np.array(inference_data['loss_corr'])
        loss3 = np.array(inference_data['loss_recon3d'])
        loss4 = np.array(inference_data['loss_uv'])
        loss5 = np.array(inference_data['loss_mat'])
        loss6 = np.array(inference_data['loss_seg'])
       
        loss_total = (loss1 * bs_list).sum() / bs_list.sum() 
        loss_corr = (loss2 * bs_list).sum() / bs_list.sum() 
        loss_recon3d = (loss3 * bs_list).sum() / bs_list.sum()
        loss_uv = (loss4 * bs_list).sum() / bs_list.sum()
        loss_mat = (loss5 * bs_list).sum() / bs_list.sum()
        loss_seg = (loss6 * bs_list).sum() / bs_list.sum()
        
        d = {
            'loss_total': loss_total,
            'loss_corr': loss_corr,
            'loss_recon3d': loss_recon3d,
            'loss_uv': loss_uv,
            'loss_mat': loss_mat,
            'loss_seg': loss_seg,
        }
        if hasattr(self.opt, 'eval'):
            d['pnp_fail'] = inference_data['pnp_fail']
            d['3DRecon'] = np.mean(inference_data['3DRecon'])
            # print(np.std(inference_data['3DRecon']))
            # print(np.median(inference_data['3DRecon']))
            d['ADD'] = np.mean(inference_data['ADD'])
            d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
            d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
            d['roll_mae'] = np.mean(inference_data['roll_mae'])
            d['tx_mae'] = np.mean(inference_data['tx_mae'])
            d['ty_mae'] = np.mean(inference_data['ty_mae'])
            d['tz_mae'] = np.mean(inference_data['tz_mae'])
            d['5°5cm'] = inference_data['strict_success'] / inference_data['total_count']
            d['5°10cm'] = inference_data['easy_success'] / inference_data['total_count']
            d['mean_IoU'] = np.mean(inference_data['IoU'])

            d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
            d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
            d['pitch_mae'] = '%.2f °' % d['pitch_mae']
            d['yaw_mae'] = '%.2f °' % d['yaw_mae']
            d['roll_mae'] = '%.2f °' % d['roll_mae']
            d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
            d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
            d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)
            #d['tz_duli_mae'] = '%.2f mm' % (d['tz_duli_mae'] * 1000)
            d['5°5cm'] = '%.2f ' % (d['5°5cm'] * 100)
            d['5°10cm'] = '%.2f' % (d['5°10cm'] * 100)
            d['mean_IoU'] = '%.4f' % d['mean_IoU']

        return d
