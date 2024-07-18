import os
import numpy as np

import torch
from torch.utils.data.dataset import Dataset

from PIL import Image, ImageFilter
import scipy.io as sio
import cv2
import torchvision.transforms as transforms

from scipy.spatial.transform import Rotation

from pdb import set_trace
from data.base_dataset import BaseDataset
from data.preprocessing import update_after_crop, update_after_resize, gen_trans_from_patch_cv
from models.op import euler_to_rotation_matrix
from util.pkl import read_pkl
import random

def rotation_matrix_z(theta):
    """
    z축을 중심으로 theta 만큼의 회전 행렬을 반환합니다.

    :param theta: 회전 각도 (라디안 단위)
    :return: 3x3 회전 행렬
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])
def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x pitch
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y yaw
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z roll
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R

def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print("dataset path is:", file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

def flip_landmarks(landmarks, flip_pair, image_width):
    """
        landmarks: numpy.ndarray, [68,2]
        flip_pair: double list, [[],[],[], ..., []]
        image_width: scalar

    """
    flipped_landmarks = landmarks.copy()
    flipped_landmarks[:, 0] = image_width - 1 - flipped_landmarks[:, 0]

    for pair in flip_pair:
        flipped_landmarks[pair[0], :], flipped_landmarks[pair[1], :] = flipped_landmarks[pair[1], :].copy(), flipped_landmarks[pair[0], :].copy()

    return flipped_landmarks

def rotate_image_and_landmarks(image, landmarks, angle):
    """
    Rotate an image and its landmarks.

    :param image: numpy.ndarray, the image to be rotated.
    :param landmarks: numpy.ndarray, the landmarks to be rotated.
    :param angle: float, rotation angle in degrees.
    :return: Rotated image and landmarks.
    """
    # 이미지의 중심점 구하기
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    # 회전 변환 행렬 구하기
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # 이미지 회전
    rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    # 랜드마크 회전
    ones = np.ones(shape=(len(landmarks), 1))
    points_ones = np.hstack([landmarks, ones])

    # 회전된 랜드마크 계산
    rotated_landmarks = rot_mat @ points_ones.T
    rotated_landmarks = rotated_landmarks.T

    return rotated_image, rotated_landmarks


class W300LPDataset(BaseDataset):
    # Head pose from 300W-LP dataset
    def __init__(self, opt):

        print(f'Load 300W-LP dataset')
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

        self.data_dir = './dataset/300W_LP'
        self.img_ext = '.jpg'
        self.annot_ext = '.mat'
        annot_file_path = os.path.join(self.data_dir, 'files.txt')
        filename_list = get_list_from_filenames(annot_file_path)

        self.X_train = filename_list
        self.y_train = filename_list
        # self.image_mode = image_mode
        self.length = len(filename_list)

        # 300W format
        flip_pair = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], [17, 26], [18, 25],
                     [19, 24],
                     [20, 23],
                     [21, 22], [31, 35], [32, 34], [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46],
                     [48, 54], [49, 53], [50, 52], [60, 64], [61, 63], [67, 65], [59, 55], [58, 56]]

        self.flip_pair = np.array(flip_pair, dtype=np.uint64)

        self.debug = True

        self.img_size = opt.img_size
        self.dst_pts = np.float32([
            [0, 0],
            [0, self.img_size - 1],
            [self.img_size - 1, 0]
        ])

        self.is_train = True

        ############################################################################
        # Augment config
        ############################################################################
        self.rot_prob = 0.5 # -1, 0.7
        self.mask_prob = 0.15 # -1, 0.7
        self.mask_size = self.img_size // 5
        self.img_crop_shift = 0.3 # 0.75: very hard
        self.rot_aug_angle = 30  # degree
        if self.mask_prob > 0:
            print(f'self.mask_size: {self.mask_size}')
        if self.rot_prob > 0:
            print(f'self.rot_aug_angle: {self.rot_aug_angle}')

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_full = img_raw.copy()
        img_h, img_w, _ = img_raw.shape
        pose_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        landmark_dir_path = os.path.join(self.data_dir, 'landmarks')

        ############################################################################
        # Augment config
        ############################################################################
        if self.is_train:
            do_rot = True if np.random.random_sample() < self.rot_prob else False
            do_mask_patch = True if np.random.random_sample() < self.mask_prob else False
        else:
            do_rot = False
            do_mask_patch = False

        ############################################################################
        # Load landmark
        ############################################################################
        img_path_split = img_path.split('/')
        sub_dir_name, file_name = img_path_split[-2], img_path_split[-1]

        sub_dir_name_real = sub_dir_name.replace('_Flip', '')

        lmk_file_name = file_name.replace('.jpg', '_pts.mat')
        lmk_file_path = os.path.join(landmark_dir_path, sub_dir_name_real, lmk_file_name)
        lmk_file = sio.loadmat(lmk_file_path)
        lmk_68 = lmk_file['pts_3d']

        ############################################################################
        # Flip landmark
        ############################################################################
        flip = False
        if '_Flip' in sub_dir_name:
            flip = True
            lmk_68 = flip_landmarks(lmk_68, self.flip_pair, img_w) # [68,2]

        lmk_68_full = lmk_68.copy()
        ############################################################################
        # Get bbox
        ############################################################################
        roi_cnt = 0.5 * (np.max(lmk_68, 0)[0:2] + np.min(lmk_68, 0)[0:2])
        size = np.max(np.max(lmk_68, 0)[0:2] - np.min(lmk_68, 0)[0:2])
        x_center = roi_cnt[0]
        y_center = roi_cnt[1]
        ss = np.array([0.75, 0.75, 0.75, 0.75])
        # ss = np.array([0.5, 0.5, 0.5, 0.5])

        # center
        if self.is_train:
            # rnd_size = 0.15 * size  # 0.15
            rnd_size = self.img_crop_shift * size
            dx = np.random.uniform(-rnd_size, rnd_size)
            dy = np.random.uniform(-rnd_size, rnd_size)
            x_center, y_center = x_center + dx, y_center + dy
            ss *= np.random.uniform(0.95, 1.05)
            # ss *= np.random.uniform(0.85, 1.25)

        left = int(x_center - ss[0] * size)
        right = int(x_center + ss[1] * size)
        top = int(y_center - ss[2] * size)
        bottom = int(y_center + ss[3] * size)

        ##############################################################################################
        # If bbox is out of image
        ##############################################################################################
        if left < 0:
            offset = -left  # offset > 0
            left = left + offset
            right = right + offset
        if right > img_w - 1:
            offset = right - (img_w - 1)  # offset > 0
            left = left - offset
            right = right - offset
        if top < 0:
            offset = -top  # offset > 0
            top = top + offset
            bottom = bottom + offset
        if bottom > img_h - 1:
            offset = bottom - (img_h - 1)  # offset > 0
            top = top - offset
            bottom = bottom - offset

        ##############################################################################################
        # Original image -> cropped image
        ##############################################################################################
        src_pts = np.float32([
            [left, top],
            [left, bottom],
            [right, top]
        ])

        tform = cv2.getAffineTransform(src_pts, self.dst_pts)
        tform_inv = cv2.getAffineTransform(self.dst_pts, src_pts)
        img = cv2.warpAffine(img_raw, tform, (self.img_size,) * 2)

        ##############################################################################################
        # Augmentation: Mask patch
        ##############################################################################################
        if do_mask_patch:
            patch_size = self.mask_size
            patch_x = int(np.random.uniform(0, self.img_size - patch_size - 1))
            patch_y = int(np.random.uniform(0, self.img_size - patch_size - 1))

            random_color = np.random.randint(0, 256, size=3)

            img[patch_x:patch_x + patch_size, patch_y:patch_y + patch_size, :] = random_color

        ############################################################################
        # Transform landmark 2d coordinates
        ############################################################################
        lmk_68_crop = cv2.transform(lmk_68[None, :, :], tform)[0] # [68,2], [0 ~ img_size-1]

        ############################################################################
        # Rotate augmentation
        ############################################################################
        if do_rot and self.is_train:
            # rotation angle
            rot_aug = np.random.uniform(-self.rot_aug_angle, self.rot_aug_angle)

            # rotate image and landmark
            img, lmk_68_crop = rotate_image_and_landmarks(img, lmk_68_crop, rot_aug)

        ############################################################################
        # img: np -> tensor
        ############################################################################
        img_org = img.copy()

        if self.is_train:
            img = self.tfm_train(Image.fromarray(img))
        else:
            img = self.tfm_test(Image.fromarray(img))

        ############################################################################
        # Valid: Truncation
        ############################################################################
        trunc_valid_x = (lmk_68_crop[:, 0] >= 0) & (lmk_68_crop[:, 0] < img_org.shape[1])
        trunc_valid_y = (lmk_68_crop[:, 1] >= 0) & (lmk_68_crop[:, 1] < img_org.shape[0])
        trunc_valid = trunc_valid_x & trunc_valid_y  # image 안에 들어오면 true, image 밖으로 나가면 False
        # trunc_validation = trunc_valid.copy()
        trunc_valid = trunc_valid.astype(np.float32)
        lmk_68_crop_norm = lmk_68_crop / self.img_size * 2 - 1 # [0~191]->[-1~1]
        lmk_68_crop_norm = np.concatenate([lmk_68_crop_norm, trunc_valid[:, None]], axis=1) # [68,2]->[68,3]

        ############################################################################
        # dummy: vtx, R_t
        ############################################################################
        vtx_img = np.zeros([1220,3], dtype=np.float32)
        vtx_cam = np.zeros([1220,4], dtype=np.float32)
        vtx_world = np.zeros([1220,4], dtype=np.float32)
        has_3d_vtx = np.zeros(1, dtype=np.float32)
        # has_rot = np.ones(1, dtype=np.float32) # apply loss
        has_rot = np.zeros(1, dtype=np.float32) # Don't apply loss

        # R_t
        R_t = np.zeros([4,4], dtype=np.float32)
        R_t[:3,:3] = np.eye(3)
        R_t[3,3] = 1
        # translation = np.zeros([3,1], dtype=np.float32)
        # R_t = np.concatenate([head_rot, translation], axis=1).T # [4,3] ?

        # K
        focal_length = 5000.0
        ppoint = [img_w / 2, img_h / 2]
        K_img = np.array([
            [focal_length, 0, 0],
            [0, focal_length, 0],
            [ppoint[0], ppoint[1], 1],
            [0, 0, 0],
        ])

        ############################################################################
        # bbox info
        ############################################################################
        bbox_info = np.array([left, top, right, bottom, focal_length, img_h, img_w], dtype=np.float32)

        ##############################################################################################
        # Update Intrinsics
        ##############################################################################################
        bbox_size = right - left
        updated_K = K_img.copy()
        K_tmp = update_after_crop(K_img[:3, :3].T, bbox_info[:4])  # [3,3]
        K_tmp = update_after_resize(K_tmp, [bbox_size, bbox_size], [self.img_size, self.img_size])  # [3,3]
        updated_K[:3, :3] = K_tmp.T

        sample = {
            'img': img,
            'img_raw': np.array(img_org),
            'img_path': img_path,
            'lmk68_full': lmk_68_full,
            'lmk68_crop': lmk_68_crop_norm,
            'R_t': R_t,
            'K': K_img.astype(np.float32),
            'vtx_img': vtx_img,
            'vtx_cam': vtx_cam,
            'vtx_world': vtx_world, # [1220,4]
            'has_3d_vtx': has_3d_vtx, # False
            'has_rot': has_rot,
            'bbox_info': bbox_info, # [left, top, right, bottom, focal_length, img_h, img_w]
            'dataset': '300wlp',
            'K_crop': updated_K.astype(np.float32),
        }

        return sample

    def __len__(self):
        # 122,450
        return self.length