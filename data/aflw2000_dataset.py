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
from data.preprocessing import update_after_crop, update_after_resize
from models.op import euler_to_rotation_matrix

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

def get_lmk_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    lmk2d = mat['pt2d'].T # [21,2]
    lmk3d_68 = mat['pt3d_68'].T # [68,3]
    return lmk2d, lmk3d_68

def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    print("dataset path is:", file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

class AFLW2000Dataset(BaseDataset):
    # Head pose from 300W-LP dataset
    def __init__(self, opt):

        print(f'Load AFLW2000-3D dataset')
        img_mean = np.array([0.485, 0.456, 0.406])
        img_std = np.array([0.229, 0.224, 0.225])

        self.tfm_train = transforms.Compose([
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std)
        ])

        self.data_dir = './dataset/AFLW2000'
        self.img_ext = '.jpg'
        self.annot_ext = '.mat'
        annot_file_path = os.path.join(self.data_dir, 'files.txt')
        filename_list = get_list_from_filenames(annot_file_path)

        self.X_train = filename_list
        self.y_train = filename_list
        # self.image_mode = image_mode
        self.length = len(filename_list)

        self.debug = True

        self.img_size = opt.img_size
        self.dst_pts = np.float32([
            [0, 0],
            [0, self.img_size - 1],
            [self.img_size - 1, 0]
        ])

        self.is_train = False

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.X_train[index] + self.img_ext)
        img_raw = cv2.imread(img_path)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_full = img_raw.copy()
        img_h, img_w, _ = img_raw.shape
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        # landmark_dir_path = os.path.join(self.data_dir, 'landmarks')

        ############################################################################
        # Get Landmark
        ############################################################################
        # lmk = get_pt2d_from_mat(mat_path)
        # lmk = lmk.T
        lmk2d, lmk = get_lmk_from_mat(mat_path)

        ############################################################################
        # Validation AFLW
        ############################################################################
        valid = (lmk[:, 0] > 0) & (lmk[:, 0] < img_w)

        ############################################################################
        # Get bbox
        ############################################################################
        lmk_tmp = lmk[valid, :].copy()
        roi_cnt = 0.5 * (np.max(lmk_tmp, 0)[0:2] + np.min(lmk_tmp, 0)[0:2])
        size = np.max(np.max(lmk_tmp, 0)[0:2] - np.min(lmk_tmp, 0)[0:2])
        x_center = roi_cnt[0]
        y_center = roi_cnt[1]
        ss = np.array([0.75, 0.75, 0.75, 0.75])
        # ss = np.array([0.5, 0.5, 0.5, 0.5])

        left = int(x_center - ss[0] * size)
        right = int(x_center + ss[1] * size)
        top = int(y_center - ss[2] * size)
        bottom = int(y_center + ss[3] * size)

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

        ############################################################################
        # Transform landmark 2d coordinates
        ############################################################################
        lmk_full = lmk.astype(np.float32)
        lmk_crop = cv2.transform(lmk[None, :, :], tform)[0] # [68,2], [0 ~ img_size-1]

        ############################################################################
        # get head pose
        ############################################################################
        # We get the pose in radians
        pose = get_ypr_from_mat(mat_path)
        # And convert to degrees.
        x_rot = pose[0] # * 180 / np.pi
        y_rot = pose[1] #* 180 / np.pi
        z_rot = pose[2] # * 180 / np.pi
        # roll = -roll

        ############################################################################
        # euler -> rotmat
        ############################################################################
        head_rot = euler_to_rotation_matrix(x_rot, y_rot, z_rot)

        ############################################################################
        # Convert AFLW2000 -> BIWI
        ############################################################################
        # yaw, pitch, roll = Rotation.from_matrix(head_rot).as_euler('yxz', degrees=False)
        # head_rot = euler_to_rotation_matrix(-pitch, yaw, roll)  # [3,3]

        ############################################################################
        # Get euler angle
        ############################################################################
        # yaw, pitch, roll = Rotation.from_matrix(head_rot).as_euler('yxz', degrees=True)

        # labels = torch.FloatTensor([x_rot, y_rot, z_rot])

        ############################################################################
        # img: np -> tensor
        ############################################################################
        img_org = img.copy()
        img = self.tfm_train(Image.fromarray(img))

        ############################################################################
        # Valid: Truncation
        ############################################################################
        trunc_valid_x = (lmk_crop[:, 0] >= 0) & (lmk_crop[:, 0] < img_org.shape[1])
        trunc_valid_y = (lmk_crop[:, 1] >= 0) & (lmk_crop[:, 1] < img_org.shape[0])
        trunc_valid = trunc_valid_x & trunc_valid_y  # image 안에 들어오면 true, image 밖으로 나가면 False
        # trunc_validation = trunc_valid.copy()
        trunc_valid = trunc_valid.astype(np.float32)
        lmk_crop_norm = lmk_crop / self.img_size * 2 - 1 # [0~191]->[-1~1]
        lmk_crop_norm = np.concatenate([lmk_crop_norm, trunc_valid[:, None]], axis=1) # [68,2]->[68,3]

        ############################################################################
        # dummy: vtx, R_t
        ############################################################################

        vtx_img = np.zeros([1220,3], dtype=np.float32)
        vtx_cam = np.zeros([1220,4], dtype=np.float32)
        vtx_world = np.zeros([1220,4], dtype=np.float32)
        has_3d_vtx = np.zeros(1, dtype=np.float32)

        # R_t
        R_t = np.zeros([4,4], dtype=np.float32)
        R_t[:3,:3] = head_rot.T
        R_t[3,3] = 1
        R_t[3, :3] = np.array([0,0,5], dtype=np.float32)
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
            # 'lmk68_full': lmk_full,
            # 'lmk68_crop': lmk_crop_norm,
            'R_t': R_t,
            'K': K_img.astype(np.float32),
            'vtx_img': vtx_img,
            'vtx_cam': vtx_cam,
            'vtx_world': vtx_world, # [1220,4]
            'has_3d_vtx': has_3d_vtx, # False
            'bbox_info': bbox_info, # [left, top, right, bottom, focal_length, img_h, img_w]
            'dataset': 'aflw',
            'K_crop': updated_K.astype(np.float32),
        }

        # sample['lmk3d'] = lmk3d
        return sample

    def __len__(self):
        # 1969
        return self.length


    def compute_metrics_(self, inference_data):
        """
         modified by jsh
        """
        d = {}
        d['3DRecon'] = np.mean(inference_data['3DRecon'])
        d['ADD'] = np.mean(inference_data['ADD'])
        d['pitch_mae'] = np.mean(inference_data['pitch_mae'])
        d['yaw_mae'] = np.mean(inference_data['yaw_mae'])
        d['roll_mae'] = np.mean(inference_data['roll_mae'])
        d['tx_mae'] = np.mean(inference_data['tx_mae'])
        d['ty_mae'] = np.mean(inference_data['ty_mae'])
        d['tz_mae'] = np.mean(inference_data['tz_mae'])

        try:
            d['angular_distance'] = np.mean(inference_data['angular_distance'])
            d['angular_distance'] = '%.2f °' % d['angular_distance']
        except KeyError:
            pass

        d['mae_r'] = '%.2f °' % ((d['roll_mae'] + d['yaw_mae'] + d['pitch_mae']) / 3)
        d['mae_t'] = '%.2f mm' % ((d['tz_mae'] + d['tx_mae'] + d['ty_mae']) / 3 * 1000)
        d['3DRecon'] = '%.2f mm' % (d['3DRecon'] * 1000)
        d['ADD'] = '%.2f mm' % (d['ADD'] * 1000)
        d['pitch_mae'] = '%.2f °' % d['pitch_mae']
        d['yaw_mae'] = '%.2f °' % d['yaw_mae']
        d['roll_mae'] = '%.2f °' % d['roll_mae']
        d['tx_mae'] = '%.2f mm' % (d['tx_mae'] * 1000)
        d['ty_mae'] = '%.2f mm' % (d['ty_mae'] * 1000)
        d['tz_mae'] = '%.2f mm' % (d['tz_mae'] * 1000)

        return d