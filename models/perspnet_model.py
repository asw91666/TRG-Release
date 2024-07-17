import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb
import cv2
import socket
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from . import networks
from .base_model import BaseModel
from .network import DeformNetUV2
from .loss import Loss
from util.face_deformnet_utils import compute_sRT_errors
from shutil import copyfile

np.set_printoptions(suppress=True)
class PerspnetModel(BaseModel):

    ''' 
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser
    '''


    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['total', 'corr', 'recon3d', 'uv', 'mat', 'seg']
        self.model_names = ['R']

        npy_path = 'npy/uv_coords_std_202109.npy'
        uv_coords_std = np.load(npy_path)[:1220]
        uv_coords = uv_coords_std * 2 - 1
        self.grid = torch.tensor(uv_coords).unsqueeze(0).unsqueeze(1).to(self.device)


        # networks.init_net
        self.netR = networks.init_net(
            DeformNetUV2(self.grid),
            gpu_ids=self.gpu_ids
        )
        #self.netR.cuda()
        self.criterion = Loss()
        if self.isTrain:
            self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_R)
        


    def set_input(self, input):
        self.img = input['img'].to(self.device)
        self.choose = input['choose'].to(self.device)
        self.model= input['model'].to(self.device)
        self.uvmap = input['uvmap'].to(self.device)
        self.nocs = input['nocs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.corr_mat = input['corr_mat'].to(self.device)
        
        if hasattr(self.opt, 'eval'):
            self.tform_inv_list = input['tform_inv'].numpy()
            self.R_t_list = input['R_t'].numpy() 
            self.img_path_list = input['img_path']


    def forward(self):
        #print(self.choose.max(),self.choose.min())
        self.assign_mat, self.seg_pred, self.uv_pred = self.netR(self.img, self.choose)   ### assign_mat, deltas, seg_pred
        bs = self.uv_pred.size(0)
        verts3d_pred = F.grid_sample(self.uv_pred, self.grid.expand(bs, -1, -1, -1), align_corners=False)   # 低版本pytorch没有align_corners=False这个选项
        self.verts3d_pred = verts3d_pred.squeeze(2).permute(0, 2, 1).contiguous()
             
    def forward_seg(self): 
        _, self.seg_pred, _, _=self.netR.module.cnn(self.img)


    def backward_R(self):
        self.loss_total, self.loss_corr, self.loss_recon3d, self.loss_uv, self.loss_mat, self.loss_seg  = self.criterion(self.assign_mat, self.seg_pred, self.verts3d_pred, self.uv_pred, self.nocs, self.model, self.mask, self.corr_mat,self.uvmap)
        self.loss_total.backward()




    def optimize_parameters(self):
        self.netR.train()
        self.forward()
        self.optimizer_R.zero_grad() 
        self.backward_R()
        self.optimizer_R.step()
   

    def init_evaluation(self):
        self.inference_data = {
            'loss_total': [], 'loss_corr': [], 'loss_recon3d': [], 'loss_uv':[], 'loss_mat':[], 'loss_seg':[],
            'batch_size': [],
            'pitch_mae': [], 'yaw_mae': [], 'roll_mae': [],
            'tx_mae': [], 'ty_mae': [], 'tz_mae': [],
            '3DRecon': [], 'ADD': [],
            'strict_success': 0, 'easy_success': 0, 'total_count': 0,
            'IoU': [],
            'pnp_fail': 0
        }
        self.netR.eval()

        npz_path = 'npy/kpt_ind.npy'
        self.kpt_ind = np.load(npz_path)


    def inference_curr_batch(self):
        bs = self.img.size(0)
        with torch.no_grad(): 
            if self.opt.seg:
                self.forward_seg()
                mask_pr = torch.argmax(self.seg_pred, 1).cpu().detach().numpy()
                chooses=np.zeros((bs,self.opt.n_pts))
                for i in range(bs):
                    choose = mask_pr[i].flatten().nonzero()[0]
                    if len(choose) > self.opt.n_pts:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:self.opt.n_pts] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, self.opt.n_pts-len(choose)), 'wrap')
                    chooses[i,:]=choose
                self.choose =chooses.astype(np.int64)
                self.choose=torch.LongTensor(self.choose).to(device='cuda')
            self.forward()

        cur_loss_total, cur_loss_corr, cur_loss_recon3d, cur_loss_uv, cur_loss_mat, cur_loss_seg = self.criterion(self.assign_mat, self.seg_pred, self.verts3d_pred, self.uv_pred, self.nocs, self.model, self.mask, self.corr_mat, self.uvmap)
        self.inference_data['loss_total'].append(cur_loss_total.item())
        self.inference_data['loss_corr'].append(cur_loss_corr.item())
        self.inference_data['loss_recon3d'].append(cur_loss_recon3d.item())
        self.inference_data['loss_uv'].append(cur_loss_uv.item())
        self.inference_data['loss_mat'].append(cur_loss_mat.item())
        self.inference_data['loss_seg'].append(cur_loss_seg.item())
        self.inference_data['batch_size'].append(bs)



        if hasattr(self.opt, 'eval'):
            # 3D vertices
            inst_shape=self.verts3d_pred/9.0
            verts3d_pred = inst_shape.cpu().numpy().reshape(-1, 1220, 3)
            verts3d_gt = (self.model).cpu().numpy().reshape(-1, 1220, 3)
           
            #pts68_pred = ((self.out1 * 0.5 + 0.5) * self.opt.img_size).cpu().numpy().reshape(-1, 68, 2)
            assign_mat = F.softmax(self.assign_mat, dim=2)
            nocs_coords = torch.bmm(assign_mat, inst_shape)
            nocs_coords = nocs_coords.detach().cpu().numpy().reshape(-1, self.opt.n_pts, 3)

            img_size = 800
            f = 1.574437 * img_size / 2
            K_img = np.array([
                [f, 0, img_size / 2.0],
                [0, f, img_size / 2.0],
                [0, 0, 1]
            ])
            T = np.array([
                [1.0,  0,  0, 0],
                [  0, -1,  0, 0],
                [  0,  0, -1, 0],
                [  0,  0,  0, 1]
            ])

            for i in range(bs):
                choose = self.choose.cpu().numpy()[i]
                choose, choose_idx = np.unique(choose, return_index=True)
                nocs_coord = nocs_coords[i, choose_idx, :]
                col_idx = choose % self.opt.img_size
                row_idx = choose // self.opt.img_size
                local_pts2d = np.concatenate((col_idx.reshape(-1,1), row_idx.reshape(-1,1)), axis=1)

                tfm = self.tform_inv_list[i]
                W, b = tfm.T[:2], tfm.T[2]
                global_pts_pred = local_pts2d @ W + b
           
                if 1: 
                    _, rvecs, tvecs, _ = cv2.solvePnPRansac(
                        nocs_coord,
                        global_pts_pred,
                        K_img,
                        None
                    )
                else:
                    retval, rvecs, tvecs = cv2.solvePnP(
                        nocs_coord,
                        global_pts_pred,
                        K_img,
                        None,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                rotM = cv2.Rodrigues(rvecs)[0].T
                tvecs = tvecs.squeeze(axis=1)

                # GL
                R_temp = np.identity(4)
                R_temp[:3, :3] = rotM
                R_temp[3, :3] = tvecs
                R_t_pred = R_temp @ T


                # pnp
                if R_t_pred[3, 2] > 0:
                    print('【pnp_fail tz_pred > 0】', self.img_path_list[i])
                    self.inference_data['pnp_fail'] += 1
                    R_t_pred = np.identity(4)

                elif R_t_pred[3, 2] < -100:
                    print('【pnp_fail tz_pred < -100】', self.img_path_list[i])
                    self.inference_data['pnp_fail'] += 1
                    R_t_pred = np.identity(4)

                ### R_t -> tx,ty,tz,yaw,pitch,roll 측정
                R_t_gt = self.R_t_list[i]
                img_path = self.img_path_list[i]
                # yaw, pitch, roll 계산
                yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(R_t_gt[:3, :3].T).as_euler('yxz', degrees=True)
                yaw_pred, pitch_pred, roll_pred = Rotation.from_matrix(R_t_pred[:3, :3].T).as_euler('yxz', degrees=True)
                # tx ty tz 계산
                tx_gt, ty_gt, tz_gt = R_t_gt[3, :3]
                tx_pred, ty_pred, tz_pred = R_t_pred[3, :3]
                # rotation error 계산
                cur_pitch_mae = np.abs(pitch_gt - pitch_pred) # <--- 각도 에러를 이렇게 측정하는게 맞아?
                cur_yaw_mae = np.abs(yaw_gt - yaw_pred)
                cur_roll_mae = np.abs(roll_gt - roll_pred)
                # translation error 계산
                cur_tx_mae = np.abs(tx_gt - tx_pred)
                cur_ty_mae = np.abs(ty_gt - ty_pred)
                cur_tz_mae = np.abs(tz_gt - tz_pred)

                self.inference_data['pitch_mae'].append(cur_pitch_mae)
                self.inference_data['yaw_mae'].append(cur_yaw_mae)
                self.inference_data['roll_mae'].append(cur_roll_mae)
                self.inference_data['tx_mae'].append(cur_tx_mae)
                self.inference_data['ty_mae'].append(cur_ty_mae)
                self.inference_data['tz_mae'].append(cur_tz_mae)

                ### vertex error 계산 : L2 norm
                # 3DRecon
                cur_3drecon = np.sqrt(((verts3d_pred[i] - verts3d_gt[i]) ** 2).sum(axis=1)).mean()
                self.inference_data['3DRecon'].append(cur_3drecon)

                # ADD
                # ADD: 모델이 예측한 R,t 가 얼마나 정확한지 판단하기 위해 gt vertex 를 이용하는 metric 임.
                # ADD 가 작을수록 정확하게 R,t 를 예측한 것.
                verts_origin = verts3d_gt[i]
                ones = np.ones([verts_origin.shape[0], 1])
                verts_homo = np.concatenate([verts_origin, ones], axis=1)

                verts_cam_pred = verts_homo @ R_t_pred
                verts_cam_gt = verts_homo @ R_t_gt

                cur_ADD = np.sqrt(np.sum((verts_cam_pred[:, :3] - verts_cam_gt[:, :3]) ** 2, 1)).mean()
                self.inference_data['ADD'].append(cur_ADD)

                ### 이런 metric 이 또 있나보네;
                # IOU, 5°5cm, 10°5cm
                R_error, T_error, IoU = compute_sRT_errors(R_t_pred.T, R_t_gt.T)
                if R_error < 5 and T_error < 0.05:
                     self.inference_data['strict_success'] += 1
                if R_error < 10 and T_error < 0.05:
                    self.inference_data['easy_success'] += 1

                self.inference_data['IoU'].append(IoU)
                self.inference_data['total_count'] += 1
    



