# This script is borrowed and extended from https://github.com/shunsukesaito/PIFu/blob/master/lib/model/SurfaceClassifier.py

from packaging import version
import torch
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.pymaf_model.core.cfgs import cfg
from .geometry import projection

import logging
logger = logging.getLogger(__name__)
from pdb import set_trace
from util.pkl import read_pkl

class MAF_Extractor(nn.Module):
    ''' Mesh-aligned Feature Extrator

    As discussed in the paper, we extract mesh-aligned features based on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    '''

    def __init__(self, full_img_size, crop_img_size, pos_emb, device=torch.device('cuda')):
        super().__init__()

        self.full_img_size = full_img_size
        self.cropped_img_size = crop_img_size
        self.device = device
        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.PyMAF.MLP_DIM
        self.last_op = nn.ReLU(True)

        for l in range(0, len(filter_channels) - 1):
            if 0 != l:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1))
            else:
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))

            self.add_module("conv%d" % l, self.filters[l])

        # positional embedding
        # assert pos_emb.shape == (305,3)
        # pos_emb = torch.FloatTensor(pos_emb).transpose(0,1) # [3,305]
        # self.register_buffer('t_verts', pos_emb[None, :, :]) # [1,3,305]
        # dim_feat = 256
        # dim_pos_emb = 3
        # self.offset_layer = nn.Conv1d(dim_feat + dim_pos_emb, 2, 1)

        # MAF extractor
        self.subsample = read_pkl('data/arkit_subsample.pkl')

    def reduce_dim(self, feature):
        '''
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        y = feature
        tmpy = feature
        for i, f in enumerate(self.filters): # len(self.filters) : 3
            y = self._modules['conv' + str(i)](
                y if i == 0
                else torch.cat([y, tmpy], 1)
            )
            if i != len(self.filters) - 1:
                # 마지막 layer 가 activation function 에 입력한다.
                y = F.leaky_relu(y)
            # if self.num_views > 1 and i == len(self.filters) // 2:
            #     y = y.view(
            #         -1, self.num_views, y.shape[1], y.shape[2]
            #     ).mean(dim=1)
            #     tmpy = feature.view(
            #         -1, self.num_views, feature.shape[1], feature.shape[2]
            #     ).mean(dim=1)

        y = self.last_op(y)

        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None, z_feat=None):
        '''
        Given 2D points, sample the point-wise features for each point, 
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps 
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        '''
        if im_feat is None:
            im_feat = self.im_feat

        batch_size = im_feat.shape[0]
        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0]

        mesh_align_feat = self.reduce_dim(point_feat)
        return mesh_align_feat

    def projection(self, points_world, extrinsic, intrinsic, bbox):
        """
            points_world : [B, nverts, 3]
            extrinsic : R_t, [B,4,4], It is already transposed.
            intrinsic : [B,4,3], It is already transposed.
            bbox : [B,5] [l,t,r,b,focal]

            points_img_crop: [B,nv,2]
        """
        device = points_world.device
        batch_size, n_vtx, _ = points_world.shape
        # points_world = points_world.reshape(batch_size, -1, 3) # [B, nverts, 3]

        ones = torch.ones([batch_size, n_vtx, 1], device=device)
        points_world_homo = torch.cat([points_world, ones], dim=2) # [B,nv,4]

        # calc point's coordinate on full size image space
        points_cam = torch.bmm(points_world_homo, extrinsic) # [B,nv,4]
        # cam coordi -> img coordi
        points_img_crop = torch.bmm(points_cam, intrinsic) # [B, nv, 3]
        z = points_img_crop[:, :, [2]]
        points_img_crop = points_img_crop / z  # divide by z
        points_img_crop = points_img_crop[:, :, :2]  # [B,nv,2]
        '''
        points_img_full = torch.bmm(points_cam, intrinsic) # [B, nv, 3]
        z = points_img_full[:,:,[2]]
        points_img_full = points_img_full / z # divide by z
        points_img_full = points_img_full[:, :, :2]  # [B,nv,2]

        # full size img space -> cropped img space
        # crop bbox
        points_img_crop = points_img_full.clone() # [B, nv, 2]
        points_img_crop[:, :, 0] = points_img_full[:, :, 0] - bbox[:, [0]] # x
        points_img_crop[:, :, 1] = points_img_full[:, :, 1] - bbox[:, [1]] # y

        # resize img
        bbox_size = bbox[:, 2] - bbox[:, 0]  # [B,5]
        resize = (self.cropped_img_size / bbox_size) # [B]
        resize = resize[:, None, None]
        points_img_crop = points_img_crop * resize
        '''

        # normalize
        points_img_crop_norm = points_img_crop / self.cropped_img_size * 2 - 1 # [0~191] -> [-1~1]
        return points_img_crop_norm, points_img_crop

    def get_offset(self, points, im_feat):
        """
        reference points -> offset
        """

        if version.parse(torch.__version__) >= version.parse('1.3.0'):
            # Default grid_sample behavior has changed to align_corners=False since 1.3.0.
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2), align_corners=True)[..., 0]
        else:
            point_feat = torch.nn.functional.grid_sample(im_feat, points.unsqueeze(2))[..., 0] # [B,256,305]

        batch_size = points.shape[0]
        t_verts = self.t_verts.expand(batch_size, -1, -1)  # [B,3,305]
        point_feat = torch.cat([point_feat, t_verts], dim=1) # [B,259,305]
        offset = self.offset_layer(point_feat) # [B,2,305]

        return offset.transpose(1,2) # [B,305,2]

    def forward(self, p, extr, intr, bbox, s_feat, **kwargs):
        ''' Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            extr (tensor): [B, 4, 4]
            intr (tensor): [B, 4, 4]
            bbox (tensor): [B, 5]
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        '''
        p = p[:, self.subsample["1220_to_305"], :]
        p_proj_2d_norm, p_proj_2d = self.projection(p, extr, intr, bbox) # normalized 2d point
        # offset = self.get_offset(p_proj_2d_norm, s_feat)
        # p_proj_2d = p_proj_2d + offset # denormed scale [0~191]
        # p_proj_2d_norm_refine = p_proj_2d / self.cropped_img_size * 2 - 1  # [0~191] -> [-1~1]
        # mesh_align_feat = self.sampling(p_proj_2d_norm_refine, s_feat)
        mesh_align_feat = self.sampling(p_proj_2d_norm, s_feat)
        return mesh_align_feat
