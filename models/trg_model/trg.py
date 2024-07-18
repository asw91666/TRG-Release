import torch
import torch.nn as nn
import numpy as np
from .pose_resnet import get_resnet_encoder, Preprocess_Heatmap
from .laf_extractor import LAF_Extractor
from ..op import rotmat_to_rot6d, rot6d_to_rotmat, create_batch_2d_gaussian_heatmaps
from util.pkl import read_pkl
from models.op import soft_argmax2d, euler_to_rotation_matrix_batch
import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1
from pdb import set_trace

class FaceRegressor(nn.Module):
    def __init__(self, feat_dim, init_face, init_pose):
        super().__init__()

        hidden_dim = 1024

        print(f'======================')
        print(f'feat_dim : {feat_dim}')
        print(f'hidden_dim : {hidden_dim}')
        nverts, dim = init_face.shape # [305, 3]
        vtx_dim = nverts * dim
        sub_vtx_dim = 305 * dim
        cam_rot = 6
        cam_param = 3
        bbox_dim = 3
        self.fc1 = nn.Linear(feat_dim + sub_vtx_dim + cam_rot + cam_param + bbox_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        self.decshape = nn.Linear(hidden_dim, vtx_dim)
        self.decpose = nn.Linear(hidden_dim, cam_rot + cam_param)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)

        assert init_pose.shape == (4,4)
        init_pose_ = torch.from_numpy(init_pose)
        init_pose_ = init_pose_.transpose(1,0)
        rotmat = init_pose_[:3, :3]

        # rotmat -> 6d
        rot6d = rotmat_to_rot6d(rotmat[None, None, :, :])  # [1,1,6]
        rot6d = rot6d[:, 0, :] # [1,6]
        correction_param = torch.tensor([[1.0, 0, 0]], dtype=torch.float32)
        init_head_pose = torch.cat([rot6d, correction_param], dim=1) # [1,9]

        init_face_ = init_face[None, :, :] # [1,1220,3]
        init_face_ = torch.from_numpy(init_face_)

        self.register_buffer('init_face', init_face_)
        self.register_buffer('init_head_pose', init_head_pose)

        self.physical_bbox_size = 0.2

        self.subsample = read_pkl('data/arkit_subsample.pkl')

    def calc_trans_from_param(self, cor_param, center, bbox_size, full_img_shape, focal_length):
        """
        convert the camera parameters from the crop camera to the full camera
        :param cor_param: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
        :param center: shape=(N, 2) bbox coordinates (c_x, c_y)
        :param bbox_size: shape=(N,) square bbox resolution
        :param full_img_shape: shape=(N, 2) original image height and width
        :param focal_length: shape=(N,)
        :return:
        """
        physical_bbox_size = self.physical_bbox_size # which cover real human face, meter unit
        img_h, img_w = full_img_shape[:, 0], full_img_shape[:, 1]
        cx, cy, b = center[:, 0], center[:, 1], bbox_size
        w_2, h_2 = img_w * 0.5, img_h * 0.5

        # Tz
        tz = physical_bbox_size * cor_param[:, 0] * focal_length / b
        # Tx
        tx_face = cor_param[:,1] * (physical_bbox_size * 0.5) * cor_param[:, 0]
        tx_bbox = physical_bbox_size * cor_param[:, 0] * (cx - w_2) / b
        tx = tx_face + tx_bbox
        # Ty
        ty_face = cor_param[:, 2] * (physical_bbox_size * 0.5) * cor_param[:, 0]
        ty_bbox = physical_bbox_size * cor_param[:, 0] * (cy - h_2) / b
        ty = ty_face + ty_bbox
        # return
        translation = torch.stack([tx, ty, tz], dim=-1)
        return translation

    def forward(self, x, pred_face, pred_pose, bbox_info, n_iter=1):
        """
            x : sampled feature
            pred_face: [B,305,3]
            pred_pose: [B,9]
            bbox : [B,7] [l,t,r,b,focal, h, w]
        """
        batch_size, n_vtx, _ = pred_face.shape
        device = pred_face.device
        ########################################################################
        # build bbox info
        ########################################################################
        bbox_size = bbox_info[:, 2] - bbox_info[:, 0]
        x_center = (bbox_info[:, 0] + bbox_info[:, 2]) * 0.5
        y_center = (bbox_info[:, 1] + bbox_info[:, 3]) * 0.5
        img_h, img_w = bbox_info[:, 5], bbox_info[:, 6]
        focal_length = bbox_info[:, 4]
        rescale = 4
        bbox_info_ = torch.zeros((batch_size, 3), device=device)
        bbox_info_[:,0] = (x_center - img_w * 0.5)/focal_length * rescale
        bbox_info_[:,1] = (y_center - img_h * 0.5)/focal_length * rescale
        bbox_info_[:,2] = bbox_size/focal_length
        bbox_center = torch.cat([x_center[:, None], y_center[:, None]], dim=1)  # [B,2]
        full_img_shape = bbox_info[:, 5:]  # [B,2]

        for i in range(n_iter):
            # reshape
            pred_sub_face = pred_face[:, self.subsample["1220_to_305"], :].clone()  # [B,305,3]
            pred_sub_face = pred_sub_face.reshape(batch_size, -1)  # [B,305*3]
            pred_face = pred_face.reshape(batch_size, -1)  # [B,1220*3]

            xc = torch.cat([x, pred_sub_face, pred_pose, bbox_info_], dim=1)
            # xc = torch.cat([x, pred_sub_face, pred_rot6d, pred_cam_param, bbox_info_], dim=1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_face_world = self.decshape(xc) + pred_face
            pred_pose = self.decpose(xc) + pred_pose # [B,9]

        pred_cor_param = pred_pose[:, 6:]
        pred_trans = self.calc_trans_from_param(pred_cor_param, bbox_center, bbox_size, full_img_shape, focal_length)

        pred_rotmat = rot6d_to_rotmat(pred_pose[:,:6]) # [B,3,3]
        pred_R_t = torch.zeros([batch_size, 4, 4], dtype=pred_rotmat.dtype, device=device)
        pred_R_t[:, :3, :3] = pred_rotmat
        pred_R_t[:, :3, 3] = pred_trans
        pred_R_t[:, 3, 3] = 1
        pred_R_t = pred_R_t.transpose(1, 2) # [B,4,4]

        # calc face in camera space
        pred_face_world = pred_face_world.reshape(batch_size, -1, 3) # [B, 1220, 3]
        ones = torch.ones([batch_size, n_vtx, 1], device=device)
        pred_face_world_homo = torch.cat([pred_face_world, ones], dim=2) # [B,305,4]

        # world space -> camera space
        # pred_face_cam = torch.bmm(pred_face_world_homo, R_t)[:, :, :3] # [B,305,3]
        pred_face_cam = torch.bmm(pred_face_world_homo, pred_R_t)[:, :, :3] # [B,305,3]

        output = {
            'pred_face_world': pred_face_world,
            'pred_pose': pred_pose,
            'pred_R_t': pred_R_t,
            'pred_face_cam': pred_face_cam,
            'pred_cor_param': pred_cor_param
        }
        return output

    def forward_init(self, batch_size, init_face=None, init_head_pose=None):
        # batch_size = x.shape[0]

        if init_face is None:
            init_face = self.init_face.expand(batch_size, -1, -1)  # [B,305,3]
        if init_head_pose is None:
            init_head_pose = self.init_head_pose.expand(batch_size, -1)  # [B,9]

        device = init_face.device

        output = {
            'pred_face_world': init_face,
            'pred_pose': init_head_pose,
            'pred_R_t': None,
            'pred_face_cam': None
        }
        return output

class TRG(nn.Module):
    def __init__(self, cfg, init_face_full, init_pose):
        super().__init__()
        self.cfg = cfg
        self.global_mode = not cfg.MODEL.TRG.LAF_ON

        self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)
        self.inplanes = self.feature_extractor.inplanes

        # deconv layers
        self.inplanes = 512
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        # load subsample dict
        self.subsample = read_pkl('data/arkit_subsample.pkl')

        self.laf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.TRG.N_ITER):
            self.laf_extractor.append(LAF_Extractor(
                full_img_size=cfg.MODEL.TRG.FULL_IMG_SIZE,
                crop_img_size=cfg.MODEL.TRG.INPUT_IMG_SIZE)
            )
        ma_feat_len = init_face_full[self.subsample["1220_to_305"], :].shape[0] * cfg.MODEL.TRG.MLP_DIM[-1]

        grid_size = 18 # 18*18=324, which is similar to 305
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.TRG.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.TRG.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len

            self.regressor.append(FaceRegressor(feat_dim=ref_infeat_dim, init_face=init_face_full, init_pose=init_pose))

        deconv_dim = 256

        self.predict_68_lmk = nn.Sequential(
            nn.Conv2d(
                in_channels=deconv_dim,
                out_channels=68,
                kernel_size=1,
                stride=1,
                padding=0)
            )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Deconv_layer used in Simple Baselines:
        Xiao et al. Simple Baselines for Human Pose Estimation and Tracking
        https://github.com/microsoft/human-pose-estimation.pytorch
        """
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        def _get_deconv_cfg(deconv_kernel, index):
            if deconv_kernel == 4:
                padding = 1
                output_padding = 0
            elif deconv_kernel == 3:
                padding = 1
                output_padding = 1
            elif deconv_kernel == 2:
                padding = 0
                output_padding = 0

            return deconv_kernel, padding, output_padding

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = _get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, K_img=None, bbox_info=None):
        """
            x : norm img [B,3,192,192]
            bbox_info : tensor, [B,5] [left, top, right, bottom, focal_length]
            K_img : tensor, [B,4,3], it is already transposed
        """
        batch_size = x.shape[0]

        ################################################################
        # 2d joints -> joint heatmaps
        ################################################################
        # s_feat, _ = self.feature_extractor(x, opt_heatmap)
        s_feat, _ = self.feature_extractor(x)

        assert self.cfg.MODEL.TRG.N_ITER >= 0 and self.cfg.MODEL.TRG.N_ITER <= 3

        if self.cfg.MODEL.TRG.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif self.cfg.MODEL.TRG.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif self.cfg.MODEL.TRG.N_ITER == 3: # Do
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]
        out_list = {}

        # initial parameters
        output = self.regressor[0].forward_init(batch_size)
        out_list['output'] = [output]
        out_list['pred_lmk68'] = []
        out_list['samp_pt'] = []

        # for visulization
        vis_feat_list = [s_feat.detach().clone()]
        # parameter predictions
        for rf_i in range(self.cfg.MODEL.TRG.N_ITER):
            pred_pose = output['pred_pose'] # [B,9]
            pred_face = output['pred_face_world'] # [B,1220,3]
            pred_R_t = output['pred_R_t'] # [B,4,4]

            s_feat_i = deconv_blocks[rf_i](s_feat)

            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach().clone())

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature = self.laf_extractor[rf_i].sampling(sample_points, s_feat_i)
            else:
                # [B, 431 * n_feat]
                ref_feature, sampling_points = self.laf_extractor[rf_i](pred_face, pred_R_t, K_img, bbox_info, s_feat_i)
                out_list['samp_pt'].append(sampling_points)

            # regress
            output = self.regressor[rf_i](ref_feature, pred_face, pred_pose, bbox_info, n_iter=1)
            out_list['output'].append(output)

        # pred_mask = self.predict_mask(s_feat) # [B,2,48,48]
        pred_lmk68_heatmap = self.predict_68_lmk(s_feat) # [B,68,48,48]
        # 2dsoft-argmax
        pred_lmk68 = soft_argmax2d(pred_lmk68_heatmap) # [B,68,2]
        pred_lmk68 = pred_lmk68 / pred_lmk68_heatmap.shape[2] * 2 - 1 # [0~47] -> [-1~1]

        out_list['pred_lmk68'].append(pred_lmk68)

        return out_list, vis_feat_list

def load_trg(config, init_face, init_pose):

    model = TRG(config, init_face, init_pose)
    return model
