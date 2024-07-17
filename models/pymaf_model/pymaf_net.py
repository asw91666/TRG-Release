import torch
import torch.nn as nn
import numpy as np
from .pose_resnet import get_resnet_encoder, Preprocess_Heatmap
# from models.pymaf_model.core.cfgs import cfg
# from .geometry import rot6d_to_rotmat, projection, rotation_matrix_to_angle_axis
from .geometry import projection, rotation_matrix_to_angle_axis
from .maf_extractor import MAF_Extractor
from .smpl import SMPL, SMPL_MODEL_DIR, SMPL_MEAN_PARAMS, H36M_TO_J14
from .iuv_predictor import IUV_predict_layer
from ..op import rotmat_to_rot6d, rot6d_to_rotmat, create_batch_2d_gaussian_heatmaps
import argparse
from util.pkl import read_pkl
from models.op import soft_argmax2d, euler_to_rotation_matrix_batch
from util.metric import degree2radian, radian2degree
from util.random import sample_uniform_distribution_torch
import logging

logger = logging.getLogger(__name__)

BN_MOMENTUM = 0.1
from pdb import set_trace

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)

class Regressor(nn.Module):
    def __init__(self, feat_dim, smpl_mean_params):
        super().__init__()

        npose = 24 * 6

        self.fc1 = nn.Linear(feat_dim + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1) # 수정해야
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose # 수정해야
            pred_shape = self.decshape(xc) + pred_shape # 수정해야
            pred_cam = self.deccam(xc) + pred_cam # 수정해야

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output

    def forward_init(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=1, J_regressor=None):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam

        pred_rotmat = rot6d_to_rotmat(pred_pose.contiguous()).view(batch_size, 24, 3, 3)

        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )

        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        pred_smpl_joints = pred_output.smpl_joints
        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)

        if J_regressor is not None:
            pred_joints = torch.matmul(J_regressor, pred_vertices)
            pred_pelvis = pred_joints[:, [0], :].clone()
            pred_joints = pred_joints[:, H36M_TO_J14, :]
            pred_joints = pred_joints - pred_pelvis

        output = {
            'theta'  : torch.cat([pred_cam, pred_shape, pose], dim=1),
            'verts'  : pred_vertices,
            'kp_2d'  : pred_keypoints_2d,
            'kp_3d'  : pred_joints,
            'smpl_kp_3d' : pred_smpl_joints,
            'rotmat' : pred_rotmat,
            'pred_cam': pred_cam,
            'pred_shape': pred_shape,
            'pred_pose': pred_pose,
        }
        return output


class PyMAF(nn.Module):
    """ PyMAF based Deep Regressor for Human Mesh Recovery
    PyMAF: 3D Human Pose and Shape Regression with Pyramidal Mesh Alignment Feedback Loop, in ICCV, 2021
    """

    def __init__(self, smpl_mean_params=SMPL_MEAN_PARAMS, pretrained=True):
        super().__init__()
        self.global_mode = not cfg.MODEL.PyMAF.MAF_ON
        self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)

        # deconv layers
        self.inplanes = self.feature_extractor.inplanes
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            self.maf_extractor.append(MAF_Extractor())
        ma_feat_len = self.maf_extractor[-1].Dmap.shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]
        
        grid_size = 21
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len
            self.regressor.append(Regressor(feat_dim=ref_infeat_dim, smpl_mean_params=smpl_mean_params))

        dp_feat_dim = 256
        self.with_uv = cfg.LOSS.POINT_REGRESSION_WEIGHTS > 0
        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            self.dp_head = IUV_predict_layer(feat_dim=dp_feat_dim)

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

    def forward(self, x, J_regressor=None):

        batch_size = x.shape[0]

        # spatial features and global features
        s_feat, g_feat = self.feature_extractor(x)

        assert cfg.MODEL.PyMAF.N_ITER >= 0 and cfg.MODEL.PyMAF.N_ITER <= 3
        if cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif cfg.MODEL.PyMAF.N_ITER == 3:
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]

        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        smpl_output = self.regressor[0].forward_init(g_feat, J_regressor=J_regressor)

        out_list['smpl_out'] = [smpl_output]
        out_list['dp_out'] = []

        # for visulization
        vis_feat_list = [s_feat.detach()]

        # parameter predictions
        for rf_i in range(cfg.MODEL.PyMAF.N_ITER):
            pred_cam = smpl_output['pred_cam']
            pred_shape = smpl_output['pred_shape']
            pred_pose = smpl_output['pred_pose']

            pred_cam = pred_cam.detach()
            pred_shape = pred_shape.detach()
            pred_pose = pred_pose.detach()

            s_feat_i = deconv_blocks[rf_i](s_feat)
            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach())

            self.maf_extractor[rf_i].im_feat = s_feat_i
            self.maf_extractor[rf_i].cam = pred_cam

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points)
            else:
                pred_smpl_verts = smpl_output['verts'].detach()
                # TODO: use a more sparse SMPL implementation (with 431 vertices) for acceleration
                pred_smpl_verts_ds = torch.matmul(self.maf_extractor[rf_i].Dmap.unsqueeze(0), pred_smpl_verts) # [B, 431, 3]
                ref_feature = self.maf_extractor[rf_i](pred_smpl_verts_ds) # [B, 431 * n_feat]

            smpl_output = self.regressor[rf_i](ref_feature, pred_pose, pred_shape, pred_cam, n_iter=1, J_regressor=J_regressor)
            out_list['smpl_out'].append(smpl_output)

        if cfg.MODEL.PyMAF.AUX_SUPV_ON:
            iuv_out_dict = self.dp_head(s_feat)
            out_list['dp_out'].append(iuv_out_dict)

        return out_list, vis_feat_list

class Regressor_face(nn.Module):
    def __init__(self, feat_dim, init_vertex, init_cam, is_last=False):
        super().__init__()

        hidden_dim = 1024

        print(f'======================')
        print(f'feat_dim : {feat_dim}')
        print(f'hidden_dim : {hidden_dim}')
        print(f'is_last : {is_last}')
        nverts, dim = init_vertex.shape # [305, 3]
        vtx_dim = nverts * dim
        sub_vtx_dim = 305 * dim
        cam_rot = 6
        cam_param = 3
        cam_trans = 3
        bbox_dim = 3
        self.fc1 = nn.Linear(feat_dim + sub_vtx_dim + cam_rot + cam_param + bbox_dim, hidden_dim)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout()
        # self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(hidden_dim, vtx_dim)
        self.deccam = nn.Linear(hidden_dim, cam_rot + cam_param)
        # nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        assert init_cam.shape == (4,4)
        init_cam_ = torch.from_numpy(init_cam)
        init_cam_ = init_cam_.transpose(1,0)
        rotmat = init_cam_[:3, :3]
        # trans = init_cam_[:3, 3]

        # rotmat -> 6d
        rot6d = rotmat_to_rot6d(rotmat[None, None, :, :])  # [1,1,6]
        rot6d = rot6d[:, 0, :] # [1,6]
        # trans = trans[None, :] # [1,3]
        # init_cam_ = torch.cat([rot6d, trans], dim=1) # [1,9]
        # weak_cam_param = torch.tensor([[10.0, 0, 0]], dtype=torch.float32)
        weak_cam_param = torch.tensor([[1.0, 0, 0]], dtype=torch.float32)
        init_cam_ = torch.cat([rot6d, weak_cam_param], dim=1) # [1,9]

        init_vertex_ = init_vertex[None, :, :] # [1,1220,3]
        init_vertex_ = torch.from_numpy(init_vertex_)

        self.register_buffer('init_vertex', init_vertex_)
        self.register_buffer('init_cam', init_cam_)

        # self.upsample = None
        # if is_last:
        #     self.upsample = nn.Linear(305,1220)

        self.physical_bbox_size = 0.2

        self.subsample = read_pkl('data/arkit_subsample.pkl')

    def cam_crop2full_jsh(self, crop_cam, center, bbox_size, full_img_shape, focal_length):
        """
        convert the camera parameters from the crop camera to the full camera
        :param crop_cam: shape=(N, 3) weak perspective camera in cropped img coordinates (s, tx, ty)
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
        tz = physical_bbox_size * crop_cam[:, 0] * focal_length / b
        # Tx
        tx_face = crop_cam[:,1] * (physical_bbox_size * 0.5) * crop_cam[:, 0]
        tx_bbox = physical_bbox_size * crop_cam[:, 0] * (cx - w_2) / b
        tx = tx_face + tx_bbox
        # Ty
        ty_face = crop_cam[:, 2] * (physical_bbox_size * 0.5) * crop_cam[:, 0]
        ty_bbox = physical_bbox_size * crop_cam[:, 0] * (cy - h_2) / b
        ty = ty_face + ty_bbox
        # return
        full_cam = torch.stack([tx, ty, tz], dim=-1)
        return full_cam

    def forward(self, x, pred_vertex, pred_cam, bbox_info, n_iter=1):
        """
            x : sampled feature
            pred_vertex: [B,305,3]
            pred_cam: [B,9]
            bbox : [B,7] [l,t,r,b,focal, h, w]
        """
        batch_size, n_vtx, _ = pred_vertex.shape
        device = pred_vertex.device
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

        # reshape
        pred_sub_vertex = pred_vertex[:, self.subsample["1220_to_305"], :].clone() # [B,305,3]
        pred_sub_vertex = pred_sub_vertex.reshape(batch_size, -1) # [B,305*3]
        pred_vertex = pred_vertex.reshape(batch_size, -1) # [B,1220*3]

        for i in range(n_iter):
            xc = torch.cat([x, pred_sub_vertex, pred_cam, bbox_info_], dim=1)
            # xc = torch.cat([x, pred_sub_vertex, pred_rot6d, pred_cam_param, bbox_info_], dim=1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_vertex_world = self.decshape(xc) + pred_vertex
            offset_R_t = self.deccam(xc)

            # rotation
            offset_rotmat = rot6d_to_rotmat(offset_R_t[:,:6]) # [B,3,3]
            cur_rotmat = rot6d_to_rotmat(pred_cam[:, :6]) # [B,3,3]
            pred_rotmat = torch.bmm(cur_rotmat, offset_rotmat) # [B,3,3]
            pred_rot6d = rotmat_to_rot6d(pred_rotmat[:,None,:,:])[:,0,:] # [B,6]

            # weak-perspective cam parameter
            pred_s_t = offset_R_t[:,6:] + pred_cam[:,6:] # [B,3]
            pred_cam = torch.cat([pred_rot6d, pred_s_t], dim=1)

        pred_trans = self.cam_crop2full_jsh(pred_s_t, bbox_center, bbox_size, full_img_shape, focal_length)

        pred_R_t = torch.zeros([batch_size, 4, 4], dtype=pred_rotmat.dtype, device=device)
        pred_R_t[:, :3, :3] = pred_rotmat
        pred_R_t[:, :3, 3] = pred_trans
        pred_R_t[:, 3, 3] = 1
        pred_R_t = pred_R_t.transpose(1, 2) # [B,4,4]

        # calc vertex in camera space
        pred_vertex_world = pred_vertex_world.reshape(batch_size, -1, 3) # [B, 305, 3]
        ones = torch.ones([batch_size, n_vtx, 1], device=device)
        pred_vertex_world_homo = torch.cat([pred_vertex_world, ones], dim=2) # [B,305,4]

        # world space -> camera space
        # pred_vertex_cam = torch.bmm(pred_vertex_world_homo, R_t)[:, :, :3] # [B,305,3]
        pred_vertex_cam = torch.bmm(pred_vertex_world_homo, pred_R_t)[:, :, :3] # [B,305,3]

        output = {
            'pred_vtx_world': pred_vertex_world,
            # 'pred_vtx_full_world': pred_vertex_full_world,
            'pred_cam': pred_cam,
            'pred_R_t': pred_R_t,
            'pred_vtx_cam': pred_vertex_cam,
            # 'pred_vtx_full_cam': pred_vertex_full_cam,
            'pred_s_t': pred_s_t
        }
        return output

    def forward_init(self, batch_size, init_vertex=None, init_cam=None):
        # batch_size = x.shape[0]

        if init_vertex is None:
            init_vertex = self.init_vertex.expand(batch_size, -1, -1)  # [B,305,3]
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)  # [B,9]

        device = init_vertex.device

        # pred_R = rot6d_to_rotmat(init_cam[:, :6].contiguous())  # [B,3,3]?
        # pred_t = init_cam[:, 6:]  # [B,3]
        # pred_R_t = torch.zeros([batch_size,4,4], dtype=pred_R.dtype, device=device)
        # pred_R_t[:, :3, :3] = pred_R
        # pred_R_t[:, :3, 3] = pred_t
        # pred_R_t[:, 3,3] = 1
        # pred_R_t = pred_R_t.transpose(1,2)

        # calc vertex in camera space
        # pred_vertex_world = pred_vertex.reshape(batch_size, -1, 3)  # [B, 305, 3]
        # n_vtx = init_vertex.shape[1]
        # ones = torch.ones([batch_size, n_vtx, 1], device=device) # [B,305,1]
        # init_vertex_world_homo = torch.cat([init_vertex, ones], dim=2)  # [B,305,4]

        # # world space -> camera space
        # init_vertex_cam = torch.bmm(init_vertex_world_homo, pred_R_t)[:, :, :3]  # [B,305,3]

        output = {
            'pred_vtx_world': init_vertex,
            'pred_cam': init_cam,
            # 'pred_R_t': pred_R_t,
            # 'pred_vtx_cam': init_vertex_cam
            'pred_R_t': None,
            'pred_vtx_cam': None

        }
        return output

class PyMAF_face(nn.Module):
    """
    Parameter regression -> vertex regression
    ResNet50 -> ResNet18
    MLP -> GCN
    """

    def __init__(self, cfg, init_vertex_full, init_cam):
        super().__init__()
        self.cfg = cfg
        self.global_mode = not cfg.MODEL.PyMAF.MAF_ON

        self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode)
        self.inplanes = self.feature_extractor.inplanes

        # deconv layers
        self.inplanes = 512 # 무조건 512 로 고정.
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )

        # load subsample dict
        self.subsample = read_pkl('data/arkit_subsample.pkl')
        # get init vtx and landmark
        # init_vertex = init_vertex_full[subsample["1220_to_305"], :]  # [305,3]
        # self.num_vtx = init_vertex.shape[0]

        self.maf_extractor = nn.ModuleList()
        for _ in range(cfg.MODEL.PyMAF.N_ITER):
            self.maf_extractor.append(MAF_Extractor(
                full_img_size=cfg.MODEL.PyMAF.FULL_IMG_SIZE,
                crop_img_size=cfg.MODEL.PyMAF.INPUT_IMG_SIZE,
                pos_emb=init_vertex_full)
            )
        ma_feat_len = init_vertex_full[self.subsample["1220_to_305"], :].shape[0] * cfg.MODEL.PyMAF.MLP_DIM[-1]

        grid_size = 18 # 18*18=324, which is similar to 305
        xv, yv = torch.meshgrid([torch.linspace(-1, 1, grid_size), torch.linspace(-1, 1, grid_size)])
        points_grid = torch.stack([xv.reshape(-1), yv.reshape(-1)]).unsqueeze(0)
        self.register_buffer('points_grid', points_grid)
        grid_feat_len = grid_size * grid_size * cfg.MODEL.PyMAF.MLP_DIM[-1]

        self.regressor = nn.ModuleList()
        for i in range(cfg.MODEL.PyMAF.N_ITER):
            if i == 0:
                ref_infeat_dim = grid_feat_len
            else:
                ref_infeat_dim = ma_feat_len

            self.regressor.append(Regressor_face(feat_dim=ref_infeat_dim, init_vertex=init_vertex_full, init_cam=init_cam))

        deconv_dim = 256
        self.predict_68_lmk = nn.Sequential(
            nn.Conv2d(
                in_channels=deconv_dim,
                out_channels=68,
                kernel_size=1,
                stride=1,
                padding=0)
        )

        num_resnet_layer = cfg.POSE_RES_MODEL.EXTRA.NUM_LAYERS
        if num_resnet_layer == 50:
            self.conv1x1 = nn.Conv2d(
                in_channels=2048,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0
            )
        else:
            self.conv1x1 = None

        #############################################################
        # keypoint
        #############################################################
        self.kpt5_idx = [39, 42, 33, 48, 54]

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

    def augment_R_t(self, R_t, pred_cam, loop_i, device=torch.device('cuda')):
        if loop_i == 1:
            roll_aug_param = self.roll_aug_param
            pitch_aug_param = self.pitch_aug_param
            yaw_aug_param = self.yaw_aug_param

        ####################################################################################
        # Augment Rotation
        ####################################################################################
        batch_size = len(R_t)
        roll = sample_uniform_distribution_torch(roll_aug_param, size=batch_size).to(device)
        pitch = sample_uniform_distribution_torch(pitch_aug_param, size=batch_size).to(device)
        yaw = sample_uniform_distribution_torch(yaw_aug_param, size=batch_size).to(device)
        delta_rotmat = euler_to_rotation_matrix_batch(roll, pitch, yaw) # [B,3,3]

        rotmat = R_t[:, :3, :3].transpose(1,2)  # [B,3,3]
        aug_rotmat = torch.bmm(rotmat, delta_rotmat.detach()) # [B,3,3]

        ####################################################################################
        # transform 4x4 matrix -> 9d vector
        ####################################################################################
        ## 3x3 rotmat -> 6d representation
        aug_rot6d = rotmat_to_rot6d(aug_rotmat[:, None, :, :])[:, 0, :]  # [B,6]

        ## translation
        # trans = R_t[:, 3, :3].clone()
        camera_vec_9d = torch.cat([aug_rot6d, pred_cam[:,6:]], dim=1) # [B,9]

        aug_R_t = torch.zeros([batch_size, 4, 4], dtype=R_t.dtype, device=device)
        aug_R_t[:, :3, :3] = aug_rotmat
        aug_R_t[:, :3, 3] = R_t[:, 3, :3]
        aug_R_t[:, 3, 3] = 1
        aug_R_t = aug_R_t.transpose(1, 2)

        return aug_R_t, camera_vec_9d

    def forward(self, x, K_img=None, bbox_info=None, is_train=False):
        """
            x : norm img [B,3,192,192]
            bbox_info : tensor, [B,5] [left, top, right, bottom, focal_length]
            K_img : tensor, [B,4,3], it is already transposed
        """
        batch_size = x.shape[0]

        s_feat, _ = self.feature_extractor(x)

        if self.conv1x1 is not None and s_feat.shape[1] == 2048:
            s_feat = self.conv1x1(s_feat) # [B, 512, 6, 6]

        assert self.cfg.MODEL.PyMAF.N_ITER >= 0 and self.cfg.MODEL.PyMAF.N_ITER <= 3

        if self.cfg.MODEL.PyMAF.N_ITER == 1:
            deconv_blocks = [self.deconv_layers]
        elif self.cfg.MODEL.PyMAF.N_ITER == 2:
            deconv_blocks = [self.deconv_layers[0:6], self.deconv_layers[6:9]]
        elif self.cfg.MODEL.PyMAF.N_ITER == 3: # Do
            deconv_blocks = [self.deconv_layers[0:3], self.deconv_layers[3:6], self.deconv_layers[6:9]]
        out_list = {}

        # initial parameters
        # TODO: remove the initial mesh generation during forward to reduce runtime
        # by generating initial mesh the beforehand: smpl_output = self.init_smpl
        output = self.regressor[0].forward_init(batch_size) # g_feat 가 사용되지 않는다. 그냥 init pose 로 복원한것.
        """
            output = {
            'pred_vtx': init_vertex,
            'pred_cam': init_cam,
            'pred_R_t': pred_R_t,
        }
        """
        out_list['output'] = [output]
        # out_list['pred_mask'] = []
        out_list['pred_lmk68'] = []
        out_list['aug_R_t'] = []
        out_list['samp_pt'] = []

        # for visulization
        vis_feat_list = [s_feat.detach().clone()]
        # parameter predictions
        for rf_i in range(self.cfg.MODEL.PyMAF.N_ITER):
            pred_cam = output['pred_cam'] # [B,9]
            pred_vtx = output['pred_vtx_world'] # [B,1220,3]
            # pred_vtx_cam = output['pred_vtx_cam'] # [B,305,3]
            pred_R_t = output['pred_R_t'] # [B,4,4]

            # if is_train and rf_i == 1:
            #     pred_R_t, pred_cam = self.augment_R_t(pred_R_t.clone(), pred_cam.clone(), loop_i=rf_i, device=pred_R_t.device)
            #     out_list['aug_R_t'].append(pred_R_t.detach().clone())

            s_feat_i = deconv_blocks[rf_i](s_feat)

            s_feat = s_feat_i
            vis_feat_list.append(s_feat_i.detach().clone())

            if rf_i == 0:
                sample_points = torch.transpose(self.points_grid.expand(batch_size, -1, -1), 1, 2)
                ref_feature = self.maf_extractor[rf_i].sampling(sample_points, s_feat_i)
            else:
                # [B, 431 * n_feat]
                ref_feature = self.maf_extractor[rf_i](pred_vtx, pred_R_t, K_img, bbox_info, s_feat_i)

            # regress
            output = self.regressor[rf_i](ref_feature, pred_vtx, pred_cam, bbox_info, n_iter=1)
            out_list['output'].append(output)

        # pred_mask = self.predict_mask(s_feat) # [B,2,48,48]
        pred_lmk68_heatmap = self.predict_68_lmk(s_feat) # [B,68,48,48]
        # 2dsoft-argmax
        pred_lmk68 = soft_argmax2d(pred_lmk68_heatmap) # [B,68,2]
        pred_lmk68 = pred_lmk68 / pred_lmk68_heatmap.shape[2] * 2 - 1 # [0~47] -> [-1~1]

        out_list['pred_lmk68'].append(pred_lmk68)

        return out_list, vis_feat_list

def pymaf_net(smpl_mean_params, pretrained=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PyMAF(smpl_mean_params, pretrained)
    return model

def pymaf_face_net(config, init_vertex, init_cam, pretrain_mode=False):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = PyMAF_face(config, init_vertex, init_cam)

    # model.deconv_layers.apply(init_weights)
    # model.maf_extractor.apply(init_weights)
    # model.regressor.apply(init_weights)

    return model
