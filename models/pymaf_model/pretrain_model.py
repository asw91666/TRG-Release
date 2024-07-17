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

class PretrainModel(nn.Module):
    """
    Parameter regression -> vertex regression
    ResNet50 -> ResNet18
    MLP -> GCN
    """

    def __init__(self, cfg, scratch=True):
        super().__init__()
        # self.cfg = cfg
        self.global_mode = True

        self.feature_extractor = get_resnet_encoder(cfg, global_mode=self.global_mode, pretrain=not scratch)
        self.inplanes = self.feature_extractor.inplanes

        # deconv layers
        self.inplanes = 512 # 무조건 512 로 고정.
        self.deconv_with_bias = cfg.RES_MODEL.DECONV_WITH_BIAS
        self.deconv_layers = self._make_deconv_layer(
            cfg.RES_MODEL.NUM_DECONV_LAYERS,
            cfg.RES_MODEL.NUM_DECONV_FILTERS,
            cfg.RES_MODEL.NUM_DECONV_KERNELS,
        )
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

    def forward(self, x):
        """
            x : norm img [B,3,192,192]
            bbox_info : tensor, [B,5] [left, top, right, bottom, focal_length]
            K_img : tensor, [B,4,3], it is already transposed
        """
        batch_size = x.shape[0]

        # CNN encoder
        high_level_feat, _ = self.feature_extractor(x)
        # Decoder
        dec_feat = self.deconv_layers(high_level_feat)

        # Get Heatmap
        heatmap_lmk68 = self.predict_68_lmk(dec_feat)

        # 2dsoft-argmax
        pred_lmk68 = soft_argmax2d(heatmap_lmk68)  # [B,68,2], [0~47]
        pred_lmk68 = pred_lmk68 / heatmap_lmk68.shape[-1] * 2 - 1 # [0~47] -> [-1~1]

        out_list = {}

        out_list['pred_lmk68'] = pred_lmk68

        return out_list

def get_pretrain_model(config, scratch=True):
    """ Constructs an PyMAF model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = PretrainModel(config, scratch=scratch)

    return model
