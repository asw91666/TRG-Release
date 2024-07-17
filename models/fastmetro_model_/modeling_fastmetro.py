# ----------------------------------------------------------------------------------------------
# FastMETRO Official Code
# Copyright (c) POSTECH Algorithmic Machine Intelligence Lab. (P-AMI Lab.) All Rights Reserved 
# Licensed under the MIT license.
# ----------------------------------------------------------------------------------------------

"""
FastMETRO model.
"""
from __future__ import absolute_import, division, print_function
import torch
import numpy as np
from torch import nn
from .transformer import build_transformer
from .position_encoding import build_position_encoding
from .smpl_param_regressor import build_smpl_parameter_regressor
import os
from pdb import set_trace
from .op import rot6d_to_rotmat

class FastMETRO_Body_Network(nn.Module):
    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""
    def __init__(self, args, backbone, mesh_sampler, num_joints=14, num_vertices=431):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        
        # the number of transformer layers
        if 'FastMETRO-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'FastMETRO-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'FastMETRO-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"
    
        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead, 
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        
        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'], hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'], hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        
        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool)) 
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool)) 
        adjacency_indices = torch.load('./src/modeling/data/smpl_431_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./src/modeling/data/smpl_431_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./src/modeling/data/smpl_431_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
        
        # learnable upsampling layer is used (from coarse mesh to intermediate mesh); for visually pleasing mesh result
        ### pre-computed upsampling matrix is used (from intermediate mesh to fine mesh); to reduce optimization difficulty
        self.coarse2intermediate_upsample = nn.Linear(431, 1723)

        # (optional) smpl parameter regressor; to obtain SMPL parameters
        if args.use_smpl_param_regressor:
            self.smpl_parameter_regressor = build_smpl_parameter_regressor()
    
    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # 1 X batch_size X 512 
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1) # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(device) # (num_joints + num_vertices) X (num_joints + num_vertices)
        
        # extract image features through a CNN backbone
        img_features = self.backbone(images) # batch_size X 2048 X 7 X 7
        _, _, h, w = img_features.shape
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        
        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 128 

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=attention_mask)
        
        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1) # 1 X batch_size X 128 
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1) # 49 X batch_size X 128 
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1) # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        cam_features_2, _, jv_features_2 = self.transformer_2(reduced_enc_img_features_1, reduced_cam_features_1, reduced_jv_features_1, pos_enc_2, attention_mask=attention_mask) 

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3) # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(0, 1)) # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:,:self.num_joints,:] # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:,self.num_joints:,:] # batch_size X num_vertices(coarse) X 3
        
        # coarse-to-intermediate mesh upsampling
        pred_3d_vertices_intermediate = self.coarse2intermediate_upsample(pred_3d_vertices_coarse.transpose(1,2)).transpose(1,2) # batch_size X num_vertices(intermediate) X 3
        # intermediate-to-fine mesh upsampling
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_intermediate, n1=1, n2=0) # batch_size X num_vertices(fine) X 3

        out = {}
        out['pred_cam'] = pred_cam
        out['pred_3d_joints'] = pred_3d_joints
        out['pred_3d_vertices_coarse'] = pred_3d_vertices_coarse
        out['pred_3d_vertices_intermediate'] = pred_3d_vertices_intermediate
        out['pred_3d_vertices_fine'] = pred_3d_vertices_fine

        # (optional) regress smpl parameters
        if self.args.use_smpl_param_regressor:
            pred_rotmat, pred_betas = self.smpl_parameter_regressor(pred_3d_vertices_intermediate.clone().detach())
            out['pred_rotmat'] = pred_rotmat
            out['pred_betas'] = pred_betas

        return out


class FastMETRO_Face_Network(nn.Module):
    """FastMETRO for 3D human pose and mesh reconstruction from a single RGB image"""

    def __init__(self, args, backbone, attn_mask):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.num_vertices = 305
        # self.num_joints = 14
        # the number of transformer layers
        if 'DCFace-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'DCFace-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'DCFace-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"

        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout,
                                     "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers,
                                     "num_dec_layers": num_dec_layers,
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout,
                                     "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers,
                                     "num_dec_layers": num_dec_layers,
                                     "pos_type": args.pos_type}
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_cam = nn.Linear(self.transformer_config_1["model_dim"],
                                            self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"],
                                            self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])

        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        # self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'],
                                                           hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'],
                                                           hidden_dim=self.transformer_config_2['model_dim'])
        # bbox token
        self.bbox_encoding_1 = nn.Linear(3, self.transformer_config_1["model_dim"])
        self.bbox_encoding_2 = nn.Linear(3, self.transformer_config_2["model_dim"])

        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 9)

        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        if isinstance(attn_mask, np.ndarray):
            attn_mask = torch.from_numpy(attn_mask)

        attention_mask = torch.full([self.num_vertices+1, self.num_vertices+1], False) # [306,306]
        attention_mask[:self.num_vertices, :self.num_vertices] = attn_mask
        self.attention_mask = attention_mask
        assert self.attention_mask.shape == (306,306)
        self.upsample = nn.Linear(305, 1220)

    def forward(self, images, bbox_info):
        device = images.device
        batch_size = images.size(0)

        # preparation cam, vtx, attn mask
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # 1 X batch_size X 512
        vtx_tokens = self.vertex_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # num_vertices X batch_size X 512
        attention_mask = self.attention_mask.to(device)  # (num_vertices) X num_vertices
        # preparation bbox token
        bbox_size = bbox_info[:, 2] - bbox_info[:, 0]
        x_center = (bbox_info[:, 0] + bbox_info[:, 2]) * 0.5
        y_center = (bbox_info[:, 1] + bbox_info[:, 3]) * 0.5
        focal_length = bbox_info[:, 4]
        bbox_info_ = torch.zeros((batch_size, 3), device=device) # [B,3]
        bbox_info_[:, 0] = x_center / focal_length
        bbox_info_[:, 1] = y_center / focal_length
        bbox_info_[:, 2] = bbox_size / focal_length
        bbox_info_ = bbox_info_[None, : ,:]
        bbox_token_1 = self.bbox_encoding_1(bbox_info_)
        bbox_token_2 = self.bbox_encoding_2(bbox_info_)

        # extract image features through a CNN backbone
        img_features = self.backbone(images)  # batch_size X 2048 X 6 X 6
        _, _, h, w = img_features.shape
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1)  # 36 X batch_size X 512

        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1)  # 36 X batch_size X 512
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1)  # 36 X batch_size X 128

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, vtx_features_1 = self.transformer_1(img_features,
                                                                               bbox_token_1,
                                                                               cam_token,
                                                                               vtx_tokens,
                                                                               pos_enc_1,
                                                                               attention_mask=attention_mask)

        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_cam(cam_features_1)  # 1 X batch_size X 128
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1)  # 49 X batch_size X 128
        reduced_vtx_features_1 = self.dim_reduce_dec(vtx_features_1)  # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        cam_features_2, _, vtx_features_2 = self.transformer_2(reduced_enc_img_features_1,
                                                              bbox_token_2,
                                                              reduced_cam_features_1,
                                                              reduced_vtx_features_1,
                                                              pos_enc_2,
                                                              attention_mask=attention_mask)

        # camera parameter
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, -1)  # batch_size X 9
        pred_R = rot6d_to_rotmat(pred_cam[:,:6].contiguous()) # [B,3,3]
        pred_t = pred_cam[:,6:] # [B,3]
        pred_R_t = torch.zeros([batch_size, 4, 4], dtype=pred_R.dtype, device=device)
        pred_R_t[:, :3, :3] = pred_R
        pred_R_t[:, :3, 3] = pred_t
        pred_R_t[:, 3, 3] = 1
        pred_R_t = pred_R_t.transpose(1, 2)  # [B,4,4]

        # vertex
        pred_vtx_sub_world = self.xyz_regressor(vtx_features_2.transpose(0, 1))  # batch_size X num_vertices X 3
        # coarse-to-fine mesh upsampling
        pred_vtx_full_world = self.upsample(pred_vtx_sub_world.transpose(1, 2)).transpose(1, 2)  # batch_size X 1220 X 3

        # calc vertex in camera space : sub-vertices
        ones = torch.ones([batch_size, pred_vtx_sub_world.shape[1], 1], device=device)
        pred_vtx_sub_world_homo = torch.cat([pred_vtx_sub_world, ones], dim=2)  # [B,305,4]
        pred_vtx_sub_cam = torch.bmm(pred_vtx_sub_world_homo, pred_R_t)[:, :, :3]  # [B,305,3]
        # calc vertex in camera space : full-vertices
        ones = torch.ones([batch_size, pred_vtx_full_world.shape[1], 1], device=device)
        pred_vtx_full_world_homo = torch.cat([pred_vtx_full_world, ones], dim=2)  # [B,1220,4]
        pred_vtx_full_cam = torch.bmm(pred_vtx_full_world_homo, pred_R_t)[:, :, :3]  # [B,1220,3]

        out = {}
        out['R_t'] = pred_R_t
        out['vtx_sub_world'] = pred_vtx_sub_world
        out['vtx_full_world'] = pred_vtx_full_world
        out['vtx_sub_cam'] = pred_vtx_sub_cam
        out['vtx_full_cam'] = pred_vtx_full_cam
        # out['pred_3d_vertices_intermediate'] = pred_3d_vertices_intermediate
        # out['pred_3d_vertices_fine'] = pred_3d_vertices_fine

        # # (optional) regress smpl parameters
        # if self.args.use_smpl_param_regressor:
        #     pred_rotmat, pred_betas = self.smpl_parameter_regressor(pred_3d_vertices_intermediate.clone().detach())
        #     out['pred_rotmat'] = pred_rotmat
        #     out['pred_betas'] = pred_betas

        return out

class FastMETRO_Hand_Network(nn.Module):
    """FastMETRO for 3D hand mesh reconstruction from a single RGB image"""
    def __init__(self, args, backbone, mesh_sampler, num_joints=21, num_vertices=195):
        """
        Parameters:
            - args: Arguments
            - backbone: CNN Backbone used to extract image features from the given image
            - mesh_sampler: Mesh Sampler used in the coarse-to-fine mesh upsampling
            - num_joints: The number of joint tokens used in the transformer decoder
            - num_vertices: The number of vertex tokens used in the transformer decoder
        """
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.mesh_sampler = mesh_sampler
        self.num_joints = num_joints
        self.num_vertices = num_vertices
        
        # the number of transformer layers
        if 'FastMETRO-S' in args.model_name:
            num_enc_layers = 1
            num_dec_layers = 1
        elif 'FastMETRO-M' in args.model_name:
            num_enc_layers = 2
            num_dec_layers = 2
        elif 'FastMETRO-L' in args.model_name:
            num_enc_layers = 3
            num_dec_layers = 3
        else:
            assert False, "The model name is not valid"
    
        # configurations for the first transformer
        self.transformer_config_1 = {"model_dim": args.model_dim_1, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead, 
                                     "feedforward_dim": args.feedforward_dim_1, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # configurations for the second transformer
        self.transformer_config_2 = {"model_dim": args.model_dim_2, "dropout": args.transformer_dropout, "nhead": args.transformer_nhead,
                                     "feedforward_dim": args.feedforward_dim_2, "num_enc_layers": num_enc_layers, "num_dec_layers": num_dec_layers, 
                                     "pos_type": args.pos_type}
        # build transformers
        self.transformer_1 = build_transformer(self.transformer_config_1)
        self.transformer_2 = build_transformer(self.transformer_config_2)
        # dimensionality reduction
        self.dim_reduce_enc_cam = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_enc_img = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        self.dim_reduce_dec = nn.Linear(self.transformer_config_1["model_dim"], self.transformer_config_2["model_dim"])
        
        # token embeddings
        self.cam_token_embed = nn.Embedding(1, self.transformer_config_1["model_dim"])
        self.joint_token_embed = nn.Embedding(self.num_joints, self.transformer_config_1["model_dim"])
        self.vertex_token_embed = nn.Embedding(self.num_vertices, self.transformer_config_1["model_dim"])
        # positional encodings
        self.position_encoding_1 = build_position_encoding(pos_type=self.transformer_config_1['pos_type'], hidden_dim=self.transformer_config_1['model_dim'])
        self.position_encoding_2 = build_position_encoding(pos_type=self.transformer_config_2['pos_type'], hidden_dim=self.transformer_config_2['model_dim'])
        # estimators
        self.xyz_regressor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        self.cam_predictor = nn.Linear(self.transformer_config_2["model_dim"], 3)
        
        # 1x1 Convolution
        self.conv_1x1 = nn.Conv2d(args.conv_1x1_dim, self.transformer_config_1["model_dim"], kernel_size=1)

        # attention mask
        zeros_1 = torch.tensor(np.zeros((num_vertices, num_joints)).astype(bool)) 
        zeros_2 = torch.tensor(np.zeros((num_joints, (num_joints + num_vertices))).astype(bool)) 
        adjacency_indices = torch.load('./src/modeling/data/mano_195_adjmat_indices.pt')
        adjacency_matrix_value = torch.load('./src/modeling/data/mano_195_adjmat_values.pt')
        adjacency_matrix_size = torch.load('./src/modeling/data/mano_195_adjmat_size.pt')
        adjacency_matrix = torch.sparse_coo_tensor(adjacency_indices, adjacency_matrix_value, size=adjacency_matrix_size).to_dense()
        temp_mask_1 = (adjacency_matrix == 0)
        temp_mask_2 = torch.cat([zeros_1, temp_mask_1], dim=1)
        self.attention_mask = torch.cat([zeros_2, temp_mask_2], dim=0)
    
    def forward(self, images):
        device = images.device
        batch_size = images.size(0)

        # preparation
        cam_token = self.cam_token_embed.weight.unsqueeze(1).repeat(1, batch_size, 1) # 1 X batch_size X 512 
        jv_tokens = torch.cat([self.joint_token_embed.weight, self.vertex_token_embed.weight], dim=0).unsqueeze(1).repeat(1, batch_size, 1) # (num_joints + num_vertices) X batch_size X 512
        attention_mask = self.attention_mask.to(device) # (num_joints + num_vertices) X (num_joints + num_vertices)
        
        # extract image features through a CNN backbone
        img_features = self.backbone(images) # batch_size X 2048 X 7 X 7
        _, _, h, w = img_features.shape
        img_features = self.conv_1x1(img_features).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        
        # positional encodings
        pos_enc_1 = self.position_encoding_1(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 512 
        pos_enc_2 = self.position_encoding_2(batch_size, h, w, device).flatten(2).permute(2, 0, 1) # 49 X batch_size X 128 

        # first transformer encoder-decoder
        cam_features_1, enc_img_features_1, jv_features_1 = self.transformer_1(img_features, cam_token, jv_tokens, pos_enc_1, attention_mask=attention_mask)
        
        # progressive dimensionality reduction
        reduced_cam_features_1 = self.dim_reduce_enc_cam(cam_features_1) # 1 X batch_size X 128 
        reduced_enc_img_features_1 = self.dim_reduce_enc_img(enc_img_features_1) # 49 X batch_size X 128 
        reduced_jv_features_1 = self.dim_reduce_dec(jv_features_1) # (num_joints + num_vertices) X batch_size X 128

        # second transformer encoder-decoder
        cam_features_2, _, jv_features_2 = self.transformer_2(reduced_enc_img_features_1, reduced_cam_features_1, reduced_jv_features_1, pos_enc_2, attention_mask=attention_mask) 

        # estimators
        pred_cam = self.cam_predictor(cam_features_2).view(batch_size, 3) # batch_size X 3
        pred_3d_coordinates = self.xyz_regressor(jv_features_2.transpose(0, 1)) # batch_size X (num_joints + num_vertices) X 3
        pred_3d_joints = pred_3d_coordinates[:,:self.num_joints,:] # batch_size X num_joints X 3
        pred_3d_vertices_coarse = pred_3d_coordinates[:,self.num_joints:,:] # batch_size X num_vertices(coarse) X 3
        pred_3d_vertices_fine = self.mesh_sampler.upsample(pred_3d_vertices_coarse) # batch_size X num_vertices(fine) X 3
        
        out = {}
        out['pred_cam'] = pred_cam
        out['pred_3d_joints'] = pred_3d_joints
        out['pred_3d_vertices_coarse'] = pred_3d_vertices_coarse
        out['pred_3d_vertices_fine'] = pred_3d_vertices_fine

        return out
