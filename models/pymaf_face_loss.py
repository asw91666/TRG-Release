import torch
import torch.nn as nn
import numpy as np
from pdb import set_trace

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.

        criterion_keypoints : torch.nn.L1loss(), torch.nn.L2loss(), ...
        pred_keypoints_3d : tensor, [B,n_kp,3]
        gt_keypoints_3d : tensor, [B,n_kp,4], last dim : [x,y,z,conf]
        has_pose_3d : tensor, [B] : is 3d annot available?
    """
    # shape of gt_keypoints_3d: Batch_size X 14 X 4 (last for confidence)
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
    if len(gt_keypoints_3d) > 0:
        return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
    else:
        return torch.FloatTensor(1).fill_(0.).to(device) # 0

def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    # shape of gt_keypoints_2d: batch_size X 14 X 3 (last for visibility)
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    return loss

def rotation_matrix_loss(criterion_rotation, pred_rotmat, gt_rotmat, has_gt, device):
    """
    Compute smpl parameter loss if smpl annotations are available.
    """
    pred_rotmat = pred_rotmat[has_gt == 1].view(-1, 3, 3)
    gt_rotmat = gt_rotmat[has_gt == 1].view(-1, 3, 3)
    # gt_betas_with_shape = gt_betas[has_gt == 1]
    if len(gt_rotmat) > 0:
        loss = criterion_rotation(pred_rotmat, gt_rotmat)
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(device)
    return loss

def mesh_edge_loss(criterion_edge, pred_verts, gt_verts, tri_index, has_gt):
    # edge length loss
    gt_verts = gt_verts[has_gt == 1]
    pred_verts = pred_verts[has_gt == 1]

    bone_losses = []
    for idx in range(3):
        pred_verts_x = pred_verts[:, tri_index[:, idx % 3], :]
        pred_verts_y = pred_verts[:, tri_index[:, (idx + 1) % 3], :]
        label_verts_x = gt_verts[:, tri_index[:, idx % 3], :]
        label_verts_y = gt_verts[:, tri_index[:, (idx + 1) % 3], :]
        dist_pred = torch.norm(pred_verts_x - pred_verts_y, p=2, dim=-1, keepdim=False)
        dist_label = torch.norm(label_verts_x - label_verts_y, p=2, dim=-1, keepdim=False)
        bone_loss = criterion_edge(dist_pred, dist_label).mean()
        bone_losses.append(bone_loss)
    loss_edge = sum(bone_losses)
    return loss_edge

if __name__ == "__main__":
    # tmp = torch.FloatTensor(1).fill_(0.)
    # print(f'tmp : {tmp}, {tmp.shape}')
    # set_trace()
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none')
    pred_keypoints_3d = torch.randn([2,1220,3])
    gt_keypoints_3d = torch.randn([2,1220,3])
    ones = torch.ones([gt_keypoints_3d.shape[0], gt_keypoints_3d.shape[1], 1])
    gt_keypoints_3d = torch.cat([gt_keypoints_3d, ones], dim=-1)
    has_pose_3d = torch.ones([gt_keypoints_3d.shape[0]])
    loss = keypoint_3d_loss(criterion_3d_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d, device='cpu')
    set_trace()