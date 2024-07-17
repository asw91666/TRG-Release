import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torchgeometry as tgm

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def rel_aa_to_abs_aa(rel_aa, kinematic_tree):
    '''

    Args:
        rel_aa : torch.tensor : [B, 72]
        kinematic_tree : (
             (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14),
            (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,15)
        )

    Returns: [B, 72]
    '''
    abs_rotmat = rel_aa_to_abs_rot(rel_aa, kinematic_tree)
    abs_aa = rotmat_to_aa(abs_rotmat)
    return abs_aa

def rel_aa_to_abs_rot(rel_aa, kinematic_tree):
    '''
    Parameters
    ----------
    rel_aa : torch.tensor : [B, 72]

    "kinematic_tree" must start from pelvis, and must have (Parent, child) forms.
    kinematic_tree : (
             (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14),
            (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,15)
        ),
        (
            (Parent, Child), (Parent, Child), (Parent, Child), ... ,(Parent, Child)
        )

    Returns
    -------
    abs_rot : [B,24,3,3]
    '''
    num_smpl_joint = 24
    batch_size = rel_aa.shape[0]
    device = rel_aa.device
    rel_aa = rel_aa.view(-1, 3) # [B*J, 3]

    # axis-angle -> rotation matrix
    rel_rot = tgm.angle_axis_to_rotation_matrix(rel_aa)[:, :3, :3] # [B*24,3,3]
    rel_rot = rel_rot.view(batch_size, num_smpl_joint, *rel_rot.shape[-2:]) # [B,24,3,3]
    # assert rel_rot.shape[1:] == (24,3,3)

    # relative rotation -> absolute rotation
    abs_rot = torch.zeros(rel_rot.shape, dtype=torch.float32, device=device) # [B,24,3,3]
    abs_rot[:, 0] = rel_rot[:, 0] # pelvis rotation is absolute rotation
    for tree in kinematic_tree:
        parent_idx, child_idx = tree[0], tree[1]
        abs_rot[:, child_idx] = torch.bmm(abs_rot[:,parent_idx], rel_rot[:, child_idx])

    return abs_rot

def abs_rot_to_rel_rot(abs_rot, smpl=True, parent=None, child=None):
    '''

    Args:
        abs_rot : torch.tensor (B,24,3,3)
        smpl:
        parent:
        child:

    Returns:
        rel_rot : torch.tensor (B,24,3,3)
    '''
    batch_size, num_joint, _, _ = abs_rot.shape
    device = abs_rot.device

    if smpl:
        # smpl format
        parent = (0, 1, 4, 7, 0, 2, 5, 8, 0, 3, 6, 9, 14, 17, 19, 21, 9, 13, 16, 18, 20, 9, 12)
        child = (1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 14, 17, 19, 21, 23, 13, 16, 18, 20, 22, 12, 15)

    rel_rot = torch.zeros(abs_rot.shape, dtype=torch.float32, device=device)  # [B,24,3,3]
    rel_rot[:, 0] = abs_rot[:, 0]
    rel_rot[:, child] = torch.linalg.solve(abs_rot[:, parent], abs_rot[:, child])

    return rel_rot

def abs_rot_to_rel_aa(abs_rot, smpl=True, parent=None, child=None):
    '''
    Parameters
    ----------
    abs_rot : torch.tensor (B,24,3,3)
    if you use smpl:
        parent = (0, 1, 4, 7, 0, 2, 5, 8, 0, 3, 6, 9, 14, 17, 19, 21, 9, 13, 16, 18, 20, 9, 12)
        child = (1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 14, 17, 19, 21, 23, 13, 16, 18, 20, 22, 12, 15)
    Returns
    -------
    rel_aa : torch.tensor (B,72)

    '''
    batch_size, num_joint, _, _ = abs_rot.shape
    device = abs_rot.device

    rel_rot = abs_rot_to_rel_rot(abs_rot, smpl=smpl, parent=parent, child=child) # [B,24,3,3]

    # rotation matrix -> axis angle
    rel_rot = rel_rot.view(-1, *rel_rot.shape[2:])  # [B*24,3,3]
    rel_rot = torch.cat([rel_rot, torch.zeros((rel_rot.shape[0], 3, 1), device=device).float()], 2)  # [B*24,3,4]
    rel_aa = tgm.rotation_matrix_to_angle_axis(rel_rot)  # [B*24, 3]
    rel_aa = rel_aa.reshape(batch_size, -1)

    return rel_aa

def rot6d_to_rotmat(x):
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def rot6d_to_aa(rot6d):
    '''
    Args:
        rot6d: batch_size X 144

    Returns:
        axis_angle: batch_size X 72
    '''
    batch_size = rot6d.shape[0]
    rotmat = rot6d_to_rotmat(rot6d)
    rotmat = torch.cat([rotmat, torch.zeros((rotmat.shape[0], 3, 1)).cuda().float()], 2)
    axis_angle = tgm.rotation_matrix_to_angle_axis(rotmat).reshape(batch_size, -1)  # [B,72]
    return axis_angle

def rotmat_to_rot6d(rotmat):
    '''

    Args:
        rotmat: torch.tensor [B,24,3,3]

    Returns:

    '''
    axis_angle = rotmat_to_aa(rotmat) # [B,72]
    rot6d = aa_to_rot6d(axis_angle) # [B,J,6]
    return rot6d

def aa_to_rot6d(axis_angle):
    '''
        :param [*,3]
        :return: pose_matrot: [B,J,6]
    '''
    batch_size = axis_angle.shape[0]
    pose_body_6d = tgm.angle_axis_to_rotation_matrix(axis_angle.reshape(-1, 3))[:, :3, :2].contiguous().view(
        batch_size, -1, 6)

    return pose_body_6d

def rotmat_to_aa(rotmat):
    '''

    Args:
        rotmat: torch.tensor [B,24,3,3]

    Returns:
        [batch_size, -1]
    '''
    assert rotmat.ndim == 4
    batch_size = rotmat.shape[0]
    # rotation matrix -> axis angle
    rotmat = rotmat.view(-1, *rotmat.shape[2:])  # [B*24,3,3]
    rotmat = torch.cat([rotmat, torch.zeros((rotmat.shape[0], 3, 1), device=rotmat.device).float()], 2)  # [B*24,3,4]
    axis_angle = tgm.rotation_matrix_to_angle_axis(rotmat)  # [B*24, 3]
    axis_angle = axis_angle.reshape(batch_size, -1) # [batch_size, 72]
    return axis_angle

def aa_to_rotmat(axis_angle):
    '''

    Args:
        axis_angle: torch.tensor [B,72]

    Returns:
        rotmat : torch.tensor [B,24,3,3]
    '''
    assert axis_angle.ndim == 2
    batch_size = axis_angle.shape[0]
    num_joint = axis_angle.shape[1] // 3 # 24
    axis_angle = axis_angle.view(-1, 3)  # [B*J, 3]

    # axis-angle -> rotation matrix
    rotmat = tgm.angle_axis_to_rotation_matrix(axis_angle)[:, :3, :3]  # [B*24,3,3]
    rotmat = rotmat.view(batch_size, num_joint, *rotmat.shape[-2:])  # [B,24,3,3]
    return rotmat

