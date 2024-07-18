import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torchgeometry as tgm

from pdb import set_trace

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
    abs_rotmat = rel_aa_to_abs_rotmat(rel_aa, kinematic_tree)
    abs_aa = rotmat_to_aa(abs_rotmat)
    return abs_aa

def rel_aa_to_abs_rotmat(rel_aa, kinematic_tree):
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

def rel_rotmat_to_abs_rotmat(rel_rotmat, parents=None):
    """
        only for SMPL
        rel_rotmat: [B,24,3,3]
        parents : list or torch.tensor, torch.int64
        global_rotmat: [B,24,3,3]
    """
    if parents is None:
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    if not isinstance(parents, torch.Tensor):
        parents = torch.tensor(parents, dtype=torch.int64)

    transform_chain = [rel_rotmat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                rel_rotmat[:, i])
        transform_chain.append(curr_res)

    abs_rotmat = torch.stack(transform_chain, dim=1)

    return abs_rotmat

def abs_rotmat_to_rel_rotmat(abs_rot, smpl=True, parent=None, child=None):
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

def abs_rotmat_to_rel_aa(abs_rot, smpl=True, parent=None, child=None):
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

    rel_rot = abs_rotmat_to_rel_rotmat(abs_rot, smpl=smpl, parent=parent, child=child) # [B,24,3,3]

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

def world_to_img_numpy(point_world_homo, extrinsics, intrinsics, full_img_height=800, bbox=None, do_crop=False, cropped_img_size=192):
    """
    point_world_homo: np.ndarray, size:[n_points, 4]
    extrinsics: np.ndarray, size:[4,4]. It is transposed matrix. extrinsics[3,:3] is translation
    intrinsics: np.ndarray, size:[4,4]. It is transposed matrix.
    bbox: np.ndarray, size: [4] or [5]. [L,T,R,B] or [L,T,R,B, focal_length]
    do_crop: Boolean. if do_crop == True and bbox is not None, Then the results are cropped image space coordinates.

    return np.ndarray, size: [n_points, 2]
    """
    ####################################################################################
    # World -> full size image space
    ####################################################################################
    vtx_cam_homo = (point_world_homo @ extrinsics)  # [1220,4]
    vtx_img_homo = (vtx_cam_homo @ intrinsics)  # [1220,4]

    z = vtx_img_homo[:, [3]]  # [nv,1]
    points_img_full = vtx_img_homo / z  # [nv,4]
    points_img_full[:, 2] = -z[:, 0]  # [nv]
    points_img_full = points_img_full[:, :3]
    points_img_full[:, 1] = full_img_height - points_img_full[:, 1]
    points_img_full = points_img_full[:, :2]  # [nv,2]

    if bbox is not None and do_crop:
        ####################################################################################
        # full size image space -> cropped image space
        ####################################################################################
        # crop bbox
        points_img_crop = points_img_full.copy()  # [nv, 2]
        points_img_crop[:, 0] = points_img_full[:, 0] - bbox[[0]]  # x
        points_img_crop[:, 1] = points_img_full[:, 1] - bbox[[1]]  # y

        # resize img
        bbox_size = bbox[2] - bbox[0]
        resize = (cropped_img_size / bbox_size)  # [1]

        points_img_crop = points_img_crop * resize # [nv,2]
        return points_img_full, points_img_crop

    else:
        return points_img_full, None

def cam_to_img_numpy(point_cam_homo, intrinsics, full_img_height=800, bbox=None, do_crop=False, cropped_img_size=192):
    """
    point_world_homo: np.ndarray, size:[n_points, 4]
    extrinsics: np.ndarray, size:[4,4]. It is transposed matrix. extrinsics[3,:3] is translation
    intrinsics: np.ndarray, size:[4,4]. It is transposed matrix.
    bbox: np.ndarray, size: [4] or [5]. [L,T,R,B] or [L,T,R,B, focal_length]
    do_crop: Boolean. if do_crop == True and bbox is not None, Then the results are cropped image space coordinates.

    return np.ndarray, size: [n_points, 2]
    """
    ####################################################################################
    # World -> full size image space
    ####################################################################################
    vtx_img_homo = (point_cam_homo @ intrinsics)  # [1220,4]

    z = vtx_img_homo[:, [3]]  # [nv,1]
    points_img_full = vtx_img_homo / z  # [nv,4]
    points_img_full[:, 2] = -z[:, 0]  # [nv]
    points_img_full = points_img_full[:, :3]
    points_img_full[:, 1] = full_img_height - points_img_full[:, 1]
    points_img_full = points_img_full[:, :2]  # [nv,2]

    if bbox is not None and do_crop:
        ####################################################################################
        # full size image space -> cropped image space
        ####################################################################################
        # crop bbox
        points_img_crop = points_img_full.copy()  # [nv, 2]
        points_img_crop[:, 0] = points_img_full[:, 0] - bbox[[0]]  # x
        points_img_crop[:, 1] = points_img_full[:, 1] - bbox[[1]]  # y

        # resize img
        bbox_size = bbox[2] - bbox[0]
        resize = (cropped_img_size / bbox_size)  # [1]

        points_img_crop = points_img_crop * resize # [nv,2]
        return points_img_full, points_img_crop

    else:
        return points_img_full, None

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def create_2d_gaussian_heatmaps(points, shape, sigma=2.5):
    """
    Create a 2D Gaussian heatmap for given 2D points.

    Parameters:
        points (torch.tensor): A tensor of shape [number_points, 2] containing (x, y) coordinates of 2D points.
        shape (tuple): A tuple containing the height and width of the heatmap.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.tensor: A tensor representing the Gaussian heatmap for each point.
                      Shape: [number_points, height, width]
    """
    height, width = shape
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    x_grid = x_grid.float()
    y_grid = y_grid.float()

    x_points, y_points = points[:, 0].view(-1, 1, 1), points[:, 1].view(-1, 1, 1)

    exponent = -((x_grid - x_points) ** 2 + (y_grid - y_points) ** 2) / (2 * sigma ** 2)
    heatmap = torch.exp(exponent)

    return heatmap

def create_batch_2d_gaussian_heatmaps(batch_points, shape, sigma=2.5):
    """
    Create a 2D Gaussian heatmap for a batch of given 2D points.

    Parameters:
        batch_points (torch.tensor): A tensor of shape [batch_size, number_points, 2] containing (x, y) coordinates of 2D points.
        shape (tuple): A tuple containing the height and width of the heatmap.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.tensor: A tensor representing the Gaussian heatmap for each point in each batch.
                      Shape: [batch_size, number_points, height, width]
    """
    batch_size, number_points, _ = batch_points.shape
    height, width = shape
    try:
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    except TypeError:
        y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))

    x_grid = x_grid.to(batch_points.device).float()
    y_grid = y_grid.to(batch_points.device).float()

    x_points = batch_points[:, :, 0].view(batch_size, number_points, 1, 1)
    y_points = batch_points[:, :, 1].view(batch_size, number_points, 1, 1)

    exponent = -((x_grid - x_points) ** 2 + (y_grid - y_points) ** 2) / (2 * sigma ** 2)
    heatmap = torch.exp(exponent)

    return heatmap

def soft_argmax2d(heatmap):
    """
    Compute 2D soft-argmax of a given heatmap.
    :param heatmap: input heatmaps of size (batch_size, num_keypoints, height, width)
    :return: 2D keypoints of size (batch_size, num_keypoints, 2)
    """
    batch_size, num_keypoints, height, width = heatmap.shape

    # Create coordinates grid
    x_coords = torch.arange(width).view(1, 1, 1, width).type_as(heatmap)
    y_coords = torch.arange(height).view(1, 1, height, 1).type_as(heatmap)
    # Compute softmax over the spatial extent of the heatmap
    heatmap = torch.nn.functional.softmax(heatmap.view(batch_size, num_keypoints, -1), dim=2).view_as(heatmap)
    # Compute soft-argmax
    keypoints_x = torch.sum(x_coords * heatmap, dim=[2, 3])
    keypoints_y = torch.sum(y_coords * heatmap, dim=[2, 3])
    # Stack and return the 2D keypoints
    keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)

    return keypoints

def euler_to_rotation_matrix_batch(roll, pitch, yaw):
    """
    roll, pitch, yaw: in radian angles.

    return
        rotation_matrix : 3x3 matrix
    """
    batch_size = roll.shape[0]

    # Roll (x-axis rotation)
    cos_r = torch.cos(roll)
    sin_r = torch.sin(roll)
    r = torch.zeros(batch_size, 3, 3, device=roll.device)
    r[:, 0, 0] = 1
    r[:, 1, 1] = cos_r
    r[:, 1, 2] = -sin_r
    r[:, 2, 1] = sin_r
    r[:, 2, 2] = cos_r

    # Pitch (y-axis rotation)
    cos_p = torch.cos(pitch)
    sin_p = torch.sin(pitch)
    p = torch.zeros(batch_size, 3, 3, device=pitch.device)
    p[:, 0, 0] = cos_p
    p[:, 0, 2] = sin_p
    p[:, 1, 1] = 1
    p[:, 2, 0] = -sin_p
    p[:, 2, 2] = cos_p

    # Yaw (z-axis rotation)
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    y = torch.zeros(batch_size, 3, 3, device=yaw.device)
    y[:, 0, 0] = cos_y
    y[:, 0, 1] = -sin_y
    y[:, 1, 0] = sin_y
    y[:, 1, 1] = cos_y
    y[:, 2, 2] = 1

    # Multiply in the ZYX order
    rotation_matrix = y.bmm(p).bmm(r)
    return rotation_matrix

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    tested
    Convert roll, pitch, yaw to rotation matrix
    """
    # Create rotation matrix from roll
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Create rotation matrix from pitch
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # Create rotation matrix from yaw
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine individual rotation matrices
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotation_matrix_to_euler_angles(R):
    """
    tested.
    Converts a 3x3 rotation matrix to ZYX Euler angles.

    Args:
        R (numpy.array): 3x3 rotation matrix

    Returns:
        tuple: (yaw, pitch, roll) Euler angles in radians
    """

    # Ensure the matrix is square
    assert R.shape == (3, 3)

    # Calculate yaw
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Calculate pitch
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))

    # Calculate roll
    roll = np.arctan2(R[2, 1], R[2, 2])

    return yaw, pitch, roll

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


def uniform_sampled_data(data, num_sample=10000, percentage=1.0):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    min_data = np.min(data)
    max_data = np.max(data)

    # re-setting range
    weight = (1 - percentage) * 0.5  # 0.1
    min_data = min_data * (1-weight) + max_data * weight
    max_data = min_data * (weight) + max_data * (1-weight)

    uniform_keys = np.random.uniform(min_data, max_data, num_sample)

    sampled_index = []
    for key in uniform_keys:
        diff = np.abs(data - key)
        closest_index = np.argmin(diff)

        sampled_index.append(closest_index)

    sampled_index = np.array(sampled_index, dtype=np.uint64)
    sampled_data = data[sampled_index]

    return sampled_index, sampled_data

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