import os
import sys
import pdb
import cv2
import time
import copy
import numpy as np
import torch
from datetime import datetime
from options.train_options import TrainOptions
from data import create_dataset
from pdb import set_trace
from util.pkl import write_pkl, read_pkl
from models.trg_model import load_trg
from models.trg_model.core.cfgs import parse_args
from models.trg_loss import keypoint_3d_loss, keypoint_2d_loss, rotation_matrix_loss, mesh_edge_loss, geodesic_loss
import os
from util.logger import setup_logger
from datetime import datetime
# from util.metric import calc_rotation_mae, calc_trans_mae, calc_keypoint_mae, calc_mean_std_dev_face_mesh_diff
from util.metric import *
from collections import defaultdict
# from util.face_deformnet_utils import compute_sRT_errors
import pprint
import json
import shutil
from util.metric import get_chordal_distance_from_rotmat, chordal2angular, radian2degree
import random

def clean_checkpoint(checkpoints_dir, best_epoch, cur_epoch):
    checkpoints = os.listdir(checkpoints_dir)
    for chp in checkpoints:
        checkpoint_path = os.path.join(checkpoints_dir, chp)
        # checkpoint dir 인지 확인
        if chp.startswith('checkpoint-') and os.path.isdir(checkpoint_path):
            chp_epoch = int(chp.split('-')[1])

            # Don't remove when satisfy condition
            if chp_epoch == best_epoch or chp_epoch == cur_epoch:
                pass
            # Remove when it cannot satisfy condition
            else:
                files = os.listdir(checkpoint_path)
                for file in files:
                    if file.startswith('metric'):
                        continue
                    remove_target = os.path.join(checkpoint_path, file)
                    # shutil.rmtree(remove_target)
                    os.remove(remove_target)
                    print(f'remove {remove_target}')

def save_checkpoint(model, save_dir, epoch, num_trial=10):
    checkpoint_dir = os.path.join(save_dir, 'checkpoint-{}'.format(
        epoch))
    # if not is_main_process():
    #     return checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    for _ in range(num_trial):
        try:
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'state_dict.bin'))
            print("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        print("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    ##############################################################################################
    # setting parse arguments and load dataset
    ##############################################################################################
    train_opt = TrainOptions().parse()   # get training options
    if len(train_opt.gpu_ids)>1:
        train_opt.batch_size = train_opt.batch_size * len(train_opt.gpu_ids)
        print(
            f'▶ total train batch_size : {train_opt.batch_size}, '
            f'and distribute into {int(train_opt.batch_size / len(train_opt.gpu_ids))}')

    train_dataset = create_dataset(train_opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.
    
    val_opt = copy.deepcopy(train_opt)
    val_opt.isTrain = False
    val_opt.batch_size = 64
    val_opt.num_thread = 8
    val_opt.serial_batches = True

    # arkit val dataset
    val_opt.dataset_mode = 'arkit'
    arkit_val_dataset = create_dataset(val_opt)

    # biwi val dataset
    val_opt.dataset_mode = 'biwi'
    biwi_val_dataset = create_dataset(val_opt)

    print('The number of training images = %d' % train_dataset_size)
    print('The number of arkit val images = %d\n' % len(arkit_val_dataset))
    print('The number of biwi val images = %d\n' % len(biwi_val_dataset))

    ##############################################################################################
    # logger
    ##############################################################################################
    global logger
    output_dir = os.path.join(train_opt.checkpoints_dir, train_opt.name)
    logger = setup_logger(f"{train_opt.name}", output_dir, 0)
    logger.info("Using {} GPUs".format(train_opt.gpu_ids))

    ##############################################################################################
    # load model
    ##############################################################################################
    device = 'cuda'
    init_face_pose_dict = read_pkl('data/init_vtx_cam.pkl')
    init_face, init_pose = init_face_pose_dict['t_vtx_1220'], init_face_pose_dict['R_t']

    # get config
    config = parse_args(train_opt.model_cfg_path)

    # load model
    model = load_trg(config, init_face, init_pose).to(device)

    ##############################################################################################
    # Model on multi-GPU
    ##############################################################################################
    if len(train_opt.gpu_ids) > 1:
        print(f'Use MULTI-GPU : {train_opt.gpu_ids}')
        model = torch.nn.DataParallel(model, device_ids=train_opt.gpu_ids)

    ##############################################################################################
    # subsample matrix
    ##############################################################################################
    subsample = read_pkl('data/arkit_subsample.pkl')

    ##############################################################################################
    # load triangles index
    ##############################################################################################
    mesh_faces = np.load('./data/triangles.npy') # [2304, 3]
    mesh_faces = mesh_faces.astype(np.int64)
    tri_index = torch.tensor(mesh_faces, dtype=torch.long).to(device)
    mesh_faces = torch.from_numpy(mesh_faces).to(device)

    ##############################################################################################
    # load lmk 68 index
    ##############################################################################################
    npz_path = 'npy/kpt_ind.npy'
    lmk68_idx = np.load(npz_path)
    lmk68_idx = lmk68_idx.tolist()

    kpt5_idx = [39, 42, 33, 48, 54]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_opt.lr, betas=(train_opt.beta1, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_opt.n_epochs)

    total_iters = 0                # the total number of training iterations
    times = []

    # define loss function
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_rotmat = torch.nn.MSELoss(reduction='none').cuda(device)
    criterion_edge = torch.nn.L1Loss(reduction='none').cuda(device)

    checkpoint_remover = {'best_err': 1000.0, 'best_epoch': -1, 'cur_epoch': -1}
    ##############################################################################################
    # Start training
    ##############################################################################################
    for epoch in range(train_opt.epoch_count, train_opt.n_epochs + train_opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        model.train()
        for i, data in enumerate(train_dataset):
            if i == 5 and train_opt.debugging:
                break
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % train_opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data['img'].size(0)
            total_iters += batch_size
            epoch_iter += batch_size

            optimize_start_time = time.time()

            ##############################################################################################
            # load data
            ##############################################################################################
            img = data['img'].to(device) # [B,3,H,W]
            crop_img_size = img.shape[2]
            gt_face_world = data['vtx_world'].to(device) # [B,1220,4]
            gt_face_cam = data['vtx_cam'].to(device) # [B,1220,4]
            gt_face_img = data['vtx_img'].to(device) # [B,1220,4]
            has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device) # [B]
            has_rot = data['has_rot'].squeeze(-1).to(device) # [B]
            # has_transl = data['has_transl'].squeeze(-1).to(device) # [B]
            gt_extrinsic = data['R_t'].to(device) # [B,4,4]
            # intrinsic = data['K'].to(device) # [B,4,3]
            intrinsic_crop = data['K_crop'].to(device) # [B,4,3]
            intrinsic_org = data['K'].to(device) # [B,4,3]
            bbox_info = data['bbox_info'].to(device) # [B,7]
            gt_lmk68_crop = data['lmk68_crop'].to(device) # [B,68,3], normalized (-1~1)
            ##############################################################################################
            # forward
            ##############################################################################################
            preds_dict, _ = model(img, intrinsic_crop, bbox_info)
            ##############################################################################################
            # loss
            ##############################################################################################
            loss = 0
            loop_loss = []
            vtx_cam_losses = 0
            vtx_world_looses = 0
            vtx_img_looses = 0
            rotmat_losses = 0
            edge_length_losses = 0
            lmk68_losses = 0
            lmk5_2d_losses = 0
            # transl_losses = 0

            len_loop = len(preds_dict['output'])
            for loop_i in range(1, len_loop):
                # loop_i: {1,2,3}
                preds = preds_dict['output'][loop_i]
                pred_R_t, pred_face_world, pred_face_cam = \
                    preds['pred_R_t'], \
                    preds['pred_face_world'], \
                    preds['pred_face_cam']

                # current loop loss weight
                cur_loop_weight = train_opt.loop_loss_weight ** (len_loop - 1 - loop_i)

                # coordi loss
                loss_vtx_world = keypoint_3d_loss(criterion_3d_keypoints, pred_face_world, gt_face_world, has_3d_vtx, device) * train_opt.vertex_world_weight
                loss_vtx_cam = keypoint_3d_loss(criterion_3d_keypoints, pred_face_cam, gt_face_cam, has_3d_vtx, device) * train_opt.vertex_cam_weight

                # 2d vtx loss
                pred_face_img = torch.bmm(pred_face_cam, intrinsic_org[:,:3,:3]) # [B,305,3]
                z = pred_face_img[:, :, [2]]
                pred_face_img = pred_face_img / z  # divide by z
                pred_face_img = pred_face_img[:, :, :2]  # [B,nv,2]
                loss_vtx_img = keypoint_2d_loss(criterion_2d_keypoints, pred_face_img, gt_face_img) * train_opt.vertex_img_weight

                ####################################################################################
                # full size img space -> cropped img space -> normalize coordinate [-1~1]
                ####################################################################################
                pred_lmk2d_full = pred_face_img[:, lmk68_idx, :].clone()  # [B, 68, 2]
                pred_pt5_full = pred_lmk2d_full[:, kpt5_idx, :].clone()
                pred_pt5_crop = pred_pt5_full.clone()
                pred_pt5_crop[:, :, 0] = pred_pt5_crop[:, :, 0] - bbox_info[:, [0]]  # x
                pred_pt5_crop[:, :, 1] = pred_pt5_crop[:, :, 1] - bbox_info[:, [1]]  # y

                # resize img
                bbox_size = bbox_info[:, 2] - bbox_info[:, 0]  # [B,5]
                resize = (crop_img_size / bbox_size)  # [B]
                resize = resize[:, None, None]
                pred_pt5_crop = pred_pt5_crop * resize

                # normalize
                pred_pt5_crop_norm = pred_pt5_crop / crop_img_size * 2 - 1  # [0~191] -> [-1~1]

                # landmark loss
                loss_lmk5_2d = keypoint_2d_loss(criterion_2d_keypoints, pred_pt5_crop_norm, gt_lmk68_crop[:, kpt5_idx, :]) * train_opt.lmk5_2d_loss_weight

                # rotation loss
                loss_rotmat = rotation_matrix_loss(criterion_rotmat, preds['pred_R_t'][:,:3,:3], gt_extrinsic[:,:3,:3], has_rot, device).mean() * train_opt.rotmat_loss_weight

                # last regressor
                if loop_i == len_loop - 1:
                    loss_edge = mesh_edge_loss(criterion_edge, pred_face_world, gt_face_world, tri_index, has_3d_vtx) * train_opt.edge_weight
                else:
                    loss_edge = torch.zeros(1, dtype=torch.float32, device=device)

                cur_loop_loss = loss_vtx_world + loss_vtx_cam + loss_rotmat + loss_vtx_img + loss_edge + loss_lmk5_2d

                # record
                vtx_world_looses = vtx_world_looses + loss_vtx_world.item() * cur_loop_weight
                vtx_cam_losses = vtx_cam_losses + loss_vtx_cam.item() * cur_loop_weight
                rotmat_losses = rotmat_losses + loss_rotmat.item() * cur_loop_weight
                vtx_img_looses = vtx_img_looses + loss_vtx_img.item() * cur_loop_weight
                edge_length_losses = edge_length_losses + loss_edge.item() * cur_loop_weight
                lmk5_2d_losses = lmk5_2d_losses + loss_lmk5_2d.item() * cur_loop_weight

                # apply small loss as low resolution
                cur_loop_loss = cur_loop_loss * cur_loop_weight
                loss = loss + cur_loop_loss

                loop_loss.append(cur_loop_loss.item())

            # mask loss
            pred_lmk68 = preds_dict['pred_lmk68'][-1]  # [B,68,2]
            loss_lmk68 = keypoint_2d_loss(criterion_2d_keypoints, pred_lmk68, gt_lmk68_crop) * train_opt.lmk68_loss_weight
            loss = loss + loss_lmk68
            lmk68_losses = loss_lmk68.item()

            ##############################################################################################
            # backward
            ##############################################################################################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if len(train_opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            ##############################################################################################
            # logger
            ##############################################################################################
            if epoch_iter % (train_opt.display_freq * train_opt.batch_size) == 0:
                now = datetime.now()
                cur_time = now.strftime('%m-%d %H:%M:%S')
                train_percentage = i / len(train_dataset.dataloader) * 100
                total_epoch = train_opt.n_epochs + train_opt.n_epochs_decay
                log = f"[{cur_time}] [{epoch}/{total_epoch}|{train_percentage:.2f}%] loss : {loss.item() * 1000.:.2f} "
                for loop_i in range(len(loop_loss)):
                    log = log + f"| loop_loss[{loop_i}] : {loop_loss[loop_i]:.4f} "
                log = log + f"| wrd : {vtx_world_looses:.4f} "
                log = log + f"| cam : {vtx_cam_losses:.4f} "
                log = log + f"| im : {vtx_img_looses:.4f} "
                log = log + f"| r : {rotmat_losses:.4f} "
                log = log + f"| lmk : {lmk68_losses:.4f} "
                log = log + f"| ed : {edge_length_losses:.4f} "
                log = log + f"| lmk5_2d : {lmk5_2d_losses:.4f} "
                # log = log + f"| transl_losses : {transl_losses:.4f} "

                log = log + f"| lr[0] : {optimizer.param_groups[0]['lr']} "
                logger.info(log)

            iter_data_time = time.time()

        logger.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.n_epochs + train_opt.n_epochs_decay, time.time() - epoch_start_time))
        ##############################################################################################
        # lr scheduler
        ##############################################################################################
        lr_scheduler.step()

        ##############################################################################################
        # Evaluate per epoch
        ##############################################################################################
        test_freq = 1
        # result dict
        result = defaultdict(list)
        prediction = defaultdict(list)
        if epoch % test_freq == 0:
            logger.info('===================== evaluation epoch %d begin =====================' % epoch)
            t_start = time.time()
            num_batch = len(arkit_val_dataset.dataloader)

            model.eval()
            for j, data in enumerate(arkit_val_dataset):
                with torch.no_grad():
                    if j == 5 and val_opt.debugging:
                        break
                    # load data
                    img = data['img'].to(device)
                    batch_size = img.shape[0]

                    gt_face_world = data['vtx_world'].to(device)  # [B,1220,4]
                    gt_face_cam = data['vtx_cam'].to(device)  # [B,1220,4]
                    has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
                    gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
                    intrinsic_crop = data['K_crop'].to(device)  # [B,4,3]
                    bbox_info = data['bbox_info'].to(device)  # [B,5]

                    # forward
                    preds_dict, _ = model(img, intrinsic_crop, bbox_info)
                    preds = preds_dict['output'][-1]

                    pred_face_world, pred_R_t, pred_face_cam = \
                        preds['pred_face_world'], \
                        preds['pred_R_t'], \
                        preds['pred_face_cam']

                    # save prediction
                    prediction["pred_face_world"].append(pred_face_world.cpu().numpy())
                    prediction["pred_R_t"].append(pred_R_t.cpu().numpy())

                    ##############################################################################################
                    # yaw, pitch, roll, tx, ty, tz, ADD
                    ##############################################################################################
                    gt_R_t_tmp = gt_extrinsic.transpose(1, 2).cpu().numpy()  # [B,4,4]
                    gt_R = gt_R_t_tmp[:, :3, :3]  # [B,3,3]
                    gt_t = gt_R_t_tmp[:, :3, 3]  # [B,3]
                    pred_R_t_tmp = pred_R_t.transpose(1, 2).cpu().numpy()  # [B,4,4]
                    pred_R = pred_R_t_tmp[:, :3, :3]  # [B,3,3]
                    pred_t = pred_R_t_tmp[:, :3, 3]  # [B,3]

                    yaw_mae, pitch_mae, roll_mae = calc_rotation_mae(pred_R, gt_R)
                    tx_mae, ty_mae, tz_mae = calc_trans_mae(pred_t, gt_t)

                    _, ge_err = calc_geodesic_error(pred_R, gt_R)
                    result['ge_err'].append(ge_err)  # batch_size

                    result['yaw_mae'].append(yaw_mae)
                    result['pitch_mae'].append(pitch_mae)
                    result['roll_mae'].append(roll_mae)

                    result['tx_mae'].append(tx_mae)
                    result['ty_mae'].append(ty_mae)
                    result['tz_mae'].append(tz_mae)

                    pred_face_cam_ADD = torch.bmm(gt_face_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    gt_face_cam_ADD = torch.bmm(gt_face_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    cur_ADD = calc_keypoint_mae(pred_face_cam_ADD, gt_face_cam_ADD).mean()
                    result['ADD'].append(cur_ADD)

                    ##############################################################################################
                    # calc 3DRecon error
                    ##############################################################################################
                    pred_face_world_np = pred_face_world.cpu().numpy()  # [B,1220,3]
                    gt_face_world_np = gt_face_world[:, :, :3].cpu().numpy()  # [B,1220,3]
                    l2_norm_verts_batch = calc_keypoint_l2(pred_face_world_np, gt_face_world_np)

                    mean_error = np.mean(l2_norm_verts_batch, axis=1)  # [B]
                    # median_error = np.median(l2_norm_verts_batch, axis=1)  # [B]

                    result['3DRecon'].append(mean_error)
                    # result['3DRecon_median'].append(median_error)
                    ##############################################################################################
                    # calc face size error
                    ##############################################################################################
                    # calc face size error
                    m2mm = 1000
                    gt_tri_area = calc_mesh_area_from_vertices_batch(gt_face_world[:, :, :3] * m2mm, mesh_faces)
                    gt_face_size = torch.sum(gt_tri_area, dim=1)

                    pred_tri_area = calc_mesh_area_from_vertices_batch(pred_face_world * m2mm, mesh_faces)
                    pred_face_size = torch.sum(pred_tri_area, dim=1)

                    size_error = torch.abs(gt_face_size - pred_face_size)
                    size_error = size_error.detach().cpu().numpy()

                    result['face_size_error'].append(size_error)

                    ##############################################################################################
                    # logger
                    ##############################################################################################
                    if j % 10 == 0:
                        now = datetime.now()
                        cur_time = now.strftime('%m-%d %H:%M:%S')
                        test_percentage = j / len(arkit_val_dataset.dataloader) * 100
                        logger.info(
                            f"evaluate : [{cur_time}] [{test_percentage:.2f}%] "
                        )

            # make checkpoint dir
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            prediction = None
            pred_face_world = None
            gt = None
            gt_face = None
            ##############################################################################################
            # Summarize all metric
            ##############################################################################################
            metric = arkit_val_dataset.dataset.compute_metrics_(result)

            logger.info('\n [metric] ')
            pprint.pprint(metric)
            logger.info('Done in %.3f sec' % (time.time() - t_start))
            # record metric
            out_json_path = os.path.join(checkpoint_dir, 'metric.json')
            logger.info(f'save metric file into: {out_json_path}')
            with open(out_json_path, 'w') as fout:
                json.dump(metric, fout, indent=4, sort_keys=True)

        # ##############################################################################################
        # # Save checkpoint
        # ##############################################################################################
        # if epoch % train_opt.save_epoch_freq == 0 and epoch % test_freq == 0:  # cache our model every <save_epoch_freq> epochs
        #     logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     checkpoint_dir = save_checkpoint(model, output_dir, epoch)
        #
        #     # update checkpoint remover
        #     checkpoint_remover['cur_epoch'] = epoch
        #     cur_error = float(metric['ADD'][:-3])
        #     if cur_error < checkpoint_remover['best_err']:
        #         checkpoint_remover['best_err'] = cur_error
        #         checkpoint_remover['best_epoch'] = epoch
        #
        #     # remove old checkpoint
        #     clean_checkpoint(output_dir, checkpoint_remover['best_epoch'], epoch)

        ##############################################################################################
        # Evaluation on BIWI dataset
        ##############################################################################################
        # result dict
        result = defaultdict(list)
        prediction = defaultdict(list)
        align_rotate = defaultdict(list)
        if epoch % test_freq == 0:
            logger.info('===================== evaluation epoch %d begin =====================' % epoch)
            t_start = time.time()
            num_batch = len(biwi_val_dataset.dataloader)

            model.eval()
            for j, data in enumerate(biwi_val_dataset):
                with torch.no_grad():
                    if j == 5 and val_opt.debugging:
                        break
                    # load data
                    img = data['img'].to(device)
                    batch_size = img.shape[0]

                    gt_face_world = data['vtx_world'].to(device)  # [B,1220,4]
                    gt_face_cam = data['vtx_cam'].to(device)  # [B,1220,4]
                    has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
                    gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
                    intrinsic_crop = data['K_crop'].to(device)  # [B,4,3]
                    bbox_info = data['bbox_info'].to(device)  # [B,5]

                    # forward
                    preds_dict, _ = model(img, intrinsic_crop, bbox_info)
                    preds = preds_dict['output'][-1]

                    pred_face_world, pred_R_t, pred_face_cam = \
                        preds['pred_face_world'], \
                        preds['pred_R_t'], \
                        preds['pred_face_cam']

                    # save prediction
                    prediction["pred_face_world"].append(pred_face_world.cpu().numpy())
                    prediction["pred_R_t"].append(pred_R_t.cpu().numpy())

                    ##############################################################################################
                    # yaw, pitch, roll, tx, ty, tz, ADD
                    ##############################################################################################
                    gt_R_t_tmp = gt_extrinsic.transpose(1, 2).cpu().numpy()  # [B,4,4]
                    gt_R = gt_R_t_tmp[:, :3, :3]  # [B,3,3]
                    gt_t = gt_R_t_tmp[:, :3, 3]  # [B,3]
                    pred_R_t_tmp = pred_R_t.transpose(1, 2).cpu().numpy()  # [B,4,4]
                    pred_R = pred_R_t_tmp[:, :3, :3]  # [B,3,3]
                    pred_t = pred_R_t_tmp[:, :3, 3]  # [B,3]

                    align_rotate['gt_R'].append(gt_R)
                    align_rotate['pred_R'].append(pred_R)

                    yaw_mae, pitch_mae, roll_mae = calc_rotation_mae(pred_R, gt_R)
                    tx_mae, ty_mae, tz_mae = calc_trans_mae(pred_t, gt_t)

                    _, ge_err = calc_geodesic_error(pred_R, gt_R)
                    result['ge_err'].append(ge_err)  # batch_size

                    result['yaw_mae'].append(yaw_mae)
                    result['pitch_mae'].append(pitch_mae)
                    result['roll_mae'].append(roll_mae)

                    result['tx_mae'].append(tx_mae)
                    result['ty_mae'].append(ty_mae)
                    result['tz_mae'].append(tz_mae)

                    pred_face_cam_ADD = torch.bmm(gt_face_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    gt_face_cam_ADD = torch.bmm(gt_face_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    cur_ADD = calc_keypoint_mae(pred_face_cam_ADD, gt_face_cam_ADD).mean()
                    result['ADD'].append(cur_ADD)

                    ##############################################################################################
                    # calc 3DRecon error
                    ##############################################################################################
                    result['3DRecon'].append(np.zeros(1, dtype=np.float32))

                    ##############################################################################################
                    # logger
                    ##############################################################################################
                    if j % 10 == 0:
                        now = datetime.now()
                        cur_time = now.strftime('%m-%d %H:%M:%S')
                        test_percentage = j / len(biwi_val_dataset.dataloader) * 100
                        logger.info(
                            f"evaluate : [{cur_time}] [{test_percentage:.2f}%] "
                        )

            prediction = None
            pred_face_world = None
            gt = None
            gt_face = None

            ##############################################################################################
            # Summarize all metric
            ##############################################################################################
            metric = biwi_val_dataset.dataset.compute_metrics_(result)

            logger.info('\n [metric] ')
            pprint.pprint(metric)
            logger.info('Done in %.3f sec' % (time.time() - t_start))
            # record metric
            out_json_path = os.path.join(checkpoint_dir, 'metric_biwi.json')
            logger.info(f'save metric file into: {out_json_path}')
            with open(out_json_path, 'w') as fout:
                json.dump(metric, fout, indent=4, sort_keys=True)

        ##############################################################################################
        # Save checkpoint
        ##############################################################################################
        if epoch % train_opt.save_epoch_freq == 0 and epoch % test_freq == 0:  # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            checkpoint_dir = save_checkpoint(model, output_dir, epoch)

            # update checkpoint remover
            checkpoint_remover['cur_epoch'] = epoch
            cur_error = float(metric['ADD'][:-3])
            if cur_error < checkpoint_remover['best_err']:
                checkpoint_remover['best_err'] = cur_error
                checkpoint_remover['best_epoch'] = epoch

            # remove old checkpoint
            clean_checkpoint(output_dir, checkpoint_remover['best_epoch'], epoch)