# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
# import matplotlib.pyplot as plt

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
from models import create_model
from util.visualizer import Visualizer
from pdb import set_trace
from util.pkl import write_pkl, read_pkl
from models.pymaf_model import pymaf_face_net
from models.pymaf_model.core.cfgs import parse_args
from models.pymaf_face_loss import keypoint_3d_loss, keypoint_2d_loss, rotation_matrix_loss, mesh_edge_loss
import os
from util.logger import setup_logger
from datetime import datetime
from util.metric import calc_rotation_mae, calc_trans_mae, calc_keypoint_mae, calc_rotation_mae_aflw
from collections import defaultdict
from util.face_deformnet_utils import compute_sRT_errors
import pprint
import json
import shutil
from util.metric import get_chordal_distance_from_rotmat, chordal2angular, radian2degree
import random
from data.arkit_dataset import ARKitDataset
from data.w300lp_dataset import W300LPDataset
from data.multiple_dataset import MultipleDataset
from models.op import rotation_matrix_convert_biwi_to_aflw

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
    # Manage Randomness
    ##############################################################################################
    seed_value = 42
    # Python library
    random.seed(seed_value)
    # NumPy
    np.random.seed(seed_value)
    # PyTorch
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

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

    # biwi val dataset
    val_opt.dataset_mode = 'aflw2000'
    aflw2000_val_dataset = create_dataset(val_opt)

    print('The number of training images = %d' % train_dataset_size)
    print('The number of arkit val images = %d\n' % len(arkit_val_dataset))
    print('The number of biwi val images = %d\n' % len(biwi_val_dataset))
    print('The number of aflw2000 val images = %d\n' % len(aflw2000_val_dataset))

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
    init_vtx_cam_dict = read_pkl('data/init_vtx_cam.pkl')
    init_vtx, init_cam = init_vtx_cam_dict['t_vtx_1220'], init_vtx_cam_dict['R_t']

    # get config
    # cfg_path = 'models/pymaf_model/configs/pymaf_face_config.yaml'
    config = parse_args(train_opt.model_cfg_path)

    # load model
    model = pymaf_face_net(config, init_vtx, init_cam).to(device)

    ##############################################################################################
    # load pretrained weights
    ##############################################################################################
    logger.info(f"Is fine-tuning model? : {train_opt.fine_tune}")
    if train_opt.fine_tune:
        # For checking loaded pretrained weight1
        weight_dict = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            weight_dict[name] = param.detach().clone()

        # load weight file
        weight_file_path = os.path.join(train_opt.checkpoints_dir, train_opt.pretrained_weight_path, 'state_dict.bin')

        logger.info("Loading state dict from checkpoint {}".format(weight_file_path))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(weight_file_path, map_location=cpu_device)
        if train_opt.only_use_backbone_w:
            # pretrain backbone weight 만 사용하고 싶을 때
            for key in list(state_dict.keys()):
                new_key = key.replace("feature_extractor.", "")
                state_dict[new_key] = state_dict.pop(key)

            model.feature_extractor.load_state_dict(state_dict, strict=False)
        else:
            # pretrain model weight 전체를 사용하고 싶을 때
            model.load_state_dict(state_dict, strict=False)
        logger.info(f'Load pretrained weigth from: {weight_file_path}')

        # For checking loaded pretrained weight2
        for i, (name, param) in enumerate(model.named_parameters()):
            if torch.sum(weight_dict[name] - param) == 0:
                logger.info(f'{name} : SCRATCH')
            else:
                logger.info(f'{name} : Load pretrained weight !!!')

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
    mesh_faces = np.load('./data/triangles.npy')  # [2304, 3]
    mesh_faces = mesh_faces.astype(np.int64)
    tri_index = torch.tensor(mesh_faces, dtype=torch.long).to(device)

    ##############################################################################################
    # load lmk 68 index
    ##############################################################################################
    npz_path = 'npy/kpt_ind.npy'
    lmk68_idx = np.load(npz_path)
    lmk68_idx = lmk68_idx.tolist()

    kpt5_idx = [39, 42, 33, 48, 54]

    ##############################################################################################
    # Optimizer and lr_scheduler
    ##############################################################################################
    # optimizer = torch.optim.Adam(model.parameters(), lr=train_opt.lr, betas=(train_opt.beta1, 0.999))

    if train_opt.fine_tune:
        backbone_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'feature_extractor' in name:
                param.requires_grad = True
                backbone_params.append(param)
                logger.info(f"[backbone_params] {name}: {param.requires_grad}")

            else:
                # 업데이트 안하는 나머지 모듈
                param.requires_grad = True
                decoder_params.append(param)
                logger.info(f"[decoder_params] {name}: {param.requires_grad}")

        # fine-tune encoder, decoder weights
        param_dicts = [{"params": backbone_params, 'lr': 5e-5}, # 1e-5, 1e-4
                       {"params": decoder_params, 'lr': train_opt.lr},
                       ]

        # # fine-tune only encoder weights
        # param_dicts = [{"params": backbone_params, 'lr': train_opt.lr}]
        optimizer = torch.optim.Adam(param_dicts, lr=train_opt.lr, betas=(train_opt.beta1, 0.999))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=train_opt.lr, betas=(train_opt.beta1, 0.999))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_opt.n_epochs)

    total_iters = 0                # the total number of training iterations
    times = []

    # define loss function
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_2d_keypoints = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_rotmat = torch.nn.MSELoss(reduction='none').cuda(device)
    criterion_mask = torch.nn.CrossEntropyLoss(reduction='none')
    # criterion_depth = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_scale = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_trans = torch.nn.L1Loss(reduction='none').cuda(device)
    criterion_edge = torch.nn.L1Loss(reduction='none').cuda(device)

    checkpoint_remover = {'best_err': 1000.0, 'best_epoch': -1, 'cur_epoch': -1}
    ##############################################################################################
    # Start training
    ##############################################################################################
    for epoch in range(train_opt.epoch_count, train_opt.n_epochs + train_opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        #train_dataset.set_epoch(epoch)
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

            # if len(train_opt.gpu_ids) > 0:
            #     torch.cuda.synchronize()

            optimize_start_time = time.time()

            ##############################################################################################
            # load data
            ##############################################################################################
            img = data['img'].to(device) # [B,3,H,W]
            crop_img_size = img.shape[2]
            gt_vtx_world = data['vtx_world'].to(device) # [B,1220,4]
            gt_vtx_cam = data['vtx_cam'].to(device) # [B,1220,4]
            gt_vtx_img = data['vtx_img'].to(device) # [B,1220,4]
            has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device) # [B]
            has_rot = data['has_rot'].squeeze(-1).to(device) # [B]
            gt_extrinsic = data['R_t'].to(device) # [B,4,4]
            # intrinsic = data['K'].to(device) # [B,4,3]
            intrinsic_crop = data['K_crop'].to(device) # [B,4,3]
            intrinsic_org = data['K'].to(device) # [B,4,3]
            bbox_info = data['bbox_info'].to(device) # [B,7]
            gt_lmk68_crop = data['lmk68_crop'].to(device) # [B,68,3], normalized (-1~1)
            ##############################################################################################
            # forward
            ##############################################################################################
            preds_dict, _ = model(img, intrinsic_crop, bbox_info, is_train=True)
            """
            pred.keys() : dict_keys(['pred_vtx_world', 'pred_vtx_full_world', 'pred_cam', 
                                        'pred_R_t', 'pred_vtx_cam', 'pred_vtx_full_cam'])
            """

            ##############################################################################################
            # loss
            ##############################################################################################
            loss = 0
            loop_loss = []
            vtx_cam_losses = 0
            vtx_world_looses = 0
            vtx_img_looses = 0
            rotmat_losses = 0
            # mask_losses = 0
            rot_losses = 0
            trans_losses = 0
            scale_losses = 0
            edge_length_losses = 0
            lmk68_losses = 0
            lmk2d_losses = 0

            len_loop = len(preds_dict['output'])
            for loop_i in range(1, len_loop):
                # loop_i: {1,2,3}
                preds = preds_dict['output'][loop_i]
                pred_R_t, pred_vtx_world, pred_vtx_cam = \
                    preds['pred_R_t'], \
                    preds['pred_vtx_world'], \
                    preds['pred_vtx_cam']

                # current loop loss weight
                cur_loop_weight = train_opt.loop_loss_weight ** (len_loop - 1 - loop_i)

                # coordi loss
                loss_vtx_world = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_world, gt_vtx_world, has_3d_vtx, device) * train_opt.vertex_world_weight
                loss_vtx_cam = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_cam, gt_vtx_cam, has_3d_vtx, device) * train_opt.vertex_cam_weight

                # 2d vtx loss
                pred_vtx_img = torch.bmm(pred_vtx_cam, intrinsic_org[:,:3,:3]) # [B,305,3]
                z = pred_vtx_img[:, :, [2]]
                pred_vtx_img = pred_vtx_img / z  # divide by z
                pred_vtx_img = pred_vtx_img[:, :, :2]  # [B,nv,2]
                loss_vtx_img = keypoint_2d_loss(criterion_2d_keypoints, pred_vtx_img, gt_vtx_img) * train_opt.vertex_img_weight

                ####################################################################################
                # full size img space -> cropped img space -> normalize coordinate [-1~1]
                ####################################################################################
                pred_lmk2d_full = pred_vtx_img[:, lmk68_idx, :].clone()  # [B, 68, 2]
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
                loss_lmk_2d = keypoint_2d_loss(criterion_2d_keypoints, pred_pt5_crop_norm, gt_lmk68_crop[:, kpt5_idx, :]) * train_opt.lmk_2d_loss_weight

                # rotation loss
                loss_rotmat = rotation_matrix_loss(criterion_rotmat, preds['pred_R_t'][:,:3,:3], gt_extrinsic[:,:3,:3], has_rot, device).mean() * train_opt.rotmat_loss_weight

                # last regressor
                if loop_i == len_loop - 1:
                    loss_edge = mesh_edge_loss(criterion_edge, pred_vtx_world, gt_vtx_world, tri_index, has_3d_vtx) * train_opt.edge_weight
                else:
                    loss_edge = torch.zeros(1, dtype=torch.float32, device=device)

                cur_loop_loss = loss_vtx_world + loss_vtx_cam + loss_rotmat + loss_vtx_img + loss_edge + loss_lmk_2d

                # record
                vtx_world_looses = vtx_world_looses + loss_vtx_world.item() * cur_loop_weight
                vtx_cam_losses = vtx_cam_losses + loss_vtx_cam.item() * cur_loop_weight
                rotmat_losses = rotmat_losses + loss_rotmat.item() * cur_loop_weight
                vtx_img_looses = vtx_img_looses + loss_vtx_img.item() * cur_loop_weight
                edge_length_losses = edge_length_losses + loss_edge.item() * cur_loop_weight
                lmk2d_losses = lmk2d_losses + loss_lmk_2d.item() * cur_loop_weight

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
                log = log + f"| r : {rotmat_losses:.4f} "
                # log = log + f"| m : {mask_losses:.4f} "
                log = log + f"| lmk : {lmk68_losses:.4f} "
                log = log + f"| im : {vtx_img_looses:.4f} "
                log = log + f"| ed : {edge_length_losses:.4f} "
                log = log + f"| lmk2d : {lmk2d_losses:.4f} "

                log = log + f"| lr[0] : {optimizer.param_groups[0]['lr']} "
                # log = log + f"| lr[1] : {optimizer.param_groups[1]['lr']} "
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
        # result dict
        result = defaultdict(list)
        result['strict_success'] = 0
        result['easy_success'] = 0
        result['total_count'] = 0
        prediction = defaultdict(list)
        if epoch % 1 == 0:
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

                    gt_vtx_world = data['vtx_world'].to(device)  # [B,1220,4]
                    gt_vtx_cam = data['vtx_cam'].to(device)  # [B,1220,4]
                    has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
                    gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
                    intrinsic_crop = data['K_crop'].to(device)  # [B,4,3]
                    bbox_info = data['bbox_info'].to(device)  # [B,5]

                    # forward
                    preds_dict, _ = model(img, intrinsic_crop, bbox_info)
                    preds = preds_dict['output'][-1]

                    pred_vtx_world, pred_R_t, pred_vtx_cam = \
                        preds['pred_vtx_world'], \
                        preds['pred_R_t'], \
                        preds['pred_vtx_cam']

                    # save prediction
                    prediction["pred_vtx_world"].append(pred_vtx_world.cpu().numpy())
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

                    result['yaw_mae'].append(yaw_mae)
                    result['pitch_mae'].append(pitch_mae)
                    result['roll_mae'].append(roll_mae)

                    result['tx_mae'].append(tx_mae)
                    result['ty_mae'].append(ty_mae)
                    result['tz_mae'].append(tz_mae)

                    pred_vtx_cam_ADD = torch.bmm(gt_vtx_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    gt_vtx_cam_ADD = torch.bmm(gt_vtx_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD).mean()
                    result['ADD'].append(cur_ADD)

                    ##############################################################################################
                    # Angular distance
                    ##############################################################################################
                    chordal_mean, _, _ = get_chordal_distance_from_rotmat(pred_R[:, None, :, :], gt_R[:, None, :, :])
                    angular_distance = radian2degree(chordal2angular(chordal_mean))
                    result['angular_distance'].append(angular_distance)

                    ##############################################################################################
                    # calc 3DRecon error
                    ##############################################################################################
                    pred_vtx_world_np = pred_vtx_world.cpu().numpy()  # [B,1220,3]
                    gt_vtx_world_np = gt_vtx_world[:, :, :3].cpu().numpy()  # [B,1220,3]
                    mae_vtx_world = calc_keypoint_mae(pred_vtx_world_np, gt_vtx_world_np).mean()
                    result['3DRecon'].append(mae_vtx_world)

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

            # gather prediction
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            out_npz_path = os.path.join(checkpoint_dir, 'prediction.npz')
            for k in prediction.keys():
                try:
                    prediction[k] = np.concatenate(prediction[k], axis=0)
                except:
                    logger.info(f'error : {k}')

            # np.savez(out_npz_path, **prediction)

            prediction = None
            pred_vtx_world = None
            gt = None
            gt_vtx = None
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

        ##############################################################################################
        # Save checkpoint
        ##############################################################################################
        if epoch % train_opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
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

        ##############################################################################################
        # Evaluation on BIWI dataset
        ##############################################################################################
        # result dict
        result = defaultdict(list)
        prediction = defaultdict(list)
        if epoch % 1 == 0:
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

                    gt_vtx_world = data['vtx_world'].to(device)  # [B,1220,4]
                    gt_vtx_cam = data['vtx_cam'].to(device)  # [B,1220,4]
                    has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
                    gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
                    intrinsic_crop = data['K_crop'].to(device)  # [B,4,3]
                    bbox_info = data['bbox_info'].to(device)  # [B,5]

                    # forward
                    preds_dict, _ = model(img, intrinsic_crop, bbox_info)
                    preds = preds_dict['output'][-1]

                    pred_vtx_world, pred_R_t, pred_vtx_cam = \
                        preds['pred_vtx_world'], \
                        preds['pred_R_t'], \
                        preds['pred_vtx_cam']

                    # save prediction
                    prediction["pred_vtx_world"].append(pred_vtx_world.cpu().numpy())
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

                    result['yaw_mae'].append(yaw_mae)
                    result['pitch_mae'].append(pitch_mae)
                    result['roll_mae'].append(roll_mae)

                    result['tx_mae'].append(tx_mae)
                    result['ty_mae'].append(ty_mae)
                    result['tz_mae'].append(tz_mae)

                    pred_vtx_cam_ADD = torch.bmm(gt_vtx_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    gt_vtx_cam_ADD = torch.bmm(gt_vtx_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD).mean()
                    result['ADD'].append(cur_ADD)

                    ##############################################################################################
                    # Angular distance
                    ##############################################################################################
                    chordal_mean, _, _ = get_chordal_distance_from_rotmat(pred_R[:, None, :, :], gt_R[:, None, :, :])
                    angular_distance = radian2degree(chordal2angular(chordal_mean))
                    result['angular_distance'].append(angular_distance)

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

            # gather prediction
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            out_npz_path = os.path.join(checkpoint_dir, 'prediction.npz')
            for k in prediction.keys():
                try:
                    prediction[k] = np.concatenate(prediction[k], axis=0)
                except:
                    logger.info(f'error : {k}')

            # np.savez(out_npz_path, **prediction)

            prediction = None
            pred_vtx_world = None
            gt = None
            gt_vtx = None
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
        # Evaluation on AFLW2000-3D dataset
        ##############################################################################################
        # result dict
        result = defaultdict(list)
        prediction = defaultdict(list)
        if epoch % 1 == 0:
            logger.info('===================== evaluation epoch %d begin =====================' % epoch)
            t_start = time.time()
            num_batch = len(aflw2000_val_dataset.dataloader)

            model.eval()
            for j, data in enumerate(aflw2000_val_dataset):
                with torch.no_grad():
                    if j == 5 and val_opt.debugging:
                        break
                    # load data
                    img = data['img'].to(device)
                    batch_size = img.shape[0]

                    gt_vtx_world = data['vtx_world'].to(device)  # [B,1220,4]
                    gt_vtx_cam = data['vtx_cam'].to(device)  # [B,1220,4]
                    has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
                    gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
                    intrinsic_crop = data['K_crop'].to(device)  # [B,4,3]
                    bbox_info = data['bbox_info'].to(device)  # [B,5]

                    # forward
                    preds_dict, _ = model(img, intrinsic_crop, bbox_info)
                    preds = preds_dict['output'][-1]

                    pred_vtx_world, pred_R_t, pred_vtx_cam = \
                        preds['pred_vtx_world'], \
                        preds['pred_R_t'], \
                        preds['pred_vtx_cam']

                        # save prediction
                    prediction["pred_vtx_world"].append(pred_vtx_world.cpu().numpy())
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

                    # yaw_mae, pitch_mae, roll_mae = calc_rotation_mae(pred_R, gt_R)
                    yaw_mae, pitch_mae, roll_mae = calc_rotation_mae_aflw(pred_R, gt_R)
                    tx_mae, ty_mae, tz_mae = calc_trans_mae(pred_t, gt_t)

                    result['yaw_mae'].append(yaw_mae)
                    result['pitch_mae'].append(pitch_mae)
                    result['roll_mae'].append(roll_mae)

                    result['tx_mae'].append(tx_mae)
                    result['ty_mae'].append(ty_mae)
                    result['tz_mae'].append(tz_mae)

                    pred_vtx_cam_ADD = torch.bmm(gt_vtx_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    gt_vtx_cam_ADD = torch.bmm(gt_vtx_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
                    cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD).mean()
                    result['ADD'].append(cur_ADD)

                    ##############################################################################################
                    # Angular distance
                    ##############################################################################################
                    chordal_mean, _, _ = get_chordal_distance_from_rotmat(pred_R[:, None, :, :],
                                                                          gt_R[:, None, :, :])
                    angular_distance = radian2degree(chordal2angular(chordal_mean))
                    result['angular_distance'].append(angular_distance)

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
                        test_percentage = j / len(aflw2000_val_dataset.dataloader) * 100
                        logger.info(
                            f"evaluate : [{cur_time}] [{test_percentage:.2f}%] "
                        )

            # gather prediction
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-{epoch}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            out_npz_path = os.path.join(checkpoint_dir, 'prediction.npz')
            for k in prediction.keys():
                try:
                    prediction[k] = np.concatenate(prediction[k], axis=0)
                except:
                    logger.info(f'error : {k}')

            # np.savez(out_npz_path, **prediction)

            prediction = None
            pred_vtx_world = None
            gt = None
            gt_vtx = None
            ##############################################################################################
            # Summarize all metric
            ##############################################################################################
            metric = aflw2000_val_dataset.dataset.compute_metrics_(result)

            logger.info('\n [metric] ')
            pprint.pprint(metric)
            logger.info('Done in %.3f sec' % (time.time() - t_start))
            # record metric
            out_json_path = os.path.join(checkpoint_dir, 'metric_aflw.json')
            logger.info(f'save metric file into: {out_json_path}')
            with open(out_json_path, 'w') as fout:
                json.dump(metric, fout, indent=4, sort_keys=True)




