import os
import sys
import cv2
import time
import pprint
import numpy as np
from datetime import datetime
from options.test_options import TestOptions
from models import create_model
from data import create_dataset
from util.pkl import write_pkl, read_pkl
from util.logger import setup_logger
from models.pymaf_model.core.cfgs import parse_args
from models.pymaf_model import pymaf_face_net
import torch
from pdb import set_trace
from models.pymaf_face_loss import keypoint_3d_loss
from collections import defaultdict
from util.face_deformnet_utils import compute_sRT_errors
from util.metric import calc_rotation_mae, calc_trans_mae, calc_keypoint_mae, calc_keypoint_l2, calc_rotation_mae_aflw
import json
from util.metric import get_chordal_distance_from_rotmat, chordal2angular, radian2degree

###usage: python -u test.py --image_size=192 --model perspnet --dataset_mode arkit --csv_path_test 'test.csv' 
if __name__ == '__main__':
    print('【process_id】', os.getpid())
    print('【command】python -u ' + ' '.join(sys.argv) + '\n')

    opt = TestOptions().parse()  # get test options
    opt.serial_batches = True
    opt.num_thread = 4
    print(f'set batch_size : {opt.batch_size}')
    dataset = create_dataset(opt)
    num_batch = len(dataset.dataloader)
    print("The number of testing images = %d" % len(dataset))

    global logger
    output_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = setup_logger(f"{opt.name}", output_dir, 0, filename=f"test_log.txt")
    logger.info("Using {} GPUs".format(opt.gpu_ids))

    ### model
    device = 'cuda'
    init_vtx_cam_dict = read_pkl('data/init_vtx_cam.pkl')
    init_vtx, init_cam = init_vtx_cam_dict['t_vtx_1220'], init_vtx_cam_dict['R_t']

    # get config
    cfg_path = 'models/pymaf_model/configs/pymaf_face_config.yaml'
    config = parse_args(cfg_path)

    # load model
    model = pymaf_face_net(config, init_vtx, init_cam).to(device)

    # parallelize
    if len(opt.gpu_ids) > 1:
        print(f'Use MULTI-GPU : {opt.gpu_ids}')
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    subsample = read_pkl('data/arkit_subsample.pkl')

    # load weight file
    save_model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    assert os.path.isdir(save_model_dir)

    checkpoints_dir = os.listdir(save_model_dir)
    for checkpoint in checkpoints_dir:
        if f'checkpoint-{opt.epoch}' in checkpoint:
            checkpoint_path = os.path.join(save_model_dir, checkpoint)
            weight_file_path = os.path.join(checkpoint_path, 'state_dict.bin')
            break

    logger.info("Loading state dict from checkpoint {}".format(weight_file_path))
    cpu_device = torch.device('cpu')
    state_dict = torch.load(weight_file_path, map_location=cpu_device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    t_start = time.time()

    # define loss function
    criterion_3d_keypoints = torch.nn.L1Loss(reduction='none').cuda(device)

    # result dict
    result = defaultdict(list)
    prediction = defaultdict(list)
    result['strict_success'] = 0
    result['easy_success'] = 0
    result['total_count'] = 0
    num_no_fan = 0
    for i, data in enumerate(dataset):

        # print('[%s][progresss] [%d/%d]'%(datetime.now().isoformat(sep=' '), i, num_batch))
        with torch.no_grad():
            # load data
            img = data['img'].to(device)
            batch_size = img.shape[0]

            gt_vtx_world = data['vtx_world'].to(device)  # [B,1220,4]
            gt_vtx_cam = data['vtx_cam'].to(device)  # [B,1220,4]
            has_3d_vtx = data['has_3d_vtx'].squeeze(-1).to(device)  # [B]
            gt_extrinsic = data['R_t'].to(device)  # [B,4,4]
            intrinsic = data['K_crop'].to(device)  # [B,4,3]
            bbox_info = data['bbox_info'].to(device)  # [B,5]
            # pred_lmk68_crop_norm = data['pred_lmk68_crop_norm'].to(device)  # [B,68,2], normalized (-1~1)
            if False:
                has_fan = data['has_fan']
            else:
                has_fan = torch.ones([batch_size], dtype=torch.bool).to('cpu')
            has_fan_np = has_fan.cpu().numpy()

            no_fan = batch_size - has_fan.sum().item()
            num_no_fan = num_no_fan + no_fan

            # forward
            # preds_dict, _ = model(img, intrinsic, bbox_info)
            preds_dict, _ = model(img, intrinsic, bbox_info)
            preds = preds_dict['output'][-1]
            pred_vtx_world, pred_R_t, pred_vtx_cam = \
                            preds['pred_vtx_world'], \
                            preds['pred_R_t'], \
                            preds['pred_vtx_cam']

            # save prediction
            prediction["pred_vtx_world"].append(pred_vtx_world.cpu().numpy())
            # prediction["pred_vtx_cam"].append(pred_vtx_cam.cpu().numpy())
            prediction["pred_R_t"].append(pred_R_t.cpu().numpy())
            prediction["gt_vtx_world"].append(gt_vtx_world[:,:,:3].cpu().numpy())
            # prediction["gt_vtx_cam"].append(gt_vtx_cam[:,:,:3].cpu().numpy())
            prediction["gt_R_t"].append(gt_extrinsic.cpu().numpy())

            try:
                # loss
                # loss_vtx_sub_world = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_sub_world, gt_vtx_sub_world, has_3d_vtx, device)
                loss_vtx_world = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_world, gt_vtx_world, has_3d_vtx, device)
                # loss_vtx_sub_cam = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_sub_cam, gt_vtx_sub_cam, has_3d_vtx, device)
                loss_vtx_cam = keypoint_3d_loss(criterion_3d_keypoints, pred_vtx_cam, gt_vtx_cam, has_3d_vtx, device)
                # loss = opt.vertex_world_weight * (loss_vtx_sub_world + loss_vtx_world) \
                #        + opt.vertex_cam_weight * (loss_vtx_sub_cam + loss_vtx_cam)
                loss = opt.vertex_world_weight * (loss_vtx_world) + opt.vertex_cam_weight * (loss_vtx_cam)
            except:
                pass

            ##############################################################################################
            # yaw, pitch, roll, tx, ty, tz, ADD
            ##############################################################################################
            gt_R_t_tmp = gt_extrinsic.transpose(1,2).cpu().numpy() # [B,4,4]
            gt_R = gt_R_t_tmp[:,:3,:3] # [B,3,3]
            gt_t = gt_R_t_tmp[:,:3,3] # [B,3]
            pred_R_t_tmp = pred_R_t.transpose(1, 2).cpu().numpy() # [B,4,4]
            pred_R = pred_R_t_tmp[:,:3,:3] # [B,3,3]
            pred_t = pred_R_t_tmp[:, :3, 3]  # [B,3]

            if opt.dataset_mode == 'aflw2000':
                yaw_mae, pitch_mae, roll_mae = calc_rotation_mae_aflw(pred_R, gt_R)
            else:
                yaw_mae, pitch_mae, roll_mae = calc_rotation_mae(pred_R, gt_R)
            tx_mae, ty_mae, tz_mae = calc_trans_mae(pred_t, gt_t)

            result['yaw_mae'].append(yaw_mae[has_fan]) # batch_size
            result['pitch_mae'].append(pitch_mae[has_fan])
            result['roll_mae'].append(roll_mae[has_fan])

            result['tx_mae'].append(tx_mae[has_fan_np]) # batch_size
            result['ty_mae'].append(ty_mae[has_fan_np])
            result['tz_mae'].append(tz_mae[has_fan_np])

            pred_vtx_cam_ADD = torch.bmm(gt_vtx_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
            gt_vtx_cam_ADD = torch.bmm(gt_vtx_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
            # cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD).mean()
            cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD)
            result['ADD'].append(cur_ADD[has_fan_np]) # batch_size
            ##############################################################################################
            # Angular distance
            ##############################################################################################
            chordal_mean, _, chordal_batch = get_chordal_distance_from_rotmat(pred_R[:, None, :, :], gt_R[:, None, :, :])
            angular_distance = radian2degree(chordal2angular(chordal_mean))
            angular_distance_batch = radian2degree(chordal2angular(chordal_batch))
            # result['angular_distance'].append(angular_distance) # batch_size
            result['angular_distance'].append(angular_distance_batch[has_fan_np]) # batch_size

            ##############################################################################################
            # calc 3DRecon error
            ##############################################################################################
            if opt.dataset_mode == 'arkit':
                pred_vtx_world_np = pred_vtx_world.cpu().numpy() # [B,1220,3]
                gt_vtx_world_np = gt_vtx_world[:,:,:3].cpu().numpy() # [B,1220,3]
                # mae_vtx_world = calc_keypoint_mae(pred_vtx_world_np, gt_vtx_world_np).mean()
                l2_norm_verts_batch = calc_keypoint_l2(pred_vtx_world_np, gt_vtx_world_np) # [B,1220]
                mean_error = np.mean(l2_norm_verts_batch, axis=1) # [B]
                median_error = np.median(l2_norm_verts_batch, axis=1) # [B]

                result['3DRecon'].append(mean_error)
                result['3DRecon_median'].append(median_error)
            elif opt.dataset_mode == 'biwi':
                result['3DRecon'].append(np.zeros(1, dtype=np.float32))
            elif opt.dataset_mode == 'aflw':
                result['3DRecon'].append(np.zeros(1, dtype=np.float32))

            ##############################################################################################
            # IOU, 5°5cm, 10°5cm
            ##############################################################################################
            gt_R_t_np = gt_extrinsic.cpu().numpy() # [B,4,4]
            pred_R_t_np = pred_R_t.cpu().numpy() # [B,4,4]
            for batch_i in range(batch_size):
                gt_R_t_np_ = gt_R_t_np[batch_i] # [4,4]
                pred_R_t_np_ = pred_R_t_np[batch_i]
                R_error, T_error, IoU = compute_sRT_errors(pred_R_t_np_.T, gt_R_t_np_.T)
                if R_error < 5 and T_error < 0.05:
                    result['strict_success'] += 1
                if R_error < 10 and T_error < 0.05:
                    result['easy_success'] += 1

                result['IoU'].append(IoU)
                result['total_count'] += 1

            # logger
            try:
                if i % 10 == 0:
                    now = datetime.now()
                    cur_time = now.strftime('%m-%d %H:%M:%S')
                    test_percentage = i / len(dataset.dataloader) * 100
                    logger.info(
                        f"[{cur_time}] [{test_percentage:.2f}%] loss : {loss.item() * 1000.:.2f} "
                        # f"| loss_vtx_cam : {loss_vtx_cam.item():.4f} | loss_vtx_sub_cam : {loss_vtx_sub_cam.item():.4f} "
                        # f"| loss_vtx_world : {loss_vtx_world.item():.4f} | loss_vtx_sub_world : {loss_vtx_sub_world.item():.4f} "
                    )
            except:
                if i % 10 == 0:
                    now = datetime.now()
                    cur_time = now.strftime('%m-%d %H:%M:%S')
                    test_percentage = i / len(dataset.dataloader) * 100
                    logger.info(
                        f"[{cur_time}] [{test_percentage:.2f}%] "
                    )

    print(f'num_no_fan: {num_no_fan}')
    # set_trace()

    ########################################################################
    # numpy concatenate
    ########################################################################
    for k in result.keys():
        try:
            result[k] = np.concatenate(result[k], axis=0)
        except:
            print(f'error : {k}')

    ########################################################################
    # Summarize
    ########################################################################
    metric = dataset.dataset.compute_metrics_(result)
    print('\n [metric] ')
    pprint.pprint(metric)
    print('Done in %.3f sec' %(time.time()-t_start))

    # model output 저장하자. pkl
    out_npz_path = os.path.join(checkpoint_path, f'prediction_{opt.dataset_mode}.npz')
    for k in prediction.keys():
        try:
            prediction[k] = np.concatenate(prediction[k], axis=0)
        except:
            print(f'error : {k}')

    np.savez(out_npz_path, **prediction)
    # logger.info(f'save prediction into: {out_npz_path}')
    # data_ = np.load(out_npz_path)

    # record metric
    # model output 저장하자. pkl
    try:
        if opt.do_mask_patch:
            out_json_path = os.path.join(checkpoint_path, f'metric_{opt.dataset_mode}_{opt.mask_size}.json')
        else:
            out_json_path = os.path.join(checkpoint_path, f'metric_{opt.dataset_mode}.json')
    except:
        out_json_path = os.path.join(checkpoint_path, f'metric_{opt.dataset_mode}.json')

    with open(out_json_path, 'w') as fout:
        json.dump(metric, fout, indent=4, sort_keys=True)
    logger.info(f'save metric file into: {out_json_path}')






