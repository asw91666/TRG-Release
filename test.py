import os
import sys
import cv2
import time
import pprint
import numpy as np
from datetime import datetime
from options.test_options import TestOptions
from data import create_dataset
from util.pkl import write_pkl, read_pkl
from util.logger import setup_logger
from models.trg_model.core.cfgs import parse_args
from models.trg_model import load_trg
import torch
from pdb import set_trace
from collections import defaultdict
from util.metric import *
import json

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
    init_face_pose_dict = read_pkl('data/init_vtx_cam.pkl')
    init_face, init_pose = init_face_pose_dict['t_vtx_1220'], init_face_pose_dict['R_t']

    # get config
    cfg_path = 'models/trg_model/configs/trg_face_config.yaml'
    config = parse_args(cfg_path)

    # load model
    model = load_trg(config, init_face, init_pose).to(device)
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

    # define mesh faces
    mesh_faces = np.load('./data/triangles.npy')  # [2304, 3]
    mesh_faces = mesh_faces.astype(np.int64)
    mesh_faces = torch.from_numpy(mesh_faces).to(device)

    # result dict
    result = defaultdict(list)
    prediction = defaultdict(list)

    result['strict_success'] = 0
    result['easy_success'] = 0
    result['total_count'] = 0
    num_no_fan = 0

    inference_time = []
    for i, data in enumerate(dataset):
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

            # forward
            start = time.time()
            preds_dict, _ = model(img, intrinsic, bbox_info)
            end = time.time()
            inference_time.append(end - start)

            preds = preds_dict['output'][-1]
            pred_vtx_world, pred_R_t, pred_vtx_cam = \
                            preds['pred_face_world'], \
                            preds['pred_R_t'], \
                            preds['pred_face_cam']

            ##############################################################################################
            # Gather rotate data for alignment reference system (cobo PR24)
            ##############################################################################################
            gt_R = gt_extrinsic.transpose(1, 2)[:, :3, :3].cpu().numpy()  # [B,4,4]
            pred_R = pred_R_t.transpose(1, 2)[:, :3, :3].cpu().numpy()  # [B,4,4]

            ##############################################################################################
            # yaw, pitch, roll, tx, ty, tz, ADD
            ##############################################################################################
            gt_R_t_tmp = gt_extrinsic.transpose(1,2).cpu().numpy() # [B,4,4]
            gt_R = gt_R_t_tmp[:,:3,:3] # [B,3,3]
            gt_t = gt_R_t_tmp[:,:3,3] # [B,3]
            pred_R_t_tmp = pred_R_t.transpose(1, 2).cpu().numpy() # [B,4,4]
            pred_R = pred_R_t_tmp[:,:3,:3] # [B,3,3]
            pred_t = pred_R_t_tmp[:, :3, 3]  # [B,3]

            yaw_mae, pitch_mae, roll_mae = calc_rotation_mae(pred_R, gt_R)
            r1_err, r2_err, r3_err, _ = calculate_mae_of_vectors(pred_R, gt_R)
            tx_mae, ty_mae, tz_mae = calc_trans_mae(pred_t, gt_t)

            _, ge_err = calc_geodesic_error(pred_R, gt_R)
            result['ge_err'].append(ge_err)  # batch_size

            result['r1_err'].append(r1_err)  # batch_size
            result['r2_err'].append(r2_err)
            result['r3_err'].append(r3_err)

            result['yaw_mae'].append(yaw_mae) # batch_size
            result['pitch_mae'].append(pitch_mae)
            result['roll_mae'].append(roll_mae)

            result['tx_mae'].append(tx_mae) # batch_size
            result['ty_mae'].append(ty_mae)
            result['tz_mae'].append(tz_mae)

            pred_vtx_cam_ADD = torch.bmm(gt_vtx_world, pred_R_t)[:, :, :3].cpu().numpy()  # [B,1220,3]
            gt_vtx_cam_ADD = torch.bmm(gt_vtx_world, gt_extrinsic)[:, :, :3].cpu().numpy()  # [B,1220,3]
            cur_ADD = calc_keypoint_mae(pred_vtx_cam_ADD, gt_vtx_cam_ADD)
            result['ADD'].append(cur_ADD) # batch_size

            ##############################################################################################
            # calc 3DRecon error
            ##############################################################################################
            if opt.dataset_mode == 'arkit':
                pred_vtx_world_np = pred_vtx_world.cpu().numpy() # [B,1220,3]
                gt_vtx_world_np = gt_vtx_world[:,:,:3].cpu().numpy() # [B,1220,3]
                l2_norm_verts_batch = calc_keypoint_l2(pred_vtx_world_np, gt_vtx_world_np) # [B,1220]
                mean_error = np.mean(l2_norm_verts_batch, axis=1) # [B]

                result['3DRecon'].append(mean_error)
            elif opt.dataset_mode == 'biwi':
                result['3DRecon'].append(np.zeros(1, dtype=np.float32))

            ##############################################################################################
            # calc face size error
            ##############################################################################################
            if opt.dataset_mode == 'arkit':
                # calc face size error
                m2mm = 1000
                gt_tri_area = calc_mesh_area_from_vertices_batch(gt_vtx_world[:, :, :3] * m2mm, mesh_faces)
                gt_face_size = torch.sum(gt_tri_area, dim=1)

                pred_tri_area = calc_mesh_area_from_vertices_batch(pred_vtx_world * m2mm, mesh_faces)
                pred_face_size = torch.sum(pred_tri_area, dim=1)

                size_error = torch.abs(gt_face_size - pred_face_size)
                size_error = size_error.detach().cpu().numpy()

                result['face_size_error'].append(size_error)

            elif opt.dataset_mode == 'biwi':
                result['face_size_error'].append(np.zeros(1, dtype=np.float32))

            # logger
            if i % 10 == 0:
                now = datetime.now()
                cur_time = now.strftime('%m-%d %H:%M:%S')
                test_percentage = i / len(dataset.dataloader) * 100
                logger.info(
                    f"[{cur_time}] [{test_percentage:.2f}%] "
                )

    # inference time
    inference_time = np.mean(np.asarray(inference_time))
    print(f'inference_time: {inference_time}')

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

    out_json_path = os.path.join(checkpoint_path, f'metric_{opt.dataset_mode}.json')

    with open(out_json_path, 'w') as fout:
        json.dump(metric, fout, indent=4, sort_keys=True)
    logger.info(f'save metric file into: {out_json_path}')