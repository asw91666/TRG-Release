import os
import numpy
import trimesh
from pdb import set_trace
import pickle
import face_alignment
import cv2

def read_pkl(path):
    with open(path, 'rb') as fr:
        data = pickle.load(fr)
    return data

def write_pkl(path, data):
    with open(path, 'wb') as fout:
        pickle.dump(data, fout)

def get_file_dir_name(dir_path, target_name=None):
    file_dir_names = os.listdir(dir_path)
    file_dir_names.sort()

    file_dir_path = []
    target_list = []
    target_path = []
    for name in file_dir_names:
        # return path
        tmp_path = os.path.join(dir_path, name)
        file_dir_path.append(tmp_path)

        if target_name is not None and target_name in name:
            # return name of file
            target_list.append(name)

            # return path
            tmp_path = os.path.join(dir_path, name)
            target_path.append(tmp_path)

    return file_dir_names, file_dir_path, target_list, target_path

def plotting_points2d_on_img_cv(img, points2d, circle_size=2, color='red'):
    """
    img: np.ndarray, [H,W,3]
    points2d: [n_points,2]
    circle_size: plottin point size

    return display, [H,W,3]
    """
    color_dict = {
        'red':(0,0,255),
        'blue':(255,0,0),
        'green':(0,255,0)
    }
    display = img.copy()
    for i, (x, y) in enumerate(points2d):
        cv2.circle(display, (int(x), int(y)), circle_size, color_dict[color], -1)

    return display

if __name__ == "__main__":
    """
    Dataset is downloaded from here [https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html]
    FAN model is from here [https://github.com/1adrianb/face-alignment]
    """
    biwi_root = '/home/kwu/data/BIWI/download_from_official_site/kinect_head_pose_db/hpdb'

    # load FAN
    model_fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False)

    save_dict = {
        'img_path': [],
        'mesh': {},
        'pred_kpt': {}
    }

    frame_cnt = 0
    subject = [str(i).zfill(2) for i in range(1, 24 + 1)]
    print(f'subject: {subject}')
    for subj in subject:
        ####################################################################################
        # gt mesh
        ####################################################################################
        obj_file_path = os.path.join(biwi_root, f'{subj}.obj')
        mesh = trimesh.load_mesh(obj_file_path)
        vertices = mesh.vertices  # [6918,3]
        mesh_faces = mesh.faces  # [13676,3]
        save_dict['mesh'][subj] = {'vertices': vertices, 'faces': mesh_faces}

        ####################################################################################
        # img path
        ####################################################################################
        subj_dir_path = os.path.join(biwi_root, subj)
        # subj_file_list = os.listdir(subj_dir_path)

        _, _, file_name_list, file_path_list = get_file_dir_name(subj_dir_path, '.png')

        for i, img_path in enumerate(file_path_list):
            pose_path = img_path.replace('_rgb.png', '_pose.txt')
            if os.path.isfile(pose_path):
                img_path_save = img_path.replace(f'{biwi_root}/', "")
                save_dict['img_path'].append(img_path_save)
                if frame_cnt % 100 == 0:
                    print(f'frame_cnt: {frame_cnt}')
                frame_cnt = frame_cnt + 1

            ####################################################################################
            # Get prediction by FAN
            ####################################################################################
            img = cv2.imread(img_path)
            preds = model_fan.get_landmarks(img)
            try:
                key = img_path.replace(f'{biwi_root}/', "")
                save_dict['pred_kpt'][key] = preds[0]
                if frame_cnt % 10 == 0:
                    plotted_img = plotting_points2d_on_img_cv(img.copy(), preds[0][:,:2])
                    cv2.imshow('plotted_img', plotted_img)
                    cv2.waitKey(1)
            except TypeError:
                print(f'FAN cannot detect face from {img_path}')
                pass

    print(f'total frame : {frame_cnt}')
    out_path = os.path.join(biwi_root, 'annot_predFAN.pkl')
    write_pkl(out_path, save_dict)
    print(f'save annotation: {out_path}')

    # test_dict = read_pkl(out_path)
    # set_trace()