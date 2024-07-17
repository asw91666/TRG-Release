import pdb
import cv2
import sys
import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
# from lib.mesh_ori import render
from lib.mesh import render


class EulerAugmentor(object):
    def __init__(self, data_root):
        npz_path = 'npy/kpt_ind.npy'
        self.kpt_ind = np.load(npz_path)
        self.nose_ind = self.kpt_ind[30]

        mat_path = 'npy/arkit_morph.mat'
        C = sio.loadmat(mat_path)
        model = C['model'][0, 0]

        tris = model['tri'] - 1
        tris_copy = tris.copy()
        tris[:, [0, 1]] = tris_copy[:, [1, 0]]
        self.tris = np.ascontiguousarray(tris.astype(np.int32))  # (2304, 3)


 
        self.tris_organ = np.array([
            # 
            [ self.kpt_ind[36], self.kpt_ind[37], self.kpt_ind[41] ],
            [ self.kpt_ind[37], self.kpt_ind[38], self.kpt_ind[41] ],
            [ self.kpt_ind[38], self.kpt_ind[40], self.kpt_ind[41] ],
            [ self.kpt_ind[38], self.kpt_ind[39], self.kpt_ind[40] ],

            # 
            [ self.kpt_ind[42], self.kpt_ind[43], self.kpt_ind[47] ],
            [ self.kpt_ind[43], self.kpt_ind[44], self.kpt_ind[47] ],
            [ self.kpt_ind[44], self.kpt_ind[46], self.kpt_ind[47] ],
            [ self.kpt_ind[44], self.kpt_ind[45], self.kpt_ind[46] ],

            # 
            [ self.kpt_ind[60], self.kpt_ind[61], self.kpt_ind[67] ],
            [ self.kpt_ind[61], self.kpt_ind[62], self.kpt_ind[67] ],
            [ self.kpt_ind[62], self.kpt_ind[66], self.kpt_ind[67] ],
            [ self.kpt_ind[62], self.kpt_ind[63], self.kpt_ind[66] ],
            [ self.kpt_ind[63], self.kpt_ind[65], self.kpt_ind[66] ],
            [ self.kpt_ind[63], self.kpt_ind[64], self.kpt_ind[65] ]
        ], dtype=np.int32)


    def __call__(self, img, M):        
        pitch, yaw, roll = M['euler_angles']

        allow_pitch_aug = ( (8 <= pitch and pitch <= 25) or (-35 <= pitch and pitch <= -20) ) and (abs(yaw) <= 15)
        allow_yaw_aug = abs(pitch) <= 25 and abs(yaw) >= 20
        allow_roll_aug = True


        # True, True, True，
        if allow_pitch_aug and allow_yaw_aug and allow_roll_aug:
            euler_name = np.random.choice(['pitch', 'yaw', 'roll'], 1, p=[0.4, 0.4, 0.2])[0]

        # True, False, True, 
        elif allow_pitch_aug and not allow_yaw_aug and allow_roll_aug:
            euler_name = np.random.choice(['pitch', 'roll'], 1, p=[0.8, 0.2])[0]

        # False, True, True, 
        elif not allow_pitch_aug and allow_yaw_aug and allow_roll_aug:
            euler_name = np.random.choice(['yaw', 'roll'], 1, p=[0.8, 0.2])[0]

        # False, False, True，
        elif not allow_pitch_aug and not allow_yaw_aug and allow_roll_aug:
            euler_name = np.random.choice(['roll', 'none'], 1, p=[0.4, 0.6])[0]



        if euler_name == 'pitch':
            pitch_delta = np.random.uniform(5, 30) * np.sign(pitch)
            return self.process(img, M, 'pitch', pitch_delta)
        
        elif euler_name == 'yaw':
            expand = 30
            if yaw >= 0:
                max_ = expand if yaw + expand <= 90 else (90 - yaw)
                yaw_delta = np.random.uniform(0, max_)
            else:
                max_ = -expand if yaw - expand >= -90 else (-90 - yaw)
                yaw_delta = np.random.uniform(max_, 0)
            return self.process(img, M, 'yaw', yaw_delta)
        
        elif euler_name == 'roll':
            roll_delta = np.random.uniform(5, 30) * np.sign(roll)
            return self.process(img, M, 'roll', roll_delta)
       
        elif euler_name == 'none':
            return img, M 
            


    def process(self, img, M, euler_name, delta_degree):      
        assert euler_name in ['pitch', 'yaw', 'roll']
        if euler_name is 'roll':
            img_aug, M = self.perform_roll(img, M, euler_name, delta_degree)
            return img_aug, M
        else:
            try:   
                img_aug, M = self.perform_pitch_or_yaw(img, M, euler_name, delta_degree)
                return img_aug, M
            except:
                print('【euler_augmentation fail】')
                return img, M

    # https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def perform_roll(self, img, M, euler_name, delta_degree):       
        assert euler_name is 'roll'
        img_h, img_w, _ = img.shape
        verts_origin = M['verts_gt']
        ones = np.ones([verts_origin.shape[0], 1])
        verts_homo = np.concatenate([verts_origin, ones], axis=1)
        R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])

        yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(R_t[:3, :3].T).as_euler('yxz', degrees=True)
        roll_gt += delta_degree


        new_R_t = R_t.copy()
        new_R_t[:3, :3] = Rotation.from_euler('yxz', [yaw_gt, pitch_gt, roll_gt], degrees=True).as_matrix().T

        M1 = np.array([
            [img_w/2,       0, 0, 0],
            [      0, img_h/2, 0, 0],
            [      0,       0, 1, 0],
            [img_w/2, img_h/2, 0, 1]
        ])

        verts = verts_homo @ new_R_t @ M['projectionMatrix'] @ M1
        w_ = verts[:, [3]]
        verts = verts / w_


        center_homo = np.array([[0, 0, 0, 1]])
        center = center_homo @ new_R_t @ M['projectionMatrix'] @ M1
        w_ = center[:, [3]]
        center = center / w_


        center_point = (center[0, 0], img_h - center[0, 1])
        img_aug = self.rotate(img, delta_degree, center_point)


        # euler_angles, faceAnchortransform, points2d
        if not isinstance(M, dict):
            M = dict(M)

        M['euler_angles'] = np.array([pitch_gt, yaw_gt, roll_gt])
        M['faceAnchortransform'] = new_R_t @ M['cameratransform']    # faceAnchortransform

        R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])
        M['translation'] = R_t[3, :3]

        verts_tfm = verts_homo @ R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm[:, [3]]
        verts_tfm = verts_tfm / w_
        
        points2d = verts_tfm[:, :2].copy()
        points2d[:, 1] = img_h - points2d[:, 1]
        M['points2d'] = points2d      
        return img_aug, M

    def perform_pitch_or_yaw(self, img, M, euler_name, delta_degree):
        assert euler_name in ['pitch', 'yaw']
        img_h, img_w, _ = img.shape

        # step1
        verts_origin = M['verts_gt']
        N1 = len(verts_origin)
        ones = np.ones([verts_origin.shape[0], 1])
        verts_homo = np.concatenate([verts_origin, ones], axis=1)
        R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])
        M1 = np.array([
            [img_w/2,       0, 0, 0],
            [      0, img_h/2, 0, 0],
            [      0,       0, 1, 0],
            [img_w/2, img_h/2, 0, 1]
        ])
        verts_tfm = verts_homo @ R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm[:, [3]]
        verts_tfm = verts_tfm / w_

        # step2
        points_list = verts_tfm[:, :2]
        hull = ConvexHull(points_list)
        edge_list = hull.simplices.tolist()
        top_ind = points_list[:, 1].argmin()  
        cur_ind = top_ind
        contour_ind = []
        while True:
            # 
            for k in range(len(edge_list)):
                if edge_list[k][0] == cur_ind:
                    break
                elif edge_list[k][1] == cur_ind:
                    break

            if edge_list[k][0] == cur_ind:
                cur_ind = edge_list[k][1]
            elif edge_list[k][1] == cur_ind:
                cur_ind = edge_list[k][0]

            contour_ind.append(cur_ind)
            edge_list.remove(edge_list[k])

            if cur_ind == top_ind:
                break

        num_contours = len(contour_ind)

        # step3
        scale_list = [1.05, 1.1, 1.15, 1.2]
        N2 = num_contours * len(scale_list)
        verts_origin_all = verts_origin.copy()
        for scale in scale_list:
            verts_scale = verts_origin * scale
            d = verts_origin[self.nose_ind] - verts_scale[self.nose_ind]
            verts_scale += d
            verts_origin_all = np.concatenate([verts_origin_all, verts_scale[contour_ind]], axis=0)

        ones = np.ones([verts_origin_all.shape[0], 1])
        verts_homo_all = np.concatenate([verts_origin_all, ones], axis=1)

        verts_tfm_all = verts_homo_all @ R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm_all[:, [3]]
        verts_tfm_all = verts_tfm_all / w_


        # step4
        points_list = []
        for p in verts_tfm_all[N1:]:
            x = round(float(p[0]), 4)
            y = round(float(p[1]), 4)
            points_list.append((x, y)) 

        # img_square
        points2d_68 = M['points2d'][self.kpt_ind, :2]

        x_min, y_min = points2d_68.min(axis=0)
        x_max, y_max = points2d_68.max(axis=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        size = max(w, h)
        ss = np.array([0.75, 0.75, 0.85, 0.65]) * 1.5

        left = x_center - ss[0] * size
        right = x_center + ss[1] * size
        top = y_center - ss[2] * size
        bottom = y_center + ss[3] * size

        # bbox
        left = self.out_of_boundary_correct(left, img_w)
        right = self.out_of_boundary_correct(right, img_w)
        top = self.out_of_boundary_correct(top, img_h) 
        bottom = self.out_of_boundary_correct(bottom, img_h) 

        divide_num = 10
        delta_x = (right - left) / divide_num
        x_list = [left + delta_x * i for i in range(divide_num + 1)]

        delta_y = (bottom - top) / divide_num
        y_list = [top + delta_y * i for i in range(divide_num + 1)]

        for i in range(divide_num):
            points_list.append( (x_list[i], img_h - top) )
        for i in range(divide_num):
            points_list.append( (right, img_h - y_list[i]) )
        for i in range(divide_num):
            points_list.append( (x_list[::-1][i], img_h - bottom) )
        for i in range(divide_num):
            points_list.append( (left, img_h - y_list[::-1][i]) )

        N3 = len(points_list) - N2

        temp = np.array(points_list)
        t1 = temp[:N2]
        t2 = temp[N2:]

        dis = cdist(t1, t2, metric='euclidean')
        min_idx = dis.argmin(axis=0)
        min_idx += N1  

        z_val = verts_tfm_all[min_idx, 2].reshape(-1, 1)

        xy_val = t2
        ones = np.ones([xy_val.shape[0], 1])
        verts_tfm_square = np.concatenate([xy_val, z_val, ones], axis=1)

        T = R_t @ M['projectionMatrix'] @ M1
        verts_origin_square = verts_tfm_square @ np.linalg.inv(T)
        w_ = verts_origin_square[:, [3]]
        verts_origin_square = verts_origin_square / w_

        verts_origin_square = verts_origin_square[:, :3]
        verts_origin_all = np.concatenate([verts_origin_all, verts_origin_square], axis=0)

        ones = np.ones([verts_origin_all.shape[0], 1])
        verts_homo_all = np.concatenate([verts_origin_all, ones], axis=1)

        verts_tfm_all = verts_homo_all @ R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm_all[:, [3]]
        verts_tfm_all = verts_tfm_all / w_


        # step5
        inner_list = contour_ind
        outer_list = range(N1 + num_contours * 0, N1 + num_contours * 1)
        tris_contour = []
        for i in range(num_contours):
            t1 = ( inner_list[i % num_contours], outer_list[i % num_contours], outer_list[(i+1) % num_contours] )
            t2 = ( inner_list[i % num_contours], inner_list[(i+1) % num_contours], outer_list[(i+1) % num_contours] )
            tris_contour.append(t1)
            tris_contour.append(t2)

        for k in range(len(scale_list) - 1):
            inner_list = range(N1 + num_contours * k, N1 + num_contours * (k+1))
            outer_list = range(N1 + num_contours * (k+1), N1 + num_contours * (k+2))
            for i in range(num_contours):
                t1 = ( inner_list[i % num_contours], outer_list[i % num_contours], outer_list[(i+1) % num_contours] )
                t2 = ( inner_list[i % num_contours], inner_list[(i+1) % num_contours], outer_list[(i+1) % num_contours] )
                tris_contour.append(t1)
                tris_contour.append(t2)

        tris_contour = np.array(tris_contour)

        subdiv = cv2.Subdiv2D((0, 0, img_w * 3, img_h * 3))

        points_list = []
        selected_ind = range(N1+N2-num_contours, N1+N2+N3)

        for p in verts_tfm_all[selected_ind]:
            x = round(float(p[0]), 4)
            y = round(float(img_h - p[1]), 4)
            points_list.append((x, y))

        subdiv.insert(points_list)

        tris_outer = []
        for t in subdiv.getTriangleList():
            pt0 = ( round(float(t[0]), 4), round(float(t[1]), 4) )
            pt1 = ( round(float(t[2]), 4), round(float(t[3]), 4) )
            pt2 = ( round(float(t[4]), 4), round(float(t[5]), 4) )
            idx0 = points_list.index(pt0)
            idx1 = points_list.index(pt1)
            idx2 = points_list.index(pt2)

            c1 = 0 <= idx0 and idx0 < num_contours
            c2 = 0 <= idx1 and idx1 < num_contours
            c3 = 0 <= idx2 and idx2 < num_contours

            if c1 and c2 and c3:
                continue

            tris_outer.append([idx0, idx1, idx2])

        offset = N1 + N2 - num_contours
        tris_outer = np.array(tris_outer) + offset

        tris_all = np.concatenate([self.tris, self.tris_organ, tris_contour, tris_outer], axis=0)
        tris_all = np.ascontiguousarray(tris_all.astype(np.int32))


        # step6
        verts_temp = verts_tfm_all[:, :3].copy()
        verts_temp[:, 1] = img_h - verts_temp[:, 1]
        verts_temp[:, 2] *= -1  #   
     
        verts_tex = verts_temp.copy()
        verts_tex[:, 2] = 0


        # step7
        yaw_gt, pitch_gt, roll_gt = Rotation.from_matrix(R_t[:3, :3].T).as_euler('yxz', degrees=True)
        
        if euler_name == 'yaw':
            yaw_gt += delta_degree
        if euler_name == 'pitch':
            pitch_gt += delta_degree


        new_R_t = R_t.copy()
        new_R_t[:3, :3] = Rotation.from_euler('yxz', [yaw_gt, pitch_gt, roll_gt], degrees=True).as_matrix().T

        verts_tfm_all = verts_homo_all @ new_R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm_all[:, [3]]
        verts_tfm_all = verts_tfm_all / w_

        verts_temp = verts_tfm_all[:, :3].copy()
        verts_temp[:, 1] = img_h - verts_temp[:, 1]
        verts_temp[:, 2] *= -1 


        img_aug = render.render_texture(
            verts_temp, tris_all,
            img, verts_tex, tris_all,
            img_h, img_w, 3
        ).astype(np.uint8)


        # euler_angles, faceAnchortransform, points2d
        if not isinstance(M, dict):
            M = dict(M)

        M['euler_angles'] = np.array([pitch_gt, yaw_gt, roll_gt])
        M['faceAnchortransform'] = new_R_t @ M['cameratransform']    # 反算faceAnchortransform

        R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])
        M['translation'] = R_t[3, :3]

        verts_tfm = verts_homo @ R_t @ M['projectionMatrix'] @ M1
        w_ = verts_tfm[:, [3]]
        verts_tfm = verts_tfm / w_
        
        points2d = verts_tfm[:, :2].copy()
        points2d[:, 1] = img_h - points2d[:, 1]
        M['points2d'] = points2d
 
        return img_aug, M

    def out_of_boundary_correct(self, val, img_size):
        if val < 0.0:
            return 0.0
        if val > img_size:
            return img_size
        return val

class HorizontalFlipAugmentor(object):

    def __init__(self):
        self.T = np.array([
            [1, -1, -1, 1],
            [-1, 1, 1, 1],
            [-1, 1, 1, 1],
            [-1, 1, 1, 1]
        ], dtype=np.float64)

        img_w, img_h = 800, 800
        self.M1 = np.array([
            [img_w/2,       0, 0, 0],
            [      0, img_h/2, 0, 0],
            [      0,       0, 1, 0],
            [img_w/2, img_h/2, 0, 1]
        ])

    def __call__(self, img, M):        
        img_flip = np.flip(img, axis=1)
        img_h, img_w, _ = img.shape    

        M['verts_gt'][:, 0] *= -1

        R_t = M['faceAnchortransform'] @ np.linalg.inv(M['cameratransform'])
        new_R_t = R_t * self.T

        # 
        verts_origin = M['verts_gt']
        ones = np.ones([verts_origin.shape[0], 1])
        verts_homo = np.concatenate([verts_origin, ones], axis=1)
        
        verts_tfm = verts_homo @ new_R_t @ M['projectionMatrix'] @ self.M1
        w_ = verts_tfm[:, [3]]
        verts_tfm = verts_tfm / w_

        points2d = verts_tfm[:, :2]
        points2d[:, 1] = img_h - points2d[:, 1]

        d = {
            'verts_gt': M['verts_gt'],
            'points2d': points2d
            
        }
        return img_flip, d
        




