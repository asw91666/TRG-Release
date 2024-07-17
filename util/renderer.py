import pdb
import cv2
import socket
import numpy as np
from pathlib import Path

from lib.Sim3DR import RenderPipeline
ip = socket.gethostbyname(socket.gethostname())

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order="C")
    return arr

cfg = {
    "intensity_ambient": 0.3,
    "color_ambient": (1, 1, 1),
    "intensity_directional": 0.6,
    "color_directional": (1, 1, 1),
    "intensity_specular": 0.1,
    "specular_exp": 5,
    "light_pos": (0, 0, 5),
    "view_pos": (0, 0, 5),
}
render_app = RenderPipeline(**cfg)


class Renderer(object):

    def __init__(self, img_size=800, alpha=0.7):
        npy_path = './npy/tris_2500x3_202110.npy'
        tris = np.load(npy_path)[:2304]
        tris[:, [0, 1]] = tris[:, [1, 0]]
        self.tris = np.ascontiguousarray(tris.astype(np.int32))  # (2304, 3)

        self.img_size = img_size
        self.alpha = alpha

        self.M_proj = np.array([
            [1.574437,        0,             0,  0],
            [       0, 1.574437,             0,  0],
            [       0,        0,   -0.99999976, -1],
            [       0,        0, -0.0009999998,  0]
        ])
        self.M1 = np.array([
            [img_size/2,          0, 0, 0],
            [         0, img_size/2, 0, 0],
            [         0,          0, 1, 0],
            [img_size/2, img_size/2, 0, 1]
        ])

        self.f = self.M_proj[0, 0] * img_size / 2






    def __call__(self, verts3d, R_t, overlap=None):
        ones = np.ones([verts3d.shape[0], 1])
        verts_homo = np.concatenate([verts3d, ones], axis=1)

        verts = verts_homo @ R_t @ self.M_proj @ self.M1
        w_ = verts[:, [3]]
        verts = verts / w_

        # points2d，x right，y down
        points2d = verts[:, :2]
        points2d[:, 1] = self.img_size - points2d[:, 1]


        verts_temp = np.concatenate([points2d, w_], axis=1)

        tz = R_t[3, 2]
        scale = self.f / tz
        verts_temp[:, 2] *= scale

        verts_temp = _to_ctype(verts_temp.astype(np.float32))

        if overlap is None:
            overlap = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8)

        overlap_copy = overlap.copy()
        overlap = render_app(verts_temp, self.tris, overlap)     # overlap上没有透明度

        img_render = cv2.addWeighted(overlap_copy, 1 - self.alpha, overlap, self.alpha, 0)
        return img_render




