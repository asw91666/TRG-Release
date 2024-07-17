import matplotlib.pyplot as plt
import torch
import numpy as np
from pdb import set_trace
import cv2
import trimesh
import pyrender
from math import cos, sin

def matplotlib_3d_ptcloud(point3d, view_angle=None):
    '''
    Example)

    view_angle = (90, -90)
    # points : [nverts, 3]
    matplotlib_3d_ptcloud(points, view_angle=view_angle)

    '''
    if isinstance(point3d, np.ndarray):
        data = point3d.copy()
    elif torch.is_tensor(point3d):
        data = point3d.detach().cpu().numpy()

    xdata = data[:,0].squeeze()
    ydata = data[:,1].squeeze()
    zdata = data[:,2].squeeze()

    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')
    if view_angle is not None:
        ax.view_init(view_angle[0],view_angle[1])
    ax.scatter3D(xdata, ydata, zdata, marker='o')
    plt.show()

def matplotlib_3d_ptcloud_list(point3d_list, view_angle=None):
    '''
    Example)

    points = [vtx1, vtx2] # vtx1 : [nverts, 3], vtx2 : [nverts, 3]
    view_angle = (90,-90)
    matplotlib_3d_ptcloud_list(points, view_angle)

    '''
    color_list = ['r', 'b', 'k', 'g', 'c']
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection='3d')

    for i, point3d in enumerate(point3d_list):
        if isinstance(point3d, np.ndarray):
            data = point3d.copy()
        elif torch.is_tensor(point3d):
            data = point3d.detach().cpu().numpy()

        xdata = data[:,0].squeeze()
        ydata = data[:,1].squeeze()
        zdata = data[:,2].squeeze()

        if view_angle is not None:
            ax.view_init(view_angle[0],view_angle[1])
        ax.scatter3D(xdata, ydata, zdata, marker='o', c=color_list[i])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plotting_points2d_on_img_cv(img, points2d, circle_size=2, color='red', vis_index=False):
    """
    img: np.ndarray, [H,W,3]
    points2d: [n_points,2]
    circle_size: plottin point size

    return display, [H,W,3]
    """
    if color == 'rainbow':
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(points2d) + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        display = img.copy()
        for i, (x, y) in enumerate(points2d):
            cv2.circle(display, (int(x), int(y)), circle_size, colors[i], -1)
            if vis_index:
                cv2.putText(display, str(i), (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i])

    else:
        color_dict = {
            'red':(0,0,255),
            'blue':(255,0,0),
            'green':(0,255,0)
        }
        display = img.copy()
        for i, (x, y) in enumerate(points2d):
            cv2.circle(display, (int(x), int(y)), circle_size, color_dict[color], -1)
            if vis_index:
                cv2.putText(display, str(i), (int(x) + 3, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_dict[color])

    return display

def vis_gaussian_heatmap_plt(heatmap, title='title'):
    """
    heatmap: torch.tensor or numpy.ndarray: [H,W] or [H,W,1]
    """
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

def vis_histogram(data, num_bins=1000):
    """
    data: vector. np.ndarray or torch.tensor
    """
    if isinstance(data, np.ndarray):
        pass
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    hist, bins = np.histogram(data, bins=num_bins)

    plt.hist(data, bins=bins)
    plt.xlabel('each value of data, m scale')
    plt.ylabel('the number of samples')
    plt.show()

def render_mesh(img, mesh, face, focal, princpt, color=None, alpha=0.5):
    """
    img: [H,W,3], np.ndarray
    mesh: [n_vtx, 3], np.ndarray, m scale, camera space coordinate
    face:
    focal: [fx, fy]
    princpt: [cx, cy]
    color_dict: set color in ['gold', 'blue', 'pink', 'gray']
    alpha: transparency
    """
    assert (alpha <= 1) & (alpha > 0)

    # mesh
    mesh = trimesh.Trimesh(mesh, face)
    rot = trimesh.transformations.rotation_matrix(
        np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    # set color
    color_dict = {
        'gold': [212. / 255., 175. / 255., 55. / 255.],
        'blue': [55. / 255., 175. / 255., 212. / 255.],
        'pink': [255. / 255., 105. / 255., 180. / 255.],
        'gray': [1 / 2, 1 / 2, 1 / 2],
        'white': [1.0, 1.0, 0.9, 1.0],
    }
    if color is not None:
        mesh_color = color_dict[color]
    else:
        # white
        mesh_color = (1.0, 1.0, 0.9, 1.0)

    material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                  baseColorFactor=(mesh_color[2], mesh_color[1], mesh_color[0], 1.0))

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    # focal, princpt = cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0], point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:, :, :3].astype(np.float32)
    valid_mask = (depth > 0)[:, :, None]
    valid_mask = np.where(valid_mask, alpha, 0)
    # save to image
    img = rgb * valid_mask + img * (1 - valid_mask)
    return img

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100):
    '''
        yaw : y-axis rotation, degree
        pitch : x-axis rotation, degree
        roll : z-axis rotation, degree

    '''
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),4)

    return img

def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size

    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y


    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img