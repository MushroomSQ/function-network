import numpy as np
from dataset.data_utils import collect_data_id
import json
import os
from mayavi import mlab
from graph.geometry_helpers import BBox3D
from util.visualization import partsdf2voxel,draw_voxel_model,draw_voxel_manymodel,sdf2voxel
import mcubes
import point_cloud_utils as pcu
import random

def compute_iou(affine1,affine2):
    '''
    :param bbox1: center(3) + size(3)
    :return: value of iou
    '''
    box1 = BBox3D()
    box1.set_by_bbox(affine1[0:3],affine1[3:6])
    box2 = BBox3D()
    box2.set_by_bbox(affine2[0:3], affine2[3:6])
    if not box1.isCollided(box2):
        return 0
    if box1.get_volume() == 0 or box2.get_volume() == 0:
        return 0
    bmin = []
    bmax = []
    for i in range(3):
        bmin.append(max(box1.min[i],box2.min[i]))
        bmax.append(min(box1.max[i],box2.max[i]))
    size = [bmax[i]-bmin[i] for i in range(3)]
    inter = size[0] * size[1] * size[2]
    all_v = box1.volume + box2.volume - inter
    return inter/all_v

def draw_points(points):
    mlab.points3d(points[:, 0], points[:, 1], points[:, 2], scale_factor=0.05, color=(0.7, 0.7, 0.7),
                  resolution=10)
    mlab.view(azimuth=300, elevation=60)
    mlab.show()




def load_part_data_info(data_root, class_name, phase):
    shape_names = collect_data_id(class_name, phase)
    with open('../data/{}_info.json'.format(class_name), 'r') as fp:
        nparts_dict = json.load(fp)
    parts_info = []
    for name in shape_names:
        shape_h5_path = os.path.join(data_root, name + '.h5')
        if not os.path.exists(shape_h5_path):  # check file existence
            continue
        parts_info.extend([(shape_h5_path, x) for x in range(nparts_dict[name])])

    return shape_names, parts_info

# voxel--->mesh,采样
def get_v_poisson(parts_voxel, sampling_threshold):
    vertices, triangles = mcubes.marching_cubes(parts_voxel, sampling_threshold)
    # mesh = trimesh.Trimesh(vertices,triangles)
    # print(mesh.bounding_box.extents)
    v = vertices
    f = triangles
    n = pcu.estimate_normals(vertices, k=16)
    number = 10000
    v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(
        v, f, n, number + 1000, use_geodesic_distance=True)
    # 打乱顺序
    indices = np.arange(len(v_poisson))
    random.shuffle(indices)
    v_poisson = v_poisson[indices[:number]]
    v_poisson = np.expand_dims(v_poisson, 0)
    return v_poisson

def get_scene_voxel(points, values, affine, resolution=64, is_show=True, save_path=None):
    n_parts = len(points)
    voxels = []
    cube_mid = np.asarray([resolution // 2, resolution // 2, resolution // 2]).reshape(1, 3)
    for idx in range(n_parts):
        part_points = points[idx]
        part_values = values[idx]
        part_translation = affine[idx, 0, :3].reshape(1, 3) * resolution
        part_size = affine[idx, 0, 3:6].reshape(1, 3)

        part_scale = np.max(part_size)
        part_points = (part_points - cube_mid) * part_scale + part_translation

        mins = part_translation - part_size * resolution / 2
        maxs = part_translation + part_size * resolution / 2
        in_bbox_indice = np.max(part_points - maxs, axis=1)
        in_bbox_indice = np.where(in_bbox_indice <= 0)[0]
        part_points = part_points[in_bbox_indice, :]
        part_values = part_values[in_bbox_indice]

        in_bbox_indice = np.max(part_points - mins, axis=1)
        in_bbox_indice = np.where(in_bbox_indice >= 0)[0]
        part_points = part_points[in_bbox_indice, :]
        part_values = part_values[in_bbox_indice]

        part_points = np.clip(part_points, 0, resolution - 1)

        part_points = np.expand_dims(part_points, 0)
        part_values = np.expand_dims(part_values, 0)
        # draw_voxel_model(partsdf2voxel(part_points, part_values))
        voxels.append(partsdf2voxel(part_points, part_values, resolution))
    if is_show is True or save_path is not None:
        draw_voxel_manymodel(voxels, n_parts,is_show=is_show, save_path=save_path)
    scene_voxel = np.zeros((resolution, resolution, resolution))
    for idx in range(n_parts):
        scene_voxel += voxels[idx]
    return scene_voxel,voxels

def get_center_trans_point_value(points, values, affine, resolution=64, save_path=None):
    n_parts = len(points)
    num_points = points.size(1)
    center_trans = np.zeros((n_parts, num_points, 3))
    cube_mid = np.asarray([resolution // 2, resolution // 2, resolution // 2]).reshape(1, 3)

    center_points = points[0]
    center_values = values[0]
    part_translation = affine[0, 0, :3].reshape(1, 3) * resolution
    part_size = affine[0, 0, 3:6].reshape(1, 3)
    part_scale = np.max(part_size)
    center_points = (center_points - cube_mid) * part_scale + part_translation
    # draw_voxel_model(sdf2voxel(center_points, center_values))

    # 中心物体从全局转换到局部
    center_trans[0] = center_points
    for idx in range(1,n_parts):
        part_translation = affine[idx, 0, :3].reshape(1, 3) * resolution
        part_size = affine[idx, 0, 3:6].reshape(1, 3)

        part_scale = np.max(part_size)
        part_points = (center_points - part_translation) / part_scale + cube_mid
        center_trans[idx] = part_points

    return center_trans

def get_parts_trans_point_value(points, values, affine, resolution=64, save_path=None):
    n_parts = len(points)
    parts_points = []
    parts_values = []
    trans_points = []
    cube_mid = np.asarray([resolution // 2, resolution // 2, resolution // 2]).reshape(1, 3)
    for idx in range(n_parts):
        part_points = points[idx]
        part_values = values[idx]
        part_translation = affine[idx, 0, :3].reshape(1, 3) * resolution
        part_size = affine[idx, 0, 3:6].reshape(1, 3)

        part_scale = np.max(part_size)
        part_points = (part_points - cube_mid) * part_scale + part_translation
        trans_points.append(part_points)

        mins = part_translation - part_size * resolution / 2
        maxs = part_translation + part_size * resolution / 2
        in_bbox_indice = np.max(part_points - maxs, axis=1)
        in_bbox_indice = np.where(in_bbox_indice <= 0)[0]
        part_points = part_points[in_bbox_indice, :]
        part_values = part_values[in_bbox_indice]

        in_bbox_indice = np.max(part_points - mins, axis=1)
        in_bbox_indice = np.where(in_bbox_indice >= 0)[0]
        part_points = part_points[in_bbox_indice, :]
        part_values = part_values[in_bbox_indice]

        part_points = np.clip(part_points, 0, resolution - 1)

        part_points = np.expand_dims(part_points, 0)
        part_values = np.expand_dims(part_values, 0)

        parts_points.append(part_points)
        parts_values.append(part_values)
    return parts_points, parts_values, trans_points

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)