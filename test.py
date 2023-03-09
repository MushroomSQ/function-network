from config import get_config
from tqdm import tqdm
import torch
from dataset import get_dataloader
from networks import get_network
import os
import h5py
from util.utils import ensure_dir
from agent import PQNET
import numpy as np
from util.visualization import partsdf2voxel,draw_voxel_model,draw_voxel_manymodel
from evaluation.eva_util import get_scene_voxel, get_v_poisson
import point_cloud_utils as pcu
from evaluation.get_graph import autoregressive_graph

def save_output(points, values, affine, scene_voxel, data_id, save_dir):
    save_path = os.path.join(save_dir, "{}.h5".format(data_id))
    with h5py.File(save_path, 'w') as fp:
        fp.create_dataset('points', data=points, compression=9)
        fp.create_dataset('values', data=values, compression=9)
        fp.create_dataset('affine', data=affine, compression=9)
        fp.create_dataset('scene_voxel', data=scene_voxel, compression=9)

def get_score(part_voxel, voxels):
    v_truth = get_v_poisson(part_voxel, 0.5)
    v_predict = get_v_poisson(voxels, 0.5)
    chamfer_Metrix = pcu.pairwise_distances(v_truth, v_predict)
    dis1 = chamfer_Metrix.min(1).sum(1)/chamfer_Metrix.shape[2]# predict匹配groundtruth
    dis2 = chamfer_Metrix.min(2).sum(1)/chamfer_Metrix.shape[1]
    chamfer_distance = dis1 + dis2
    return chamfer_distance

def main():
    config = get_config('pqnet')('test')
    config.batch_size = 1
    config.num_worker = 1
    config.g = 1

    fake_name = "test"
    save_dir = os.path.join(config.proj_dir, "results/{}".format(fake_name))
    ensure_dir(save_dir)
    graph_save_path = os.path.join(save_dir, 'graph.txt')
    open(graph_save_path, 'w').close()

    pqnet = PQNET(config)

    category_path = os.path.join(config.proj_dir, config.category)
    if os.path.exists(category_path):
        load_model_path = os.path.join(config.proj_dir, config.category, 'classification', 'model', 'best.pth')
    else:
        load_model_path = os.path.join(config.proj_dir, 'easyscene', 'classification', 'model', 'best.pth')

    atgraph = autoregressive_graph(load_model_path, graph_save_path)

    test_loader = get_dataloader('test', config)
    pbar = tqdm(test_loader)


    for data in pbar:
        data_id = data['path'][0].split('/')[-1].split('.')[0]
        with torch.no_grad():
            print(data_id)
            # groundtruth
            pointcloud = data['pointcloud'].cuda()
            # nodes = data['nodes'].long().cuda()
            # edges = data['edges'].long().cuda()
            # nodes_to_graph = data['nodes_to_graph'].cuda()
            # edges_to_graph = data['edges_to_graph'].cuda()
            
            # predict graph
            nodes, edges, nodes_to_graph, edges_to_graph = atgraph.get_graph(pointcloud, data_id)
            nodes = nodes.cuda()
            edges = edges.cuda()
            nodes_to_graph = nodes_to_graph.cuda()
            edges_to_graph = edges_to_graph.cuda()

            # print(nodes_to_graph.shape)
            # print(edges_to_graph)
            batch_size = 1
            n_parts = len(nodes)

            # 保存位置文件夹不存在就创建
            save_hdf5 = os.path.join(save_dir,"hdf5")
            ensure_dir(save_hdf5)
            save_picture = os.path.join(save_dir, "picture")
            ensure_dir(save_picture)
            save_picture = os.path.join(save_picture, data_id)

            # 数据进入网络生成z和bbox
            node_vecs, edge_vecs = pqnet.graph(nodes, edges, nodes_to_graph, edges_to_graph, pointcloud)
            node_vecs = node_vecs.view(batch_size, n_parts, -1).transpose(0, 1)

            pqnet.n_parts = n_parts
            pqnet.output_part_codes = node_vecs

            resolution = 64
            x = np.arange(0, resolution)
            y = np.arange(0, resolution)
            z = np.arange(0, resolution)
            X, Y, Z = np.meshgrid(x, y, x)
            point = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1))).astype(np.float32)
            point = torch.FloatTensor(point).cuda()
            part_point = point.unsqueeze(0).repeat(n_parts,1,1)

            values = []
            for idx in range(pqnet.n_parts):
                part_points = part_point[idx] / resolution
                part_values = pqnet.eval_part_points(part_points, idx).astype(np.double)
                part_values = part_values.reshape((resolution * resolution * resolution,1))
                values.append(part_values)
            
            points = part_point.cpu().numpy()

            # replace center voxel
            center_voxel = data['vox3d'].numpy().astype(np.float)[0][0]
            center_values = np.zeros((64 * 64 * 64, 1))
            mm = 0
            for j in range(64):
                for k in range(64):
                    for l in range(64):
                        center_values[mm,0] = center_voxel[j, k, l]
                        mm = mm + 1
            values[0] = center_values


            # center object in position（0，0，0）, size（1，1，1）
            affine_temp = edge_vecs.view(n_parts-1,1,6).cpu().numpy()
            center_affine = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float).reshape(1, 1, 6)
            affine_temp = np.concatenate((center_affine, affine_temp), axis=0)
            max_point = np.empty((3))
            min_point = np.empty((3))
            for i in range(n_parts):
                for j in range(3):
                    min_p = affine_temp[i,0,j]-affine_temp[i,0,j+3]/2
                    max_p = affine_temp[i,0,j]+affine_temp[i,0,j+3]/2
                    if i==0:
                        min_point[j] = min_p
                        max_point[j] = max_p
                    else:
                        if min_p<min_point[j]:
                            min_point[j] = min_p
                        if max_p>max_point[j]:
                            max_point[j]=max_p
            size_scene = np.max(max_point - min_point)
            center_scene = (max_point+min_point)/2
            translation = ((affine_temp[:, 0, 0:3]-center_scene)/size_scene).reshape(n_parts,3)
            size = (affine_temp[:, 0, 3:6]/size_scene).reshape(n_parts,3)
            translation = translation + 0.5
            affine = np.concatenate((translation, size), axis=1).reshape(n_parts,1,6)
            
            # save_picture = None
            scene_voxel, voxels = get_scene_voxel(points, values, affine, resolution = resolution, is_show=False, save_path=save_picture)
            save_output(points, values, affine_temp, scene_voxel, data_id, save_hdf5)


if __name__ == '__main__':
    main()
