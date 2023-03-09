from torch.utils.data import Dataset
import torch
import numpy as np
import os
import json
import random
from dataset.data_utils import collect_data_id, load_from_hdf5_partae_graph
import time

# Part AE dataset
######################################################
class GraphAEDataset(Dataset):
    def __init__(self, phase, data_root, class_name, points_batch_size, all_points=False, resolution=64):
        super(GraphAEDataset, self).__init__()
        self.data_root = data_root
        self.class_name = class_name
        self.parts_info = self.load_part_data_info(phase)
        self.phase = phase
        self.points_batch_size = points_batch_size

        self.all_points = all_points
        self.resolution = resolution

    def load_part_data_info(self, phase):
        shape_names = collect_data_id(self.class_name, phase)
        parts_info = []
        for name in shape_names:
            shape_h5_path = os.path.join(self.data_root, 'easyscene', name + '.h5')
            if not os.path.exists(shape_h5_path):  # check file existence
                continue
            parts_info.extend([shape_h5_path])
        return parts_info

    def __getitem__(self, index):
        shape_path = self.parts_info[index]
        n_parts, parts_voxel, data_point, data_value, nodes, edges, pointcloud, affine, data_category = load_from_hdf5_partae_graph(shape_path, self.resolution)

        # shuffle selected points

        if not self.all_points and len(data_point[0]) > self.points_batch_size:
            data_points = np.zeros((n_parts[0], self.points_batch_size, 3))
            data_values = np.zeros((n_parts[0], self.points_batch_size, 1))
            for i in range(n_parts[0]):
                indices = np.arange(len(data_points[0]))
                random.shuffle(indices)
                # np.random.shuffle(indices)
                indices = indices[:self.points_batch_size]
                data_points[i,:] = data_point[i, indices]
                data_values[i,:] = data_value[i, indices]
        else:
            data_points = np.array(data_point)
            data_values = np.array(data_value)

        batch_nparts = torch.tensor(n_parts, dtype=torch.int8)
        batch_voxels = torch.tensor(parts_voxel.astype(np.float), dtype=torch.float32).unsqueeze(1)  # (1, dim, dim, dim)
        batch_points = torch.tensor(data_points, dtype=torch.float32)  # (points_batch_size, 3)
        batch_values = torch.tensor(data_values, dtype=torch.float32)  # (points_batch_size, 1)
        batch_nodes = torch.tensor(nodes).long()
        batch_edges = torch.tensor(edges).long()
        batch_clouds = torch.tensor(pointcloud.astype(np.float), dtype=torch.float32).unsqueeze(0) #(1, 1024, 3)
        batch_affine =  torch.tensor(affine, dtype=torch.float32)
        batch_category =  torch.tensor(data_category, dtype=torch.float32)

        return {"vox3d": batch_voxels,
                "points": batch_points,
                "values": batch_values,
                "nodes":batch_nodes,
                "edges":batch_edges,
                "pointcloud":batch_clouds,
                "path":shape_path,
                "affine":batch_affine,
                "categories": batch_category,
                "n_parts": batch_nparts}

    def __len__(self):
        return len(self.parts_info)

def graph_collate_fn(batch):
    all_nodes = []
    all_edges = []
    all_voxel = []
    all_parts = []
    all_data_points = []
    all_data_values = []
    all_nodes_to_graph = []
    all_edges_to_graph = []
    all_pointcloud = []
    all_path = []
    all_affine = []
    all_category = []
    node_offset = 0
    for i, data in enumerate(batch):
        voxel = data["vox3d"]
        points = data["points"]
        values = data["values"]
        node = data["nodes"]
        edge = data["edges"]
        n_parts = data["n_parts"]
        pointcloud = data["pointcloud"]
        path = data["path"]
        affine = data["affine"]
        categories = data["categories"]
        if node.dim() == 0 or edge.dim() == 0:
            continue
        N, E = node.size(0), edge.size(0)
        all_nodes.append(node)

        edge[:, 0] += node_offset
        edge[:, 5] += node_offset
        all_edges.append(edge)
        all_nodes_to_graph.append(torch.LongTensor(N).fill_(i))
        all_edges_to_graph.append(torch.LongTensor(E).fill_(i))
        node_offset += N

        all_data_points.append(points)
        all_data_values.append(values)

        all_voxel.append(voxel)
        all_parts.append(n_parts)
        all_pointcloud.append(pointcloud)
        all_path.append(path)
        all_category.append(categories)
        all_affine.append(affine[1:])
        # all_affine.append(affine)

    all_nodes = torch.cat(all_nodes)
    all_edges = torch.cat(all_edges)
    all_voxel = torch.cat(all_voxel)
    all_parts = torch.cat(all_parts)
    all_category = torch.cat(all_category)

    all_data_points = torch.cat(all_data_points)
    all_data_values = torch.cat(all_data_values)
    all_nodes_to_graph = torch.cat(all_nodes_to_graph)
    all_edges_to_graph = torch.cat(all_edges_to_graph)
    all_pointcloud = torch.cat(all_pointcloud)
    all_affine = torch.cat(all_affine)
    out = {"vox3d":all_voxel,
           "points":all_data_points,
           "values":all_data_values,
           "nodes":all_nodes,
           "edges":all_edges,
           "n_parts":all_parts,
           "categories":all_category,
           "pointcloud":all_pointcloud,
           "path":all_path,
           "affine":all_affine,
           "nodes_to_graph": all_nodes_to_graph,
           "edges_to_graph": all_edges_to_graph}
    return out

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = GraphAEDataset("train", "D:/pycharm/project/pqnet/data", "easyscene", 16384)
    dataloader = DataLoader(dataset, batch_size=15, shuffle=False,
                            num_workers=2, collate_fn=graph_collate_fn)
    for data in dataloader:
        print("--------------------------")
        data_parts = data["n_parts"]
        print("parts",data_parts.shape)
        # nodes = data["nodes"]
        # print("nodes",nodes.shape)
        # edges = data["edges"]
        # print("edges",edges.shape)
        # voxel = data["vox3d"]
        # print("voxel",voxel.shape)
        # data_points = data["points"]
        # print("points",data_points.shape)
        # data_values = data["values"]
        # print("values", data_values.shape)
        print("--------------------------")


    pass
