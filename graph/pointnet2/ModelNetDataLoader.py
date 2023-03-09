import os
from torch.utils.data import Dataset
import numpy as np
import sys
import json

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self, root,  npoint=1024, split='train', category='easyscene', uniform=False, normal_channel=False, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [name.rstrip() + '.npy' for name in json.load(open(os.path.join(self.root, 'train_val_test_split', f'{category}.train.json'))).keys()]
        shape_ids['test'] = [name.rstrip() + '.npy' for name in json.load(open(os.path.join(self.root, 'train_val_test_split', f'{category}.test.json'))).keys()]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('.')[0:-1]) for x in shape_ids[split]]
        # print(shape_names)
        self.datapath = [(shape_names[i], os.path.join(self.root, 'graph', "pointcloud_normal", shape_ids[split][i])) for i
                         in range(len(shape_ids[split]))]
        self.scene_datapath = [(shape_names[i], os.path.join(self.root, 'graph', "scene_graph_normal_numpy", shape_ids[split][i])) for i
                         in range(len(shape_ids[split]))]
        # print(self.datapath)
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            # print("index:{}".format(index))
            point_set, graph_set = self.cache[index]
        else:
            fn = self.datapath[index]
            gn = self.scene_datapath[index]
            if fn[0] == gn[0]:
                point_set = np.load(fn[1]).astype(np.float32)
                graph_set = np.load(gn[1]).astype(np.float32)
            else:
                point_set = None
                graph_set = None

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            # point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, graph_set)

        return point_set, graph_set

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('graph/new_dataset', split='train', uniform=False, normal_channel=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=5, shuffle=False)
    for point,graph in DataLoader:
        print(point.shape)
        print(graph.shape)
        # for i in range(len(graph)):
        #     node_num = torch.where(graph[i] == -2)[0][0]
        #     print(node_num)
