import trimesh
import point_cloud_utils as pcu
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from graph.geometry_helpers import *
import random



class read_off():

    def __init__(self, object, object_num):
        self.object = object
        self.object_num = object_num
        self.model_name=[]
        self.object_meshs = {}
        self.object_name = {}
        self.inter_obj_label = {}
        self.scene_name = object + object_num
        self.scene_mesh = trimesh.base.Trimesh([])

        self.extract_mesh_and_label()

    def object_fution(self, mesh1, mesh2):
        scene = trimesh.util.concatenate(mesh1, mesh2)
        return scene

    # 提取off文件以及label：object、function
    def extract_mesh_and_label(self):
        dir = "dataset\\Crop_plane\\sceneslist_positive\\" + self.object + "\\" + self.object + "_" + self.object_num + ".txt"
        model_dir = "dataset\\Crop_plane\\models\\"
        label_dir = "label\\label\\" + self.object + "\\" + self.object + "_" + self.object_num + ".txt"
        inter_label = "label\\inter_obj_label\\" + self.object + "_" + self.object_num + "_inter_obj_label.txt"
        # print(dir)

        # object model
        f = open(dir)
        lines = f.readlines()
        f.close()
        model_num = int(lines[0].split(" ")[1].rstrip())
        self.model_name = [lines[i].split(" ")[2].rstrip() for i in range(2, model_num + 2)]
        path = [model_dir + self.model_name[i] + ".off" for i in range(model_num)]
        self.object_meshs = {self.model_name[i]: trimesh.load_mesh(path[i]) for i in range(model_num)}
        # print(self.meshs)
        # mesh1 = trimesh.load_mesh(path[0])
        # print(mesh1.bounding_box())

        # object categories
        f = open(label_dir)
        lines = f.readlines()
        f.close()
        self.label = [lines[i].split(" ")[1].rstrip() for i in range(model_num)]
        self.object_name = {self.model_name[i]: self.label[i] for i in range(model_num)}
        # print(self.labels)

        # object interaction label
        f = open(inter_label)
        lines = f.readlines()
        f.close()
        name = [lines[i].split(" ")[0].rstrip() for i in range(len(lines))]
        interlabel = [lines[i].split(" ")[2].rstrip() for i in range(len(lines))]
        self.inter_obj_label = {name[i]: interlabel[i] for i in range(len(lines))}
        # print(type(self.inter_obj_label.values()))
        # print(self.inter_obj_label.values())

        # 场景图
        for i in range(model_num):
            self.scene_mesh = self.object_fution(self.scene_mesh, self.object_meshs[self.model_name[i]])
        # print(self.scene.vertices.shape)
        # print(self.scene.faces.shape)


    class Node:
        def __init__(self, node_id, category_name,bbox, modelID, interaction = None,graph=None):
            self.id = node_id
            self.category_name = category_name
            self.bbox = bbox
            self.interaction = interaction
            self.modelID = modelID

    # 画3d框
    def plot3Dboxes(self,corners):
        for i in range(corners.shape[0]):
            corner = corners[i]
            self.plot3Dbox(corner)
    def plot3Dbox(self,corner):
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[0, idx]
        y = corner[1, idx]
        z = corner[2, idx]
        mlab.plot3d(x, y, z, color=(0.23, 0.6, 1), colormap='Spectral', representation='wireframe', line_width=5)
        # mlab.show()

    def get_nodes(self):
        nodes = []
        for key,value in self.object_name.items():
            mesh = self.object_meshs[key]
            # cloud = self.mesh2cloud(mesh)
            bbox = BBox3D()
            bbox.set_by_object(mesh.vertices)
            # mlab.points3d(cloud[:, 0], cloud[:, 1], cloud[:, 2], scale_factor=1.0, color=(0.7, 0.7, 0.7),resolution=10)
            # self.plot3Dboxes(bbox.corner())
            if key in self.inter_obj_label.keys():
                node = read_off.Node(key, value, bbox, self.scene_name, interaction=self.inter_obj_label[key])
            else:
                node = read_off.Node(key, value, bbox, self.scene_name)
            nodes.append(node)
        # mlab.view(azimuth=300, elevation=60)
        # mlab.show()
        # print(nodes)
        return nodes

    def show(self):
        print(self.model_name)
        # self.object_meshs[self.model_name[0]].show()
        self.scene_mesh.show()
        print(self.scene_mesh)

    def mesh2cloud(self,mesh):
        # mesh = self.object_meshs[self.model_name[0]]
        v = mesh.vertices
        f = mesh.faces
        n = mesh.vertex_normals
        n = np.asanyarray(n,order='C',dtype=np.float64)
        bbox = np.max(v, axis=0) - np.min(v, axis=0)
        bbox_diag = np.linalg.norm(bbox)
        v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(
            v, f, n, num_samples=-1, radius=0.01*bbox_diag, use_geodesic_distance=True)
        # print(v_poisson.shape)
        np.random.shuffle(v_poisson)
        v_poisson = v_poisson[:1024, :]
        return v_poisson

    #采样1024个点
    def sampe_cloud(self,num_points):
        mesh = self.object_meshs[self.model_name[0]]
        v_poisson = self.mesh2cloud(mesh)
        np.random.shuffle(v_poisson)
        v_poisson = v_poisson[:num_points, :]
        return v_poisson

    def save_cloud(self,point,path):
        print(point.shape)
        assert(point.shape[0]==1024)
        np.save(path,point)

    def cloud_show(self):
        mesh = self.object_meshs[self.model_name[0]]
        v_poisson = self.mesh2cloud(mesh)
        # np.random.shuffle(v_poisson)
        # v_poisson = v_poisson[:1024,:]
        point = mlab.points3d(v_poisson[:, 0], v_poisson[:, 1], v_poisson[:, 2], scale_factor=0.45, color=(0.7, 0.7, 0.7),
                              resolution=10)

        mlab.view(azimuth=300, elevation=60)
        # mlab.savefig('G:/project' + ps_files[i][:-4] + '.png',
        #              size=(500, 500))
        # mlab.clf()
        mlab.show()

if __name__ == '__main__':
    # show scene
    path = "new_dataset\\datalist.txt"
    f = open(path)
    lines = f.readlines()
    f.close()
    flag = 0
    for i in range(len(lines)):
        line = lines[i].rstrip('\n')
        object = line.split("_")[0]
        object_num = line.split("_")[1]
        if object == "Shelf":
            data = read_off(object, object_num)
            data.scene_mesh.show()
            for key in data.object_name:
                if data.object_name[key] == "Shelf":
                    data.object_meshs[key].show()


    # # 提取所有中心物体点云
    # path = "new_dataset\\datalist.txt"
    # f = open(path)
    # lines = f.readlines()
    # f.close()
    # root_dir = "new_dataset\\pointcloud"
    # flag = 0
    # for i in range(len(lines)):
    #     line = lines[i].rstrip('\n')
    #     object = line.split("_")[0]
    #     object_num = line.split("_")[1]
    #     print(object+object_num)
    #     data = read_off(object, object_num)
    #     dir = object + "_" + object_num
    #     path = root_dir + "\\" + dir + ".npy"
    #     data.save_cloud(path)


    ## 测试单个提取点云
    # object = "Backpack"
    # object_num = "02"
    # data = read_off(object, object_num)
    # root_dir = "new_dataset\\pointcloud"
    # dir = object + "_" + object_num
    # path = root_dir + "\\" + dir + ".npy"
    # data.save_cloud(path)
    # data.extract_mesh_and_label()
    # data.get_nodes()
    # data.show()
    # data.cloud_show()

    # 提取点云并存储在txt中
    # path = "new_dataset\\datalist_scene_test_9-1_25cate.txt"
    # f = open(path)
    # lines = f.readlines()
    # f.close()
    # # path = "new_dataset\\pointcloud"
    # for i in range(len(lines)):
    #     line = lines[i].rstrip('\n')
    #     line = line.split(".")[0]
    #     print(line)
    #     line = line.split("_")
    #     object = line[0]
    #     object_num = line[1]
    #     data = read_off(object,object_num)
    #     points = data.sampe_cloud(1024)
    #     name = object+"_"+object_num+"_1024.txt"
    #     np.savetxt("1024points/"+name, points, delimiter=' ')
    # object = "Basket"
    # object_num = "03"
    # data = read_off(object, object_num)
    # points = data.sampe_cloud(1024)
    # np.savetxt("Banana_1024.txt", points, delimiter=' ')

    # pqnet，提取bounding box大小和中心点,加入h5文件中
    # import math
    # import h5py
    # min_bounds = [0,0,0]
    # max_bounds = [0,0,0]
    # bbox_length = [0,0,0]
    # path = "new_dataset\\datalist.txt"
    # output_file_dir = "E:/代码+数据/voxe/scene/"
    # f = open(path)
    # lines = f.readlines()
    # f.close()
    # for i in range(len(lines)):
    #     line = lines[i].rstrip('\n')
    #     object = line.split("_")[0]
    #     object_num = line.split("_")[1]
    #     print(object+object_num)
    #     data = read_off(object, object_num)
    #     scene = data.scene_mesh
    #     scene_bounds = scene.bounds
    #     scene_bbox = scene.bounding_box.extents
    #     move_length = [0, 0, 0]
    #     for j in range(3):
    #         if scene_bounds[0][j] < 0:
    #             move_length[j] = math.ceil(abs(scene_bounds[0][j]))
    #     scene_name = object + "_" + object_num + ".h5"
    #     h5_file = output_file_dir + scene_name
    #     points = np.zeros((len(data.object_meshs),3))
    #     bbox = np.zeros((len(data.object_meshs),3))
    #     k = 0
    #     for value in data.object_meshs.values():
    #         object_bounds = value.bounds
    #         object_bbox = value.bounding_box.extents
    #         center_point = [object_bounds[0][j]+object_bbox[j]/2+move_length[j] for j in range(3)]
    #         for j in range(3):
    #             bbox[k][j] = round(object_bbox[j],1)
    #             points[k][j] = round(center_point[j],1)
    #         k += 1
    #     f = h5py.File(h5_file, 'a')
    #     f.attrs['n_parts'] = len(data.object_meshs)
    #     f['translations'] = points
    #     f['size'] = bbox

    # pqnet，提取json文件,object名对应所有场景,存放在{name}_info.json
    # from geometry_helpers import Center_Node
    # import json
    # path = "new_dataset\\datalist.txt"
    # f = open(path)
    # lines = f.readlines()
    # f.close()
    # scene_name_all = Center_Node.category
    # for j in range(Center_Node.number):
    #     scene = {}
    #     for i in range(len(lines)):
    #         line = lines[i].rstrip('\n')
    #         object = line.split("_")[0]
    #         object_num = line.split("_")[1]
    #         if object == scene_name_all[j]:
    #             print(object + object_num)
    #             data = read_off(object, object_num)
    #             scene_name = object + "_" + object_num
    #             parts = len(data.model_name)
    #             scene[scene_name] = parts
    #     print("----------------------------------------")
    #     jsondata = json.dumps(scene)
    #     output_file_dir = "D:/pycharm/project/pqnet/data/scene_all/{}_info.json".format(scene_name_all[j])
    #     with open(output_file_dir, "w+") as f:
    #         f.write(jsondata)
    #         print("加载入文件完成...")

    # pqnet，提取json文件,object名对应train test val场景,存放在{name}_info.json
    # import json
    # from geometry_helpers import Center_Node
    # scene_name_all = Center_Node.category
    # with open("D:/pycharm/project/pqnet/data/scene_info.json", "r") as f:
    #     jsondata = json.load(f)
    # f.close()
    # with open("D:/pycharm/project/try/new_dataset/datalist_scene_train_9-1_25cate.txt","r") as f:
    #     readlines = f.readlines()
    # f.close()
    # train_name = [readlines[i].strip().split(".")[0] for i in range(len(readlines))]
    # train = {}
    # test = {}
    # for key in jsondata.keys():
    #     if key in train_name:
    #         train[key] = jsondata[key]
    #     else:
    #         test[key] = jsondata[key]
    # for j in range(Center_Node.number):
    #     train_split = {}
    #     test_split = {}
    #     for key in train.keys():
    #         if key.split('_')[0] == scene_name_all[j]:
    #             train_split[key] = train[key]
    #     for key in test.keys():
    #         if key.split('_')[0] == scene_name_all[j]:
    #             test_split[key] = test[key]
    #     print(train_split)
    #     print(test_split)
    #     json_trian = json.dumps(train_split)
    #     json_test = json.dumps(test_split)
    #     train_output_file_dir = "D:/pycharm/project/pqnet/data/scene_train_test/{}.train.json".format(scene_name_all[j])
    #     test_output_file_dir = "D:/pycharm/project/pqnet/data/scene_train_test/{}.test.json".format(scene_name_all[j])
    #     val_output_file_dir = "D:/pycharm/project/pqnet/data/scene_train_test/{}.val.json".format(scene_name_all[j])
    #     with open(train_output_file_dir, "w+") as f:
    #         f.write(json_trian)
    #         print("加载入文件完成...")
    #     with open(test_output_file_dir, "w+") as f:
    #         f.write(json_test)
    #         print("加载入文件完成...")
    #     with open(val_output_file_dir, "w+") as f:
    #         f.write(json_test)
    #         print("加载入文件完成...")

    # 单种场景
    # import json
    # from geometry_helpers import Center_Node
    # scene_name_all = Center_Node.category
    # for j in range(Center_Node.number):
    #     all_file_dir = "D:/pycharm/project/pqnet/data/scene_all/{}_info.json".format(scene_name_all[j])
    #     train_output_file_dir = "D:/pycharm/project/pqnet/data/scene_train_test/{}.train.json".format(scene_name_all[j])
    #     test_output_file_dir = "D:/pycharm/project/pqnet/data/scene_train_test/{}.test.json".format(scene_name_all[j])
    #     with open(all_file_dir, "r") as f:
    #         all_jsondata = json.load(f)
    #     f.close()
    #     with open(train_output_file_dir, "r") as f:
    #         train_jsondata = json.load(f)
    #     f.close()
    #     with open(test_output_file_dir, "r") as f:
    #         test_jsondata = json.load(f)
    #     f.close()
    #     collect_train_data = []
    #     collect_test_data = []
    #     max = 0
    #     number = 0
    #     for key in all_jsondata.keys():
    #         name = key.split("_")[0]
    #         num = key.split("_")[1]
    #         number += all_jsondata[key]
    #         if max < all_jsondata[key]:
    #             max = all_jsondata[key]
    #     print(scene_name_all[j])
    #     print(number/len(all_jsondata.keys()))
    #     print(max)

    pass