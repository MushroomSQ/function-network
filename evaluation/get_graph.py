import argparse
import os
import torch
import importlib
import numpy as np
from graph.geometry_helpers import Obj_Interaction,Center_Node,Surround_Node,All_Node
from graph.graph_extraction import RelationshipGraph
from graph.model import GraphNet

torch.backends.cudnn.enabled=False


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class autoregressive_graph():
    def __init__(self, load_model_path, graph_path):
        self.load_model_path = load_model_path
        self.graph_path = graph_path
        self.load_network()

    def load_network(self):
        model = 'graph.model'
        print("load autoregressive graph model from: {}".format(self.load_model_path))

        '''graph config'''
        shuffle_nodes = False
        shuffle_edges = False
        cuda = True
        config = {
            "shuffle_nodes": shuffle_nodes,
            "shuffle_edges": shuffle_edges,
            "hidden_size": 384,
            "initializing_layer": "3",
            "propagation_layer": "3",
            "aggregation_layer": "3",
            "choose_node_graph_vector": True,
            "node_and_type_together": True,
            "init_with_graph_representation": True,
            "everything_together": True,
            "include_one_hot": False,
            "cuda": cuda,
            "rounds_of_propagation_dict": {"ita": 3,"edge":3},
        }
        '''MODEL LOADING'''
        MODEL = importlib.import_module(model)

        for (key, value) in config.items():
            setattr(MODEL.GraphNetConfig, key, value)
        MODEL.GraphNetConfig.compute_derived_attributes()
        # print(MODEL.GraphNetConfig.__dict__)

        if cuda:
            classifier = MODEL.GraphNet(normal_channel=False).cuda()
        else:
            classifier = MODEL.GraphNet(normal_channel=False)

        checkpoint = torch.load(self.load_model_path)
        classifier.load_state_dict(checkpoint)
        classifier.eval()
        self.classifier = classifier
    
    def get_graph(self, pointcloud, data_id):
        
        # points = np.load(self.root_pointcloud + data_id + ".npy")
        # points = points[0: 1024,:]
        # points = torch.from_numpy(points).float().unsqueeze(0)
        # points = points.transpose(2, 1)
        # points = points.cuda()
        pointcloud = pointcloud.transpose(2, 1)

        interact_list,predict_edge_list,center_id,box,success = self.classifier.predict(pointcloud)

        edge_num = len(predict_edge_list)
        node_list = np.zeros((edge_num + 1,2))
        edge_list = np.zeros((edge_num,6))

        node_list[0, 0] = center_id
        node_list[0, 1] = Obj_Interaction.number

        list_pre = []
        for i in range(len(interact_list)):
            start, end, edge_type = interact_list[i].get()
            edge = "{}---{}---{}".format(start, edge_type, end)
            print(edge)
            list_pre.append(edge)
            node_list[i+1, 0] = All_Node.category.index(end)
            node_list[i+1, 1] = Obj_Interaction.function.index(edge_type)
        for i in range(len(predict_edge_list)):
            start, end, state, drt, dst, v_drt = predict_edge_list[i].get()
            edge = "{} --- {}-{}-{}-{} --- {}".format(start, state, drt, dst, v_drt, end)
            print(edge)
            list_pre.append(edge)
            edge_list[i, 0:6] = np.array([0, state, drt, dst, v_drt, i+1])
        
        with open(self.graph_path, 'a') as f:
            f.write(data_id + ": ")
            f.write(str(list_pre) + '\n')

        node_list = torch.tensor(node_list).long()
        edge_list = torch.tensor(edge_list).long()
        N, E = node_list.size(0), edge_list.size(0)
        nodes_to_graph = torch.LongTensor(N).fill_(0)
        edges_to_graph = torch.LongTensor(E).fill_(0)

        return node_list, edge_list, nodes_to_graph, edges_to_graph
        
            # print(node_list.shape)
            # print(edge_list.shape)


if __name__ == '__main__':
    auto_graph = autoregressive_graph()
    data_id = 'Bathtub_0095'
    auto_graph.get_graph(data_id)
    # main(args, data_id)