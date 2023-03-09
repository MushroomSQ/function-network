import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import *
from graph.geometry_helpers import Obj_Interaction,Center_Node,Surround_Node,Space_Relationship
import random
import numpy as np
from graph.model_util import *

# from graph_extraction import RelationshipGraph

st_num = Space_Relationship.st_num
dr_num = st_num + Space_Relationship.dr_num
ds_num = dr_num + Space_Relationship.ds_num
vdr_num = ds_num + Space_Relationship.vdr_num

class GroundTruthNode():

    def __init__(self, category, adjacent, bbox, interaction=None, function = np.zeros((1,18)), id=None):
        self.category = category
        self.adjacent = adjacent
        self.bbox = bbox
        self.interaction = interaction
        self.function = function
        self.id = id

# graph net truth node
class Node():
    def __init__(self, category,interation = None,is_center=False):
        self.category = category
        self.interaction = interation
        self.is_center = is_center
        self.incoming = [] #Node is the end point of the edge
        self.outgoing = [] #Node is the starting point

class Graph_all():
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.state = []
        self.direct = []
        self.distance = []
        self.v_direct = []
        self._node_vectors = None
        self._node_one_hot = None
        self.edge_vectors = None
        self.u_indices = []
        self.v_indices = []
        self._node_vectors_backup = {}
        # self.backup("initial") #backup initial state
        self.num_center = 1

    @property
    def node_vectors(self):
        return self._node_vectors

    @node_vectors.setter #useless abstraction that end up only used once.
    def node_vectors(self, new_nv):
        self._node_vectors = new_nv

    def show(self):
        for i in range(len(self.nodes)):
            if self.nodes[i].is_center == True:
                print("node_category:{}".format(Center_Node.category[self.nodes[i].category]))
            else:
                print("node_category:{}".format(Surround_Node.category[self.nodes[i].category]))
        print("edge_types:{}".format(len(self.state)))
        if self.node_vectors != None:
            print("node vectors:{}".format(self.node_vectors.size()))

class GraphNet(nn.Module):
    """
    Main class
    """

    def __init__(self,normal_channel=False):
        super(GraphNet, self).__init__()

        self.pointnet2 = Pointnet2(GraphNetConfig.num_center_cat,normal_channel=normal_channel)

        self.criterion = get_loss()

        self.center_initializer = CenterInitializer()
        self.initializer = Initializer()

        self.f_interact = InteractionType()
        self.f_add_node = nn.ModuleList([AddNode() for i in range(GraphNetConfig.num_interaction)])
        self.f_edge_type = Edgetype()
        # self.f_box = LocationSize()

    def train_step(self, pointcloud, graph_data):
        # Training code
        # self.clear()  # clear internal states
        # optimizer.zero_grad()

        losses = {}
        losses["acn"] = torch.zeros(1)  # add center node
        losses["an"] = torch.zeros(1)  # add node
        losses["ita"] = torch.zeros(1)
        losses["edge"] = torch.zeros(1)  # choose node
        losses["box"] = torch.zeros(1)

        if GraphNetConfig.cuda:
            for key in losses.keys():
                losses[key] = losses[key].cuda()

        graph = {}
        self.batch_size = len(graph_data)
        self.node_size = 0

        logits_center_node,trans_feat = self.pointnet2(pointcloud)
        trans_feat = trans_feat.view(self.batch_size,1024)
        target = graph_data[:,0,0].int()
        losses["acn"] = self.criterion(logits_center_node,target.long())

        for j in range(self.batch_size):
            graph[j] = Graph_all()
            self.new_center_node(graph_data[j,0,0].int(), logits_center_node[j], trans_feat[j], graph[j])
            
            node_num = graph_data[j,0,1].int()
            self.node_size += node_num - 1
            nodes = graph_data[j,1:node_num,:]
            for (i,gt_node) in enumerate(nodes):#teacher force through all nodes in training data
                # First, train the add node module to predict the ground truth node category

                interaction_id, category = gt_node[0:2].int()
                # gt_adj, gt_edge_type, gt_direction = gt_node[2:8].int()
                gt_adj, gt_state, gt_direct, gt_dst, gt_v_direct, gt_direction = gt_node[2:8].int()
                bbox = gt_node[-6:]

                logits_interact = self._get_logits_interact_type(graph[j])
                target = interaction_id.view(-1)
                losses["ita"] += F.cross_entropy(logits_interact, target.long())

                logits_add_node = self._get_logits_add_node(interaction_id,graph[j])
                target = category.view(-1)
                losses["an"] += F.cross_entropy(logits_add_node, target.long())

                self.new_node(category, interaction_id, graph[j])

                logits_edge = self._get_logits_edge_type(graph[j])
                gt_idx = gt_adj * 2 + gt_direction
                
                target = (gt_idx * GraphNetConfig.num_edge_types + gt_state).view(-1)
                losses["edge"] += F.cross_entropy(logits_edge[:, :st_num], target.long())

                target = (gt_idx * GraphNetConfig.num_edge_types + gt_direct).view(-1)
                losses["edge"] += F.cross_entropy(logits_edge[:, st_num:dr_num], target.long())

                target = (gt_idx * GraphNetConfig.num_edge_types + gt_dst).view(-1)
                losses["edge"] += F.cross_entropy(logits_edge[:, dr_num:ds_num], target.long())

                target = (gt_idx * GraphNetConfig.num_edge_types + gt_v_direct).view(-1)
                losses["edge"] += F.cross_entropy(logits_edge[:, ds_num:vdr_num], target.long())


                if gt_direction == 0:  # Incoming to new node
                    self.new_edge(int(gt_adj),len(graph[j].nodes) - 1, gt_state, gt_direct, gt_dst, gt_v_direct, graph[j])
                else:  # Outgoing from new node
                    self.new_edge(len(graph[j].nodes) - 1, int(gt_adj), gt_state, gt_direct, gt_dst, gt_v_direct, graph[j])

                # target = bbox.view(1,-1)
                # logits_box = self._get_logits_box(graph[j])
                # losses["box"] += F.smooth_l1_loss(logits_box,target)

            # finished iterating over all nodes
            # finally, train the add node module to learn to stop
            logits_interact = self._get_logits_interact_type(graph[j])
            if GraphNetConfig.cuda:
                target = torch.zeros(1, dtype=torch.long).cuda()
            else:
                target = torch.zeros(1, dtype=torch.long)
            target[0] = GraphNetConfig.num_interaction
            losses["ita"] += F.cross_entropy(logits_interact, target)

        losses["ita"] = losses["ita"] / self.batch_size
        losses["an"] = losses["an"]/self.batch_size
        losses["edge"] = losses["edge"]/self.batch_size
        losses["box"] = losses["box"]/self.batch_size
        # print("************************************")
        # for i in range(self.batch_size):
        #     graph[i].show()
        return losses

    def predict(self, pointcloud):
        graph = Graph_all()

        logits_center_node, trans_feat = self.pointnet2(pointcloud)
        trans_feat = trans_feat.view(1,-1)
        pred = Categorical(logits=logits_center_node).sample()
        start_category = Center_Node.category[pred]
        self.new_center_node(pred, logits_center_node[0], trans_feat[0], graph)

        box_list = []
        success = 0
        count1 = 0
        edge_li = []
        interact_li = []
        # 加点
        while(True):
            logits_interact = self._get_logits_interact_type(graph)
            # interact_id = Categorical(logits=logits_interact).sample()
            interact_id = logits_interact[0].argmax()

            if interact_id == Obj_Interaction.number:
                break

            count1 += 1
            if count1 >= 30:
                success = 1
                print("add node have problem:{},count1:{}".format(pred, count1))
                break

            logits_add_node = self._get_logits_add_node(interact_id,graph)
            # node_id = Categorical(logits=logits_add_node).sample()
            node_id = logits_add_node[0].argmax()
            end_category = Surround_Node.category[node_id]

            interaction = Obj_Interaction.function[interact_id]
            interact_li.append(Ite_predict(start_category,end_category,interaction))
            self.new_node(node_id, interact_id, graph)

            # 加边
            logits_edge = self._get_logits_edge_type(graph)
            # edge_type = Categorical(logits=logits_edge).sample()
            state = logits_edge[0, :st_num].argmax()
            direct = logits_edge[0, st_num:dr_num].argmax()
            distance = logits_edge[0, dr_num:ds_num].argmax()
            v_direct = logits_edge[0, ds_num:vdr_num].argmax()
            self.new_edge(0, len(graph.nodes) - 1, state, direct, distance, v_direct, graph)
            edge_li.append(Edge_predict(start_category, end_category, int(state.cpu()), int(direct.cpu()), int(distance.cpu()), int(v_direct.cpu())))

            # box = self._get_logits_box(edge_type,graph)
            # box_list.append(box.cpu().numpy())

        return interact_li,edge_li,pred.cpu(),box_list,success


    def _get_logits_add_node(self, cat, graph):
        logits_add_node = self.f_add_node[cat](graph)
        return logits_add_node
    def _get_logits_edge_type(self, graph, edge_type=None):
        f_edge_type = self.f_edge_type
        logits_nodes = f_edge_type(graph, -1)
        return logits_nodes
    def _get_logits_interact_type(self,graph):
        logits_interact = self.f_interact(graph)
        return logits_interact
    def _get_logits_box(self, cat, graph):
        logits_box = self.f_box(graph)
        return logits_box

    # add center node
    def new_center_node(self, category, logits, pointnet_feature, graph):
        new_node = Node(category,is_center=True)
        feature_size = GraphNetConfig.num_center_cat
        feature_size += 1024
        if GraphNetConfig.cuda:
            e = torch.zeros(1, feature_size).cuda()
        else:
            e = torch.zeros(1, feature_size)
        e[0, :GraphNetConfig.num_center_cat] = logits
        e[0, GraphNetConfig.num_center_cat:] = pointnet_feature
        h_v = self.center_initializer(e)
        self._add_node(h_v,graph)
        graph.nodes.append(new_node)

    # add surround node
    def new_node(self, category, interaction_idx, graph):
        new_node = Node(category, interaction_idx)
        num_cat = GraphNetConfig.num_cat
        feature_size = num_cat
        feature_size += GraphNetConfig.num_interaction
        if GraphNetConfig.cuda:
            e = torch.zeros(1, feature_size).cuda()
        else:
            e = torch.zeros(1, feature_size)
        e[0, category] = 1
        e[0, interaction_idx + num_cat] = 1
        h_v = self.initializer(graph, e)
        self._add_node(h_v, graph)
        graph.nodes.append(new_node)

    def _add_node(self, h_v, graph):
        if graph._node_vectors is None:
            graph._node_vectors = h_v
        else:
            graph._node_vectors = torch.cat((graph._node_vectors, h_v), 0)

        # print(self._node_vectors.size())

    # add center ---> sur edge
    def new_edge(self, u_index, v_index, state, direct, dst, v_direct, graph):
        # assert edge_type is not None
        if GraphNetConfig.cuda:
            x_u_v = torch.zeros(1, GraphNetConfig.edge_size).cuda()
        else:
            x_u_v = torch.zeros(1, GraphNetConfig.edge_size)
        x_u_v[0, state] = 1
        x_u_v[0, st_num+direct] = 1
        x_u_v[0, dr_num+dst] = 1
        x_u_v[0, ds_num+v_direct] = 1
        self._add_edge(u_index, v_index, x_u_v, state, direct, dst, v_direct, graph)
    def _add_edge(self, u_index, v_index, x_u_v, state, direct, dst, v_direct, graph):
        if graph.edge_vectors is None:
            graph.edge_vectors = x_u_v
        else:
            graph.edge_vectors = torch.cat((graph.edge_vectors, x_u_v), 0)
        graph.u_indices.append(u_index)
        graph.v_indices.append(v_index)
        e_index = len(graph.edges)
        graph.edges.append((u_index, v_index))
        graph.nodes[u_index].outgoing.append(e_index)
        graph.nodes[v_index].incoming.append(e_index)
        graph.state.append(state)
        graph.direct.append(direct)
        graph.distance.append(dst)
        graph.v_direct.append(v_direct)

class Ite_predict():
    def __init__(self, start_category, end_category, interact):
        self.start_category = start_category
        self.end_category = end_category
        self.interact = interact

    def get(self):
        return self.start_category, self.end_category, self.interact

class Edge_predict():
    def __init__(self, start_category, end_category, state, drt, dst, v_drt):
        self.start_category = start_category
        self.end_category = end_category
        self.state = state
        self.direct = drt
        self.distance = dst
        self.v_direct = v_drt

    def get(self):
        return self.start_category, self.end_category, self.state, self.direct, self.distance, self.v_direct

if __name__ == '__main__':
    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))
    dir = "\\new_dataset\\scene_graph\\"
    object = "Desk"
    object_num = "40"
    path = root_dir + dir + object + "_" +object_num + ".pkl"
    # graph = RelationshipGraph()
    # graph.load_from_file(path)
    # gt_nodes, graph_gt_center = load_from_relation_graph(graph)
    #     # print(gt_nodes[0].adjacent)