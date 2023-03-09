import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from graph.geometry_helpers import Obj_Interaction,Center_Node,Surround_Node,Space_Relationship
from graph.pointnet2.pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction

#Different linear layers for ablation
class Linear1(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear1, self).__init__()
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(in_size, out_size),
                         )
        else:
            self.model = nn.Linear(in_size, out_size)

    def forward(self, x):
        return self.model(x)

class Linear2(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear2, self).__init__()
        if GraphNetConfig.hidden_size is None:
            hidden_size = max(in_size, out_size)
        else:
            hidden_size = GraphNetConfig.hidden_size
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(hidden_size, out_size),
                         )
        else:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, out_size),
                         )

    def forward(self, x):
        return self.model(x)

class Linear3(nn.Module):
    def __init__(self, in_size, out_size, hidden_size=None, dropout=None):
        super(Linear3, self).__init__()
        if GraphNetConfig.hidden_size is None:
            hidden_size = max(in_size, out_size)
        else:
            hidden_size = GraphNetConfig.hidden_size
        if dropout is not None:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Dropout(p=GraphNetConfig.dropout_p),
                            nn.Linear(hidden_size, out_size),
                         )
        else:
            self.model = nn.Sequential(
                            nn.Linear(in_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, hidden_size),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_size, out_size),
                         )

    def forward(self, x):
        return self.model(x)

layer_name_to_class = {
    "1": Linear1,
    "2": Linear2,
    "3": Linear3,
}

class GraphNetConfig():

    num_center_cat = Center_Node.number
    num_interaction = Obj_Interaction.number
    num_cat = Surround_Node.number
    num_edge_types = Space_Relationship.number# number of edge types

    cen_categories = Center_Node.category
    sur_categories = Surround_Node.category

    node_size = 64  # Size of node embedding
    hG_multiplier = 2  # Size of graph embedding / size of node embedding

    # 打乱顺序
    shuffle_nodes = False #shuffle nodes when training
    shuffle_edges = False #shuffle edges when training

    # 传播、聚合、初始化模块 选用线性层1
    propagation_layer = Linear1 #single layer doesn't really work, final version uses Linear3
    aggregation_layer = Linear1
    initializing_layer = Linear1
    cuda = False #GPU training is slower

    hidden_size = None  # Size of hidden layers of linear modules, none means pick the layer of in_size & out_size

    decision_module_names = ["an", "edge", "ita", "box"]

    decision_layer = Linear1
    decision_layer_dict = {}
    rounds_of_propagation = 3 #T
    rounds_of_propagation_dict = {} #This can be different for each module

    init_with_graph_representation = True  # If true, graph embedding is used to initialize new nodes
    include_one_hot = False  # If true, include one hot category encodings for message passing
    choose_node_graph_vector = False  # If true, include graph embedding when choosing node to connect edge to

    predict_edge_type_first = False
    # If true, AddEdge predicts the edge type instead of just boolean indicating add or not

    # separate_wall_edges_step = True
    # # If true, Use different sets of weights for wall edges and handle them separately

    node_and_type_together = False  # If true, predict which node to connect to and which edge to connect to together
    everything_together = False
    # Implies the above one, if true, skip add edge step and use a fixed logit to represent "stop adding edge" instead
    no_edge_logit = math.log(10)  # Used if everything_together, choose this value so logit of 0 is roughly prob of 0.1

    auxiliary_choose_node = False  # Forgot what this option is
    per_node_prediction = False  # Instead of softmaxing over all nodes, predciting node wise if there is an edge

    dropout_p = None

    @classmethod
    def compute_derived_attributes(cls):

        # cls.categories += ["Stop"] #Add "stop" category

        if cls.everything_together:
            cls.node_and_type_together = True

        cls.hG_size = cls.node_size * cls.hG_multiplier  # Size of graph embedding
        cls.edge_size = cls.num_edge_types # 修改

        # if cls.per_node_prediction: #Special case, special treatment. Well this is not used at all now
        #     cls.decision_module_names = ["an", "et"]

        # Change layer type from strings to actual classes
        if isinstance(cls.propagation_layer, str):
            cls.propagation_layer = layer_name_to_class[cls.propagation_layer]
        if isinstance(cls.initializing_layer, str):
            cls.initializing_layer = layer_name_to_class[cls.initializing_layer]

        if isinstance(cls.aggregation_layer, str):
            cls.aggregation_layer = layer_name_to_class[cls.aggregation_layer]
        if isinstance(cls.decision_layer, str):
            cls.decision_layer = layer_name_to_class[cls.decision_layer]

        for name in cls.decision_module_names:
            if not name in cls.rounds_of_propagation_dict:
                cls.rounds_of_propagation_dict[name] = cls.rounds_of_propagation  # set deault T
            if not name in cls.decision_layer_dict:
                cls.decision_layer_dict[name] = cls.decision_layer  # set default decision layer type
            elif isinstance(cls.decision_layer_dict[name], str):
                cls.decision_layer_dict[name] = layer_name_to_class[cls.decision_layer_dict[name]]  # also update string to class names

""" ==================================================================
                           Propagation
================================================================== """
class Propagator(nn.Module):
    def __init__(self, rounds_of_propagation):
        super(Propagator,self).__init__()
        self.rounds_of_propagation = rounds_of_propagation

        # Models for propagating the state vector
        #Gathering message from an adjacent node
        #Separate weights for rounds_of_propagation rounds

        node_size = GraphNetConfig.node_size
        edge_size = GraphNetConfig.edge_size
        message_size = node_size * 2 + edge_size

        # Forward direction
        self.f_ef = nn.ModuleList([
            GraphNetConfig.propagation_layer(message_size, node_size * 2)
            for i in range(rounds_of_propagation)])

        # Reverse direction
        self.f_er = nn.ModuleList([
            GraphNetConfig.propagation_layer(message_size, node_size * 2)
            for i in range(rounds_of_propagation)])


        output_size = node_size
        # Mapping aggregated message vector to new node state vector
        self.f_n = nn.ModuleList([
            nn.GRUCell(node_size * 2, output_size)
            for i in range(rounds_of_propagation)])

    def forward(self, gn):
        if len(gn.nodes) == 0 or len(gn.edges) == 0:
            return
        for i in range(self.rounds_of_propagation):
            # compute messages
            messages_raw = torch.cat((gn.node_vectors[gn.u_indices],
                                      gn.node_vectors[gn.v_indices],
                                      gn.edge_vectors), 1)
            messages_forward = self.f_ef[i](messages_raw)
            messages_reverse = self.f_er[i](messages_raw)

            # aggregate messages
            aggregated = None
            for node in gn.nodes:  # this should be ideally done with scatter...
                if GraphNetConfig.cuda:
                    a_v = torch.zeros(1, GraphNetConfig.node_size * 2).cuda()
                else:
                    a_v = torch.zeros(1, GraphNetConfig.node_size * 2)
                if len(node.incoming) > 0:
                    a_v += messages_forward[node.incoming].sum(dim=0)
                if len(node.outgoing) > 0:
                    a_v += messages_reverse[node.outgoing].sum(dim=0)
                if aggregated is None:
                    aggregated = a_v
                else:
                    aggregated = torch.cat((aggregated, a_v), 0)

            # update node embedding
            gn.node_vectors = self.f_n[i](aggregated, gn._node_vectors)

""" ==================================================================
                           Aggregation
================================================================== """
class Aggregator(nn.Module):
    """
    Aggregates information across nodes to create a graph vector
    """
    def __init__(self):
        super(Aggregator, self).__init__()
        node_size = GraphNetConfig.node_size
        hG_size = GraphNetConfig.hG_size

        # Model for computing graph representation
        self.f_m = GraphNetConfig.aggregation_layer(node_size, hG_size)
        #Gated parameter when aggregating for graph representation
        self.g_m = nn.Sequential(
            GraphNetConfig.aggregation_layer(node_size, hG_size),
            nn.Sigmoid()
        )

    def forward(self, gn):
        #Default for empty graph
        if len(gn.nodes) == 0:
            if GraphNetConfig.cuda:
                h_G = torch.zeros(1, GraphNetConfig.hG_size).cuda()
            else:
                h_G = torch.zeros(1, GraphNetConfig.hG_size)
        else:
            h_G = (self.f_m(gn.node_vectors) * self.g_m(gn.node_vectors)).sum(dim=0).unsqueeze(0)
        return h_G

""" ==================================================================
                           Node module
================================================================== """
class InteractionType(nn.Module):
    """
    Decide whether to add and the interaction type
    """

    def __init__(self):
        super(InteractionType, self).__init__()
        hG_size = GraphNetConfig.hG_size

        if GraphNetConfig.rounds_of_propagation_dict["ita"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["ita"])
        self.aggre = Aggregator()

        num_interact_classes = GraphNetConfig.num_interaction
        output = num_interact_classes
        self.f_interact = GraphNetConfig.decision_layer_dict["ita"](hG_size, output + 1,
                                                                    dropout=GraphNetConfig.dropout_p)

    def forward(self, gn):
        if GraphNetConfig.rounds_of_propagation_dict["ita"] > 0:
            self.prop(gn)
        aggre = self.aggre(gn)
        return self.f_interact(aggre)

class AddNode(nn.Module):
    """
    Decide what type of nodes
    """

    def __init__(self):
        super(AddNode, self).__init__()
        num_cat = GraphNetConfig.num_cat
        hG_size = GraphNetConfig.hG_size
        self.aggre = Aggregator()

        self.f_add_node = GraphNetConfig.decision_layer_dict["an"](hG_size, num_cat,
                                                                   dropout=GraphNetConfig.dropout_p)

    def forward(self, gn):
        return self.f_add_node(self.aggre(gn))

""" ==================================================================
                           Edge module
================================================================== """
class Edgetype(nn.Module):
    """
    Decide which node to add an edge to
    """
    def __init__(self):
        super(Edgetype, self).__init__()
        node_size = GraphNetConfig.node_size

        if GraphNetConfig.choose_node_graph_vector:  # if uses graph embedding, need to include this to compute graph embedding
            self.aggre = Aggregator()

        if GraphNetConfig.rounds_of_propagation_dict["edge"] > 0:
            self.prop = Propagator(GraphNetConfig.rounds_of_propagation_dict["edge"])

        if GraphNetConfig.choose_node_graph_vector:
            in_size = node_size * 2 + GraphNetConfig.hG_size
        else:
            in_size = node_size * 2

        out_size = 2
        if GraphNetConfig.node_and_type_together:
            out_size = out_size * GraphNetConfig.num_edge_types

        self.f_s = GraphNetConfig.decision_layer_dict["edge"](in_size, out_size, dropout=GraphNetConfig.dropout_p)

    def forward(self, gn, target_idx):
        if GraphNetConfig.rounds_of_propagation_dict["edge"] > 0:
            self.prop(gn)

        # compute the logits for all pairs
        concat = torch.cat((gn.node_vectors[:gn.num_center], gn.node_vectors[target_idx].repeat(gn.num_center, 1)), 1)

        if GraphNetConfig.choose_node_graph_vector:
            h_G = self.aggre(gn)
            concat = torch.cat((concat, h_G.repeat(concat.size()[0], 1)), 1)

        return self.f_s(concat).view(1, -1)

""" ==================================================================
                           Bbox module
================================================================== """
class LocationSize(nn.Module):
    """
    Decide location and size of the node bbox
    """
    def __init__(self):
        super(LocationSize, self).__init__()

        node_size = GraphNetConfig.node_size
        hG_size = GraphNetConfig.hG_size
        # input_size = node_size * 3

        input_size = hG_size
        output = 6
        self.aggre = Aggregator()
        # self.init_edge = EdgeInitializer()
        # # Model for computing graph representation
        # self.f_m = GraphNetConfig.aggregation_layer(node_size, hG_size)
        # # Gated parameter when aggregating for graph representation
        # self.g_m = nn.Sequential(
        #     GraphNetConfig.aggregation_layer(node_size, hG_size),
        #     nn.Sigmoid()
        # )
        self.f_box = GraphNetConfig.decision_layer_dict["box"](input_size, output,
                                                                    dropout=GraphNetConfig.dropout_p)

    def forward(self, gn):
        # h_e = self.init_edge(gn.edge_vectors[-1].view(1,-1))
        # h_v = gn.node_vectors[-1].view(1,-1)
        # h_u = gn.node_vectors[0].view(1,-1)
        # concat = torch.cat((h_e,h_v,h_u),0)
        # h_G = (self.f_m(concat) * self.g_m(concat)).sum(dim=0).unsqueeze(0)
        h_G = self.aggre(gn)
        return self.f_box(h_G)

class LocationSize_All(nn.Module):
    """
    Decide location and size of the node bbox
    """
    def __init__(self):
        super(LocationSize_All, self).__init__()

        node_size = GraphNetConfig.node_size
        input_size = node_size * 3
        output = 6
        self.init_edge = EdgeInitializer()
        self.f_box = GraphNetConfig.decision_layer_dict["box"](input_size, output,
                                                                    dropout=GraphNetConfig.dropout_p)

    def forward(self, gn):
        if gn.edge_vectors is None:
            return
        h_e = self.init_edge(gn.edge_vectors)
        h_v = gn.node_vectors[0].repeat(len(gn.edge_vectors), 1)
        h_u = gn.node_vectors[1:]
        # vector = gn.node_vectors[-1].unsqueeze(0) + gn.node_vectors[0].unsqueeze(0)
        concat = torch.cat((h_e,h_v,h_u),1)
        return self.f_box(concat)
""" ==================================================================
                           Init Node
================================================================== """
# 中心物体映射
class CenterInitializer(nn.Module):
    """
    init center node embedding
    """
    def __init__(self):
        super(CenterInitializer, self).__init__()


        node_size = GraphNetConfig.node_size
        in_size = GraphNetConfig.num_center_cat
        in_size += 1024

        self.f_init = GraphNetConfig.initializing_layer(in_size, node_size)
        self.tanh = nn.Tanh()

    def forward(self, e):
        h_v = self.f_init(e)
        return self.tanh(h_v)

class Initializer(nn.Module):
    """
    init node embedding
    """

    def __init__(self):
        super(Initializer, self).__init__()

        node_size = GraphNetConfig.node_size
        num_cat = GraphNetConfig.num_cat

        # Models for initializing
        # For initializing state vector of a new node
        feature_size = num_cat
        feature_size += GraphNetConfig.num_interaction

        if GraphNetConfig.init_with_graph_representation:
            hG_size = GraphNetConfig.hG_size
            self.f_init = GraphNetConfig.initializing_layer(feature_size + hG_size, node_size)
            self.aggre = Aggregator()
        else:
            self.f_init = GraphNetConfig.initializing_layer(feature_size, node_size)

        self.tanh = nn.Tanh()

    def forward(self, gn, e):
        # One hot for now
        if GraphNetConfig.init_with_graph_representation:
            h_G = self.aggre(gn)
            h_v = self.f_init(torch.cat((e, h_G), 1))
        else:
            h_v = self.f_init(e)
        return self.tanh(h_v)

class EdgeInitializer(nn.Module):
    """
    init edge_type embedding
    """
    def __init__(self):
        super(EdgeInitializer, self).__init__()

        node_size = GraphNetConfig.node_size
        in_size = GraphNetConfig.num_edge_types

        self.f_init = GraphNetConfig.initializing_layer(in_size, node_size)
        self.tanh = nn.Tanh()

    def forward(self, e):
        h_v = self.f_init(e)
        return self.tanh(h_v)

""" ==================================================================
                           PointNet++
================================================================== """
class Pointnet2(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(Pointnet2, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x,l3_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.cross_entropy(pred, target)

        return total_loss