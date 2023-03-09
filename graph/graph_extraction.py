from graph.geometry_helpers import *
from graph.read_off import read_off
import pickle
import os
import copy
from mayavi import mlab

class RelationshipGraph:
    class Node:
        def __init__(self, node_id, category_name, bbox, interaction = None, \
                     out_edge_indices=[], in_edge_indices=[],\
                     function = np.zeros((1,18)), modelID=None, graph=None):
            self.id = node_id
            self.category_name = category_name
            self.bbox = bbox
            self.interaction = interaction
            self.function = function
            self.__out_edge_indices = out_edge_indices
            self.__in_edge_indices = in_edge_indices
            self.__modelID = modelID
            self.__graph = graph

        def __repr__(self):
            rep = f'{self.category_name} ({self.id}) --- bbox({self.bbox})--- {self.interaction}'
            return rep

        # -----------------------------属性------------------------------------------
        #出边、入边、所有边的数量
        @property
        def out_edges(self):
            return [self.__graph.edges[i] for i in self.__out_edge_indices]
        @property
        def in_edges(self):
            return [self.__graph.edges[i] for i in self.__in_edge_indices]
        @property
        def all_edges(self):
            return self.in_edges + self.out_edges

        #出边邻居、入边邻居、所有邻居
        @property
        def out_neighbors(self):
            return [e.end_node for e in self.out_edges]
        @property
        def in_neighbors(self):
            return [e.start_node for e in self.in_edges]
        @property
        def all_neighbors(self):
            return list(set(self.in_neighbors + self.out_neighbors))

        #是否为边的功能：support
        @property
        def is_second_tier(self):
            return len([e for e in self.in_edges if e.edge_type.is_support]) > 0

        @property
        def modelID(self):
            if not hasattr(self, '_Node__modelID'):
                setattr(self, '_Node__modelID', None)
            return self.__modelID
        def set_modelID(self, mid):
            self.__modelID = mid

        # ------------------------------方法-----------------------------------------
        #边处理
        def clear_edges(self):
            self.__out_edge_indices = []
            self.__in_edge_indices = []
        def add_out_edge(self, edge_idx):
            self.__out_edge_indices.append(edge_idx)
        def add_in_edge(self, edge_idx):
            self.__in_edge_indices.append(edge_idx)

        #图
        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Node(self.id, self.category_name, self.bbox, self.interaction, \
                self.__out_edge_indices, self.__in_edge_indices, \
                self.function, self.modelID, graph)
        def without_graph(self):
            return RelationshipGraph.Node(self.id, self.category_name, self.bbox, self.interaction, \
                self.__out_edge_indices, self.__in_edge_indices, \
                self.function, self.modelID)
        # ---------------------------------------------------------------------------

    # 空间边
    class Edge:
        def __init__(self, start_id, end_id, edge_type,
                graph=None):
            self.__start_id = start_id
            self.__end_id = end_id
            self.edge_type = edge_type
            # self.dist = dist
            # self.target_percentage_visible = target_percentage_visible
            # self.anchor_percentage_visible = anchor_percentage_visible
            self.__graph = graph

        def __repr__(self):
            edge_name = self.edge_type
            edge_name = f'{edge_name}'
            return f'{self.start_node.category_name} ({self.start_node.id}) ---- {edge_name} ---> {self.end_node.category_name} ({self.end_node.id})'

        # -----------------------------属性------------------------------------------
        @property
        def start_node(self):
            assert(self.__graph is not None)
            return self.__graph.get_node_by_id(self.__start_id)
        @property
        def end_node(self):
            assert(self.__graph is not None)
            return self.__graph.get_node_by_id(self.__end_id)
        @property
        def neighbors(self):
            return self.start_node, self.end_node

        # -----------------------------方法------------------------------------------

        def with_graph(self, graph):
            if self.__graph == graph:
                return self
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, self.edge_type,graph)
        def without_graph(self):
            return RelationshipGraph.Edge(self.__start_id, self.__end_id, self.edge_type)

        # ---------------------------------------------------------------------------

    # class information:
    #     def __init__(self,node_category,interaction,edge_type):
    #         self.node_category = node_category
    #         self.interaction = interaction
    #         self.edge_type = edge_type
    #
    #     def __repr__(self):
    #         return f'{self.node_category} - {self.interaction} - {self.edge_type}'

        # def compare_to(self,inf):
        #     if self.node_category == inf.node_category and self.interaction == inf.interaction \
        #             and self.edge_type == inf.edge_type:
        #         return True
        #     return False

    # ******************************************************************************************************************
    # (Main body for RelationshipGraph)
    def __init__(self, nodes=[], edges=[]):
        self.__nodes = {n.id:n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()

    @property
    def nodes(self):
        return list(self.__nodes.values())

    def __record_node_edges(self):
        for node in self.nodes:
            node.clear_edges()
        for edge_idx in range(len(self.edges)):
            edge = self.edges[edge_idx]
            edge.start_node.add_out_edge(edge_idx)
            edge.end_node.add_in_edge(edge_idx)

    def get_node_by_id(self, id_):
        if not id_ in self.__nodes:
            print(f'Could not find node with id {id_}')
        return self.__nodes[id_]

    def add_node(self, node):
        self.__nodes[node.id] = node.with_graph(self)

    def extract_function(self,read):
        inter_obj_label = read.inter_obj_label
        function = np.zeros((1,18))
        line = list(inter_obj_label.values())
        for value in line:
            function[0,Obj_Interaction.function.index(value)] = 1
        return function

    def extract_from_data(self,object,object_num):

        read = read_off(object, object_num)

        self.__nodes.clear()
        self.edges.clear()

        nodes = read.get_nodes()

        # 提取节点
        for i, node in enumerate(nodes):
            node_id = node.id
            # print(node_id)
            category = node.category_name
            interaction = node.interaction
            # pos = node.pos
            # volumn = node.volume
            bbox = node.bbox
            if(i==0):
                # 提取中心物体功能种类
                function = self.extract_function(read)
                # print(function)
                self.add_node(RelationshipGraph.Node(
                    node_id, category, bbox, interaction, function = function, modelID=node.modelID, graph=self
                ))
            else:
                self.add_node(RelationshipGraph.Node(
                    node_id, category, bbox, interaction, modelID=node.modelID, graph=self
                ))

        self.order_node()
        center_node = nodes[0]
        for i in range(1,len(nodes)):
            start = center_node.id
            end = nodes[i].id
            # print(nodes[i].bbox.is_add_edge(center_node.bbox))
            state, direction, distance = nodes[i].bbox.relation_to(center_node.bbox)
            # print(f'{Space_Relationship.state[state]} --- {Space_Relationship.Direction[direction]} --- {Space_Relationship.Distance[distance]}')
            edgetype = state * 18 + direction * 3 + distance
            # print(edgetype)
            self.edges.append(RelationshipGraph.Edge(
                start,end,edgetype,graph = self
            ))
        self.__record_node_edges()

    def order_node(self):
        keys = Obj_Interaction.function
        obj = Surround_Node.category
        nodes = self.nodes
        center = self.nodes[0]

        # inteaction 按字典形式存放
        list_ita = {}
        for key in keys:
            list_ita[key] = []
        for i in range(1,len(nodes)):
            node_ita = nodes[i].interaction
            list_ita[node_ita].append(i)

        # 对 interaction 中的内容进行排序
        for key in keys:
            list_obj = list_ita[key]
            if len(list_obj) > 1 :
                obj_order = []
                obj_order.append(list_obj[0])
                for i in range(1,len(list_obj)):
                    category_idx = obj.index(nodes[list_obj[i]].category_name)
                    flag = 0
                    for j in range(len(obj_order)):
                        order_category_idx = obj.index(nodes[obj_order[j]].category_name)
                        # category id 相同 ， 按bbox大小排序
                        if category_idx == order_category_idx:
                            volume = nodes[list_obj[i]].bbox.get_volume()
                            order_volume = nodes[obj_order[j]].bbox.get_volume()
                            if volume < order_volume:
                                obj_order.insert(j, list_obj[i])
                                flag = 1
                                break
                        # category id不同 , 按从小到大排序
                        elif category_idx < order_category_idx:
                            obj_order.insert(j,list_obj[i])
                            flag = 1
                            break
                    if flag == 0:
                        obj_order.append(list_obj[i])
                list_ita[key] = obj_order
        list_all = []
        for key in keys:
            if list_ita[key]:
                list_all = list_all + list_ita[key]

        self.__nodes.clear()
        self.add_node(center)
        for i in range(len(list_all)):
            self.add_node(nodes[list_all[i]])

    def change_bbox(self, centroid, scale):
        num = len(self.nodes)
        for i in range(num):
            self.nodes[i].bbox.min -=centroid
            self.nodes[i].bbox.max -=centroid
            self.nodes[i].bbox.min /= scale
            self.nodes[i].bbox.max /= scale
        # for i in range(num):
        #     print(self.nodes[i].bbox)
        # print("-------------------------")

    def show(self):
        print(self.__nodes)
        print(self.edges)
        # mlab.show()

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            nodes = [n.without_graph() for n in self.nodes]
            edges = [e.without_graph() for e in self.edges]
            pickle.dump((nodes, edges), f, pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, filename):
        # fname = os.path.split(filename)[1]
        # self.id = int(os.path.splitext(fname)[0])
        with open(filename, 'rb')as f:
            nodes, edges = pickle.load(f)
        self.__nodes = {n.id:n.with_graph(self) for n in nodes}
        self.edges = [e.with_graph(self) for e in edges]
        self.__record_node_edges()
        return self

if __name__ == '__main__':

    root_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_list_path = os.path.join('new_dataset', 'datalist.txt')
    f = open(dataset_list_path)
    dataset_lines = f.readlines()
    f.close()

    ## 提取所有场景图和中心物体点云
    graph_dir = os.path.join('new_dataset', 'scene_graph')
    point_dir = os.path.join('new_dataset', 'pointcloud')
    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)
    if not os.path.exists(point_dir):
        os.mkdir(point_dir)
    for i in range(len(dataset_lines)):
        line = dataset_lines[i].rstrip('\n')
        object = line.split("_")[0]
        object_num = line.split("_")[1]
        print(object+object_num)

        graph = RelationshipGraph()
        graph.extract_from_data(object, object_num)
        # graph.show()
        graph_path = os.path.join(root_dir, graph_dir, object + "_" + object_num + ".pkl")
        graph.save_to_file(graph_path)

        data = read_off(object, object_num)
        points = data.sampe_cloud(1024)
        pc_path = os.path.join(root_dir, point_dir, object + "_" + object_num + ".npy")
        data.save_cloud(pc_path)

    ## 读取场景图数据和点云数据 并进行归一化
    graph_dir = os.path.join('new_dataset', 'scene_graph')
    point_dir = os.path.join('new_dataset', 'pointcloud')
    save_graph_dir = os.path.join('new_dataset', 'scene_graph_normal')
    save_point_dir = os.path.join('new_dataset', 'pointcloud_normal')
    if not os.path.exists(save_graph_dir):
        os.mkdir(save_graph_dir)
    if not os.path.exists(save_point_dir):
        os.mkdir(save_point_dir)
    from geometry_helpers import BBox3D
    for i in range(len(dataset_lines)):
        line = dataset_lines[i].rstrip('\n')
        object = line.split("_")[0]
        object_num = line.split("_")[1]
        print(object+object_num)
        graph_filename = os.path.join(root_dir, graph_dir, object + "_" + object_num + ".pkl")
        point_filename = os.path.join(root_dir, point_dir, object + "_" + object_num + ".npy")
        points = np.load(point_filename)
        bbox = BBox3D(points)
        centroid = bbox.get_CenterPoint()
        # centroid = np.mean(points, axis=0)
        pc = points - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        bbox_scale = BBox3D(pc)
        graph = RelationshipGraph()
        graph.load_from_file(graph_filename)
        graph.change_bbox(centroid,m)
        graph_path = os.path.join(root_dir, save_graph_dir, object + "_" + object_num + ".pkl")
        graph.save_to_file(graph_path)
        points_path = os.path.join(root_dir, save_point_dir, object + "_" + object_num + ".npy")
        np.save(points_path, pc)




    # normal表达方式 画scene bbox 和 点云
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    # path = "new_dataset/datalist.txt"
    # f = open(path)
    # lines = f.readlines()
    # f.close()
    # save_graph_dir = "/new_dataset/scene_graph_normal/"
    # save_point_dir = "/new_dataset/pointcloud_normal/"
    # from draw_bbox import plot3Dbox
    # for i in range(len(lines)):
    #     line = lines[i].rstrip('\n')
    #     object = line.split("_")[0]
    #     object_num = line.split("_")[1]
    #     if object == "Basket":
    #         graph_filename = root_dir + save_graph_dir + object + "_" + object_num + ".pkl"
    #         graph = RelationshipGraph()
    #         graph.load_from_file(graph_filename)
    #         graph.show()
    #         for j in range(len(graph.nodes)):
    #             color = (0,1,1)
    #             plot3Dbox(graph.nodes[j].bbox.corner()[0],color)
    #         point_filename = root_dir + save_point_dir + object + "_" + object_num + ".npy"
    #         v_poisson = np.load(point_filename)
    #         point = mlab.points3d(v_poisson[:, 0], v_poisson[:, 1], v_poisson[:, 2], scale_factor=0.05,
    #                               color=(0.7, 0.7, 0.7),
    #                               resolution=10)
    #
    #         mlab.view(azimuth=300, elevation=60)
    #         mlab.show()



    pass