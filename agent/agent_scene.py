import torch
import torch.nn as nn
from agent.base import BaseAgent
from util.visualization import project_voxel_along_xyz, visualize_sdf
from networks import get_network, set_requires_grad
import os
from util import provider
import torch.optim as optim
# import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
# import copy
# import numpy as np


class GraphAE(BaseAgent):
    def __init__(self, config):
        super(GraphAE, self).__init__(config)
        self.points_batch_size = config.points_batch_size
        self.resolution = config.resolution
        self.batch_size = config.batch_size

    def build_net(self, config):
        # restore part encoder
        part_imnet = get_network('part_ae', config)
        if not os.path.exists(config.partae_modelpath):
            raise ValueError("Pre-trained part_ae path not exists: {}".format(config.partae_modelpath))
        part_imnet.load_state_dict(torch.load(config.partae_modelpath)['model_state_dict'])
        print("Load pre-trained part AE from: {}".format(config.partae_modelpath))
        # if config.freeze:
        # self.part_ae = part_imnet.cuda().eval()
        # set_requires_grad(self.part_ae, requires_grad=False)
        self.part_encoder = part_imnet.encoder.cuda().eval()
        self.part_decoder = part_imnet.decoder.cuda().eval()
        set_requires_grad(self.part_encoder, requires_grad=False)
        set_requires_grad(self.part_decoder, requires_grad=False)
        del part_imnet
        # build rnn
        net = get_network('graph', config).cuda()

        return net

    def set_loss_function(self):
        self.criterion1 = nn.MSELoss().cuda()
        self.criterion2 = nn.MSELoss().cuda()

    def forward(self, data):
        input_vox3d = data['vox3d'].cuda()  # (shape_batch_size, 1, dim, dim, dim)
        categories = data['categories'].cuda()
        z = self.part_encoder(input_vox3d, categories)
        n_parts = data['n_parts'].cuda()
        # center = z.view(-1, 3, 128)[:, 0, :]

        nodes = data['nodes'].cuda()
        edges = data['edges'].cuda()
        nodes_to_graph = data['nodes_to_graph'].cuda()
        edges_to_graph = data['edges_to_graph'].cuda()
        affines = data['affine'].cuda()
        # pointcloud = data['pointcloud'].numpy()
        # pointcloud = provider.random_point_dropout(pointcloud)
        # pointcloud = torch.Tensor(pointcloud)
        # pointcloud = pointcloud.cuda()
        pointcloud = data['pointcloud'].cuda()
        node_vec,edge_vec = self.net(nodes, edges, nodes_to_graph, edges_to_graph, pointcloud)

        nv_loss = self.criterion1(node_vec, z)
        af_loss = self.criterion2(edge_vec, affines)

        # point_batch_size = points.size(1)
        # batch_z = z.unsqueeze(1).repeat((1, point_batch_size, 1)).view(-1, z.size(1))
        # batch_points = points.view(-1, 3)
        # output_sdf = self.part_ae.decoder(batch_points, batch_z)
        # pt_loss = self.criterion(output_sdf, target_sdf)

        return [z, node_vec], {"nv_loss": nv_loss,"af_loss": af_loss}
        

    def trans_part2global(part_points, affine):
        cube_mid = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).reshape(1, 3).cuda()

        part_translation = affine[0:3].unsqueeze(0).repeat((262144, 1))
        part_size = affine[3:].unsqueeze(0).repeat((262144, 1))
        part_scale = torch.max(part_size, 1)[0].view(262144, 1).cuda()

        part_points = ((part_points - cube_mid) / resolution) * part_scale + part_translation
        return part_points

    '''def set_optimizer(self, config):
        """set optimizer and lr scheduler used in training"""
        self.optimizer = optim.Adam(
            self.net.parameters(), 
            lr=config.lr, 
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, config.lr_step_size)'''

    def visualize_batch(self, data, mode, outputs=None):
        tb = self.train_tb if mode == 'train' else self.val_tb

        parts_voxel = data['vox3d'][0][0].numpy()
        data_points64 = data['points'][0].numpy() * self.resolution
        data_values64 = data['values'][0].numpy()
        output_sdf = outputs[0].detach().cpu().numpy()

        target = visualize_sdf(data_points64, data_values64, concat=True, vox_dim=self.resolution)
        output = visualize_sdf(data_points64, output_sdf, concat=True, vox_dim=self.resolution)
        voxel_proj = project_voxel_along_xyz(parts_voxel, concat=True)
        tb.add_image("voxel", torch.from_numpy(voxel_proj), self.clock.step, dataformats='HW')
        tb.add_image("target", torch.from_numpy(target), self.clock.step, dataformats='HW')
        tb.add_image("output", torch.from_numpy(output), self.clock.step, dataformats='HW')
