from torch.utils.data import DataLoader
from dataset.dataset_partae import PartAEDataset
from dataset.dataset_partae_graph import GraphAEDataset, graph_collate_fn
import numpy as np


def get_dataloader(phase, config, use_all_points=False, is_shuffle=None):
    is_shuffle = phase == 'train' if is_shuffle is None else is_shuffle

    if config.module == 'part_ae':
        dataset = PartAEDataset(phase, config.data_root, config.category, config.points_batch_size,
                                all_points=use_all_points, resolution=config.resolution)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers, worker_init_fn=np.random.seed())
    elif config.module == 'graph':
        dataset = GraphAEDataset(phase, config.data_root, config.category, config.points_batch_size,
                                all_points=use_all_points, resolution=config.resolution)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle,
                                num_workers=config.num_workers, collate_fn=graph_collate_fn)
    else:
        raise NotImplementedError
    return dataloader
