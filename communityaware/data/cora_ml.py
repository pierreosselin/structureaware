from typing import Callable, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import CitationFull
from torch_geometric.io import read_npz

from communityaware.data.utils import assign_graph_ids, split_data


class CoraML(CitationFull):

    def __init__(self, root, training_nodes_per_class=20, transform=None, pre_transform=None):
        self.training_nodes_per_class = training_nodes_per_class
        super().__init__(root, 'cora_ml', transform=transform, pre_transform=pre_transform)

    def download(self):
        return super().download()

    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.data, slices = self.collate([data])

        idx = np.arange(len(data.y))
        train_idx, valid_test_idx = train_test_split(idx, train_size = self.num_classes * self.training_nodes_per_class, stratify=data.y)
        valid_idx, test_idx = train_test_split(valid_test_idx, train_size = self.num_classes * self.training_nodes_per_class, stratify=data.y[valid_test_idx])

        self.data.train_mask = torch.zeros(self.data.num_nodes, dtype=bool).scatter_(0, torch.tensor(train_idx), 1)
        self.data.val_mask = torch.zeros(self.data.num_nodes, dtype=bool).scatter_(0, torch.tensor(valid_idx), 1)
        self.data.test_mask = torch.zeros(self.data.num_nodes, dtype=bool).scatter_(0, torch.tensor(test_idx), 1)
        self.data.idx = torch.tensor(0)

        torch.save((self.data, slices), self.processed_paths[0])

    def make_noise_matrix(self, p_add, p_delete):
        noise = np.ones((self.data.num_nodes, self.data.num_nodes)) * p_add
        for index in self.data.edge_index.numpy().T:
            noise[index[0], index[1]] = p_delete
        return noise

    @property
    def testset_length(self):
        return len(self.data.test_mask)
