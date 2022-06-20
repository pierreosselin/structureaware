
from typing import Union

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (Compose, RemoveIsolatedNodes,
                                        ToUndirected)

from communityaware.data.utils import AssignID, split_data


class DataSet(InMemoryDataset):
    def __init__(self, name, root='data', train_split=0.5, test_split=20, cleaned=True):
        self.name = name
        self.train_split = train_split
        self.test_split = test_split
        self.cleaned = cleaned
        self.dataset = TUDataset('data', name, cleaned=cleaned, pre_transform=Compose([
            ToUndirected(),
            RemoveIsolatedNodes(),
            AssignID(),
        ]))
        super().__init__(root)
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return self.dataset.raw_file_names

    @property
    def processed_file_names(self):
        return f'{self.name}_{self.cleaned}_{self.train_split}_{self.test_split}.pt'

    def download(self):
        pass

    def process(self):
        self.split = split_data(self.dataset.data.y, self.train_split, self.test_split)
        self.data = self.dataset.data
        self.slices = self.dataset.slices
        torch.save((self.data, self.slices, self.split), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size)

    @property
    def testset_labels(self):
        return torch.concat([self[i].y for i in self.split['test']])
