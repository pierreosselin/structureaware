
import os.path as osp
from typing import Union

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (Compose, RemoveIsolatedNodes,
                                        ToUndirected)

from communityaware.data.utils import AssignID, split_data


class TUDataset(InMemoryDataset):
    def __init__(self, name, train_split=0.5, test_split=20, cleaned=False):
        self.name = name
        self.train_split = train_split
        self.test_split = test_split
        self.cleaned = cleaned
        #pre_transform = Compose([RemoveIsolatedNodes(), ToUndirected(), AssignID()])
        self.dataset = torch_geometric.datasets.TUDataset('data', name, cleaned=cleaned, pre_filter = lambda graph: graph.num_nodes > 6)
        super().__init__(f'data/{name}')
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'transformed')

    @property
    def processed_file_names(self):
        return f'{self.name}_{self.cleaned}_{self.train_split}_{self.test_split}.pt'

    def process(self):
        self.data = self.dataset.data
        self.slices = self.dataset.slices
        self.split = split_data(self.dataset.data.y, self.train_split, self.test_split)
        torch.save((self.data, self.slices, self.split), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        shuffle = True if split=='train' else False
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size, shuffle=shuffle)

    @property
    def testset_labels(self):
        return torch.concat([self[i].y for i in self.split['test']])

if __name__=='__main__':
    dataset = TUDataset('AIDS', cleaned=False)
    print(min([x.num_nodes for x in dataset.dataset]))
