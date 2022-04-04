import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from os.path import join, isdir
import os

class Synthetic(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.process()
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])
        
    def download(self):
        # todo: move script/generate_data.py to here.
        raise NotImplementedError

    @property
    def raw_dir(self):
        return join(self.root, 'synthetic', 'raw')
    
    @property
    def raw_file_names(self):
        return ['graphs.pt', 'split.pt']

    @property
    def processed_dir(self):
        return join(self.root, 'synthetic', 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt',]

    def process(self):
        if not (os.path.isfile(self.raw_paths[0]) and os.path.isfile(self.raw_paths[1])):
            raise FileNotFoundError('Raw files not found. Run scripts/generate_data.py first.')
        data_list = torch.load(self.raw_paths[0])
        split = torch.load(self.raw_paths[1])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
    
        data, slices = self.collate(data_list)
        torch.save((data, slices, split), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size)