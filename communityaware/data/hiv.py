import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from os.path import join, isdir
import os
import warnings
from torchdrug import datasets
import pickle
import torchdrug
from tqdm import tqdm 
from rdkit.Chem import AtomValenceException

class HIV(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    def download(self):
        with warnings.catch_warnings(): # torch drug throws warnings about not recognising some atom types.
            warnings.simplefilter("ignore")
            dataset = datasets.HIV(self.root)
        torch.save(dataset, self.raw_paths[0])

    @property
    def raw_dir(self):
        return join(self.root, 'HIV', 'raw')
    
    @property
    def raw_file_names(self):
        return ['graphs.pt']

    @property
    def processed_dir(self):
        return join(self.root, 'HIV', 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt',]

    def process(self):
        dataset = torch.load(self.raw_paths[0])
        data_list = []
        graphs_not_processed = 0.0
        for datapoint in tqdm(dataset):
            try:
                data_list.append(to_torch_geometric(datapoint['graph'], datapoint['HIV_active']))
            except AtomValenceException:
                graphs_not_processed += 1

        print(f'Graphs not processed: {graphs_not_processed} of {len(dataset)}.')

        if self.pre_filter is not None:
             data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
             data_list = [self.pre_transform(data) for data in data_list]
    
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size)


def to_torch_geometric(graph: torchdrug.data.molecule.Molecule, label: int):
        """Convert a data sample from a torchdrug dataset to a pytorch dataset with communities membership determined by if nodes are aromatic. 

        Args:
            data (_type_): _description_

        Returns:
            torch_geometric.data.Data: A pyg representation of the graph. The nodes are ordered so first are aromatic and then non-aromatic. 
                                    Number of eaches is given by data.sizes.
        """

        # extract data to rdkid 
        molecule = graph.to_molecule()

        # number of aromatic nodes and non aromatic nodes 
        aromatic_nodes = [x.GetIdx() for x in molecule.GetAromaticAtoms()]

        # assign aromatic nodes to one community and the rest to another 
        not_aromatic_nodes = sorted(list(set(range(graph.num_node)) - set(aromatic_nodes)))
        node_map = {aromatic_nodes[i]: i for i in range(len(aromatic_nodes))}
        node_map = node_map | {node: i for i, node, in zip(range(len(aromatic_nodes), graph.num_node), not_aromatic_nodes)}

        # edge index with permuation
        edge_index = graph.edge_list[:, :2]
        edge_index = edge_index.apply_(node_map.get) # TODO: check this works... 
        edge_index = edge_index.T

        # node features with permutation
        x = graph.node_feature
        perm = torch.tensor(list(node_map.keys()))
        x = x[perm] # TODO: check this works

        # label and sizes
        y = torch.LongTensor((label,))
        sizes = (len(aromatic_nodes), len(not_aromatic_nodes))

        # pyg graph
        graph = Data(x=x, edge_index=edge_index, y=y, sizes=sizes)
        return graph 