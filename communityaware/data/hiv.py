import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from os.path import join, isdir
import warnings
from torchdrug import datasets 
import torchdrug
from tqdm import tqdm 
from rdkit.Chem import AtomValenceException
from communityaware.data.utils import split_data, assign_graph_ids
from communityaware.cert.utils import triangle_number
from typing import Tuple, Callable
import numpy as np


class HIV(InMemoryDataset):

    def __init__(self, root: str, split_proportions: Tuple[float]=(0.8, 0.1, 0.1), min_required_edge_flips: int=0, transform: Callable=None, pre_transform: Callable=None, pre_filter: Callable=None):
        """HIV Dataset.

        Args:
            root (str): Root directory where the dataset should be saved.
            split_proportions (tuple, optional): Tuple specifying the size of the train, valid and test set. Defaults to (0.8, 0.1, 0.1).
            min_required_edge_flips (int, optional): If specified, the data will be filtered so there are at least `min_required_edge_flips` edges between aromatic nodes and between non-aromatic nodes. Defaults to 0.
            transform (Callable, optional): see InMemoryDataset. Defaults to None.
            pre_transform (Callable, optional): see InMemoryDataset. Defaults to None.
            pre_filter (Callable, optional): see InMemoryDataset. Defaults to None.
        """
        self.root = root
        self.split_proportions = split_proportions
        self.min_required_edge_flips = min_required_edge_flips
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])
        
    def download(self):
        """Download HIV dataset from torchdrug and save it in PyG format."""
        
        # download dataset using torchdrug 
        with warnings.catch_warnings(): # torch drug throws warnings about not recognising some atom types.
            warnings.simplefilter("ignore")
            dataset = datasets.HIV(self.raw_dir)
        
        # put into pyg format before saving
        data_list = []
        graphs_not_processed = 0
        for datapoint in tqdm(dataset, desc='Converting to PyG format.'):
            try:
                data_list.append(to_torch_geometric(datapoint['graph'], datapoint['HIV_active']))
            except AtomValenceException:
                graphs_not_processed += 1

        # save dataset
        print(f'Graphs not processed: {graphs_not_processed} of {len(dataset)}.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.raw_paths[0])

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
        fname = f'{self.min_required_edge_flips}_{self.split_proportions[0]}_{self.split_proportions[1]}_{self.split_proportions[2]}.pt'
        return [fname,]

    def process(self):
        """Process the dataset. This will filter graphs using `min_required_edge_flips`, assign ids to graphs, and split the dataset."""

        # load dataset into data_list
        self.data, self.slices = torch.load(self.raw_paths[0])
        data_list = [self.get(i) for i in range(self.__len__())]

        # filter graphs with too few potential edge flips
        data_list = [graph for graph in data_list if enough_potential_edge_flips(graph, self.min_required_edge_flips)]
    
        # split data into train, val, test and assign unique ids.
        assign_graph_ids(data_list)    
        self.data, self.slices = self.collate(data_list)
        self.split = split_data(self.data.y, *self.split_proportions)

        torch.save((self.data, self.slices, self.split), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size)

    def make_noise_matrix(self, graph, p_inner, p_outer):
        """_summary_

        Args:
            graph (_type_): _description_
            p_inner (_type_): Noise parameter for edges in aromatic rings. 
            p_outer (_type_): Noise parameter for edges not in aromatic rings. 

        Returns:
            _type_: _description_
        """
        noise = np.ones((graph.num_nodes, graph.num_nodes)) * p_outer
        mask = np.array(np.outer(graph.aromatic_nodes.numpy(), graph.aromatic_nodes.numpy()), dtype=bool)
        noise[mask] = p_inner
        return noise 


def enough_potential_edge_flips(graph, min_required_edge_flips):
    """Given the graph and min_required_edge_flips will return True if `min_required_edge_flips` can be flipped between aromatic nodes and non-aromatic nodes.

    Args:
        graph (torch_geometric.data.Data): Input graph.
        min_required_edge_flips (int): Filter threshold.

    Returns:
        bool: True if graph passes the threshold.
    """
    number_aromatic_nodes = graph.aromatic_nodes.sum().item()
    number_non_aromatic_nodes = graph.num_nodes - number_aromatic_nodes
    potential_edge_flips = triangle_number(min(number_aromatic_nodes, number_non_aromatic_nodes))
    return potential_edge_flips >= min_required_edge_flips


def to_torch_geometric(graph: torchdrug.data.molecule.Molecule, label: int):
    """Convert a data sample from a torchdrug dataset to a pytorch dataset with communities membership determined by if nodes are aromatic. 

    Args:
        data (torchdrug.data.molecule.Molecule): Molecule.

    Returns:
        torch_geometric.data.Data: A pyg representation of the graph. The nodes are ordered so first are aromatic and then non-aromatic. 
                                Number of eaches is given by data.sizes.
    """

    # aromatic nodes from rdkit 
    aromatic_nodes = torch.tensor([x.GetIdx() for x in graph.to_molecule().GetAromaticAtoms()])
    aromatic_nodes = torch.zeros(graph.num_node).scatter_(0, aromatic_nodes, 1)

    # data for pyg datastructure 
    edge_index = graph.edge_list[:, :2].T
    x = graph.node_feature
    y = torch.LongTensor((label,))

    # pyg graph
    graph = Data(x=x, edge_index=edge_index, y=y, aromatic_nodes=aromatic_nodes)
    return graph 
