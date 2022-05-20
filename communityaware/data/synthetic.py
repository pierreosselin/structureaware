from os.path import join

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import (from_networkx, to_networkx,
                                   to_scipy_sparse_matrix)
from tqdm import tqdm

from .utils import assign_graph_ids, split_data


class Synthetic(InMemoryDataset):

    def __init__(self, root, graphs_per_class=1000, size_of_community=20, number_of_communities=3, split_proportions=(0.8, 0.1, 0.1), transform=None, pre_transform=None, pre_filter=None):
        """_summary_

        Args:
            root (_type_): _description_
            graphs_per_class (int, optional): _description_. Defaults to 300.
            sbm_community_sizes (tuple, optional): _description_. Defaults to (20, 20, 20).
            sbm_parameters (tuple, optional): (p_inner, p_outer). Defaults to (0.2, 0.02).
            er_parameter (float, optional): _description_. Defaults to 0.2.
            split_proportions (tuple, optional): _description_. Defaults to (0.8, 0.1, 0.1).
            transform (_type_, optional): _description_. Defaults to None.
            pre_transform (_type_, optional): _description_. Defaults to None.
            pre_filter (_type_, optional): _description_. Defaults to None.
        """
        self.root = root
        self.graphs_per_class = graphs_per_class
        self.size_of_community = size_of_community
        self.number_of_communities = number_of_communities
        self.split_proportions = split_proportions
        self.number_of_nodes = size_of_community * number_of_communities
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])
        
    def download(self):

        # generate graphs
        progress_bar = tqdm(desc='Generating synthetic data', total=2*self.graphs_per_class)
        data_list = []
        for i in range(self.graphs_per_class):
            data_list.append(SBM(self.number_of_nodes, self.number_of_communities))
            data_list.append(ER(self.number_of_nodes))
            progress_bar.update(2)

        # save
        torch.save(data_list, self.raw_paths[0])

    @property
    def raw_dir(self):
        return join(self.root, 'synthetic', 'raw')
    
    @property
    def raw_file_names(self):
        return ['graphs.pt', ]

    @property
    def processed_dir(self):
        return join(self.root, 'synthetic', 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt',]

    def process(self):
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        assign_graph_ids(data_list)
        self.data, self.slices = self.collate(data_list)
        self.split = split_data(self.data.y, *self.split_proportions)

        torch.save((self.data, self.slices, self.split), self.processed_paths[0])

    def dataloader(self, split, batch_size=32):
        return DataLoader([self[i] for i in self.split[split]], batch_size=batch_size)
    
    def make_noise_matrix(self, graph, p_inner, p_outer=None):
        """_summary_

        Args:
            graph (_type_): _description_
            p_inner (_type_): Noise parameter for edges within communites for SBM or for all edges for ER.
            p_outer (_type_):  Noise parameter for edges between communites for SBM.

        Returns:
            _type_: _description_
        """
        if graph.y.item() == 0: 
            noise = p_inner * np.ones((graph.num_nodes, graph.num_nodes))
        else:
            community_size = int(graph.num_nodes / graph.number_of_communities)
            noise = p_outer * np.ones((graph.num_nodes, graph.num_nodes))
            for i in range(graph.number_of_communities):
                noise[i * community_size: (i+1) * community_size, i * community_size: (i+1) * community_size] = p_inner
        return noise 

    @property
    def testset_labels(self):
        return torch.concat([self[i].y for i in self.split['test']])


def ER(number_of_nodes):
    graph = from_networkx(connected_critical_er_graph(number_of_nodes))
    graph.y = torch.LongTensor((0,))
    assign_synthetic_features(graph)
    graph.number_of_communities = 1
    return graph

def connected_critical_er_graph(number_of_nodes):
    p = np.log(number_of_nodes)/number_of_nodes
    graph = nx.erdos_renyi_graph(number_of_nodes, p)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(number_of_nodes, p)
    return graph

def SBM(number_of_nodes, number_of_communities):
    graph = from_networkx(connected_critical_sbm_graph(number_of_nodes, number_of_communities))
    del graph.block
    graph.y = torch.LongTensor((1,))
    assign_synthetic_features(graph)
    graph.number_of_communities = number_of_communities
    return graph

def connected_critical_sbm_graph(n, k, p_in_over_p_out = 10.0):
    """Graph with n nodes and k communities. n must be divisible by k. p_in_over_p_out is the ratio of p_in (within communities) and p_out (between communities)."""
    if n % k != 0:
        raise ValueError("n must be divisible by k.")

    # Calculate critical values of p_in, p_out which respect the p_in_over_p_out parameter.
    b = k / (p_in_over_p_out + k - 1) 
    a = k - (k-1) * b
    p_in = a * np.log(n)/n
    p_out = b * np.log(n)/n

    # sanity checks
    assert np.isclose(p_in, p_out * p_in_over_p_out)
    assert 0 <= p_in <= 1 
    assert 0 <= p_out <= 1 
    assert np.isclose(a + (k-1)*b, k)

    # construct graph
    p = np.ones((k, k)) * p_out
    np.fill_diagonal(p, p_in)
    sizes = np.ones(k, dtype=int) * int(n/k)
    g = nx.stochastic_block_model(sizes, p)
    while not nx.is_connected(g):
        g = nx.stochastic_block_model(sizes, p)
    return g

def assign_synthetic_features(graph, feature='PE'):
    if feature=='ones':
        graph.x = torch.ones(graph.num_nodes, 1)
    elif feature=='katz':
        graph_nx = to_networkx(graph)
        graph.x = torch.FloatTensor([nx.katz_centrality_numpy(graph_nx)[i] for i in range(graph_nx.number_of_nodes())]).unsqueeze(1)
    elif feature=='PE':
        _, vecs = sp.linalg.eigsh(to_scipy_sparse_matrix(graph.edge_index, num_nodes=graph.num_nodes))
        graph.x = torch.FloatTensor(vecs)
    else:
        raise ValueError('feature should be one of: "ones", "katz" or "PE".')
    assert graph.num_nodes == graph.x.shape[0]
