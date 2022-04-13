import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from os.path import join
from torch_geometric.utils import from_networkx, to_networkx, to_scipy_sparse_matrix
import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
from .utils import split_data, assign_graph_ids


class Synthetic(InMemoryDataset):

    def __init__(self, root, graphs_per_class=300, sbm_community_sizes=(20, 20, 20), sbm_parameters=(0.2, 0.02), er_parameter=0.2, split_proportions=(0.8, 0.1, 0.1), transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.graphs_per_class = graphs_per_class
        self.sbm_community_sizes = sbm_community_sizes
        self.sbm_parameters = sbm_parameters
        self.er_parameter = er_parameter
        self.split_proportions = split_proportions
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, self.split = torch.load(self.processed_paths[0])
        
    def download(self):
        # convert (p_inner, p_outer) to format that networkx sbm takes
        number_of_communities = len(self.sbm_community_sizes)
        nx_p = np.ones((number_of_communities, number_of_communities)) * self.sbm_parameters[0]
        np.fill_diagonal(nx_p, self.sbm_parameters[1])

        # generate graphs
        progress_bar = tqdm(desc='Generating synthetic data', total=2*self.graphs_per_class)
        data_list = []
        for i in range(self.graphs_per_class):
            data_list.append(SBM(self.sbm_community_sizes, nx_p))
            data_list.append(ER(sum(self.sbm_community_sizes), self.er_parameter))
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

def ER(n, p):
    graph = from_networkx(nx.erdos_renyi_graph(n, p))
    graph.y = torch.LongTensor((1,))
    assign_synthetic_features(graph)
    graph.sizes = (n,)
    return graph

def SBM(sizes, p):
    graph = from_networkx(nx.stochastic_block_model(sizes, p))
    graph.y = torch.LongTensor((0,))
    del graph.block
    assign_synthetic_features(graph)
    graph.sizes = tuple(sizes)
    return graph

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
