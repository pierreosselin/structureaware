from multiprocessing.sharedctypes import Value
import os 
from os.path import join
from torch_geometric.utils import from_networkx, to_networkx, to_scipy_sparse_matrix
import scipy.sparse as sp
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


def generate_synthetic_data(graphs_per_class, sbm_community_sizes, sbm_parameters, er_nodes, er_parameter, data_path, data_split):

    # generate dataset
    progress_bar = tqdm(desc='Generating synthetic data', total=2*graphs_per_class)
    data_list = []
    for _ in range(graphs_per_class):
        data_list.append(SBM(sbm_community_sizes, sbm_parameters))
        data_list.append(ER(er_nodes, er_parameter))
        progress_bar.update(2)
    
    # split dataset
    y = np.array([data.y.item() for data in data_list])
    split = split_data(y, *data_split)

    # save
    os.makedirs(join(data_path, 'raw'), exist_ok=True)
    torch.save(data_list, join(data_path, 'raw', 'graphs.pt'))
    torch.save(split,  join(data_path, 'raw', 'split.pt'))


def ER(n, p):
    graph = from_networkx(nx.erdos_renyi_graph(n, p))
    graph.node_community = torch.zeros((graph.num_nodes))
    graph.y = torch.LongTensor((1,))
    assign_synthetic_features(graph)
    return graph

def SBM(sizes, p):
    graph = from_networkx(nx.stochastic_block_model(sizes, p))
    graph.node_community = graph.block
    del graph.block
    graph.y = torch.LongTensor((0,))
    assign_synthetic_features(graph)
    return graph

def assign_synthetic_features(graph, feature='katz'):
    if feature=='ones':
        graph.x = torch.ones(graph.num_nodes, 1)
    elif feature=='katz':
        graph_nx = to_networkx(graph)
        graph.x = torch.FloatTensor([nx.katz_centrality_numpy(graph_nx)[i] for i in range(graph_nx.number_of_nodes())]).unsqueeze(1)
    elif feature=='PE':
        _, vecs = sp.linalg.eigsh(to_scipy_sparse_matrix(graph.edge_index))
        graph.x = torch.FloatTensor(vecs)
    else:
        raise ValueError('feature should be one of: "ones", "katz" or "PE".')

def split_data(y, train_size, val_size, test_size):
    idx = np.arange(len(y))
    train_idx, valid_test_idx = train_test_split(idx, train_size = train_size, stratify=y)
    valid_idx, test_idx = train_test_split(valid_test_idx, train_size = val_size/(val_size+test_size), stratify=y[valid_test_idx])
    split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return split