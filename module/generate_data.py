import os 
from os.path import join
from torch_geometric.utils import from_networkx
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def generate_synthetic_data(graphs_per_class, sbm_community_sizes, sbm_parameters, er_nodes, er_parameter, data_path, data_split):
    # make folders
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(join(data_path, 'raw'), exist_ok=True)

    # generate dataset
    data_list = []
    for _ in range(graphs_per_class):
        data_list.append(SBM(sbm_community_sizes, sbm_parameters))
    for _ in range(graphs_per_class):
        data_list.append(ER(er_nodes, er_parameter))
    
    # split dataset
    labels = np.array([data.label.item() for data in data_list])
    split = split_data(labels, *data_split)

    # save
    torch.save(data_list, join(data_path, 'raw', 'graphs.pt'))
    torch.save(split,  join(data_path, 'split.pt'))


def ER(n, p):
    graph = from_networkx(nx.erdos_renyi_graph(n, p))
    graph.node_community = torch.zeros((graph.num_nodes))
    graph.label = torch.LongTensor((1,))
    assign_synthetic_features(graph)
    return graph

def SBM(sizes, p):
    graph = from_networkx(nx.stochastic_block_model(sizes, p))
    graph.node_community = graph.block
    del graph.block
    graph.label = torch.LongTensor((0,))
    assign_synthetic_features(graph)
    return graph

def assign_synthetic_features(graph):
    graph.x = torch.ones((graph.num_nodes, 1))

def split_data(labels, train_size, val_size, test_size):
    idx = np.arange(len(labels))
    train_idx, valid_test_idx = train_test_split(idx, train_size = train_size, stratify=labels)
    valid_idx, test_idx = train_test_split(valid_test_idx, train_size = val_size/(val_size+test_size), stratify=labels[valid_test_idx])
    split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}