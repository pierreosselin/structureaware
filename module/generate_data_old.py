### FIle to run experiment on server directly
import numpy as np
import torch
import networkx as nx
import random
import os.path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from .utils import compute_p_from_sbm

def generate_data(name_dataset, n_data=None, list_blocks=None, p=None, er_param=None, save_path=None, prop_train_test=None):
    """Generate desired dataset and split it in train/test dataset

    Args:
        name_dataset (str): Name dataset to download
        n_data (int, optional): For synthetic dataset, number of element per class. Defaults to None.
        list_blocks ([int], optional): For synthetic dataset, number of element per sbm cluster. Defaults to None.
        p (2d Array float, optional): For synthetic dataset, array of cluster probability for SBM. Defaults to None.
        er_param (int, optional): For synthetic dataset, if specified, p parameter for ER graph generation. Defaults to None.
        save_path (str, optional): Path to save data. Defaults to None.
        prop_train_test (float, optional): Proportion of train element. Defaults to None.

    Returns:
        (list, list): Couple train/test set
    """
    if name_dataset == "synthetic":
        return generate_synthetic(n_data, list_blocks, p, er_param, save_path, prop_train_test)

    if name_dataset == "mutag":
        return generate_mutag(save_path, prop_train_test)

    if name_dataset == "reddit":
        return generate_reddit(save_path, prop_train_test)
    
    raise Exception("Incorrect data set name")
    
def generate_SBMS(n_data, list_blocks, p):
    """Function generating SBMS graphs

    Args:
        n_data (int): Number of elements per class
        list_blocks ([]): List number of elements per clusters
        p (2d array): Matrix of probability for block SBM

    Returns:
        [torch_geometric.data.Data]: Dataset SBMs
    """

    n_graph = sum(list_blocks)
    l_data = []
    
    for _ in tqdm(range(n_data), desc = "Generate SBMs graphs..."):
        G = nx.generators.community.stochastic_block_model(list_blocks, p)
        
        ### Compute features
        x = torch.FloatTensor([nx.katz_centrality_numpy(G)[i] for i in range(G.number_of_nodes())]).unsqueeze(1)
        edge_idx = torch.from_numpy(np.array(G.edges).T)
        edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)
        y = torch.zeros(1).long()

        ### Compute Data
        data = Data(x=x, edge_index=edge_idx, y = y)

        ### Compute clustering
        data.node_community = torch.tensor(sum([[i for _ in range(el)] for i, el in enumerate(list_blocks)], []))
        cumsum_list_blocks = [0] + list(np.cumsum(list_blocks))
        data.node_community = [list(range(cumsum_list_blocks[i], cumsum_list_blocks[i+1])) for i in range(len(list_blocks))]
        data.community_size = torch.tensor(list_blocks)
        data.community_prob = torch.tensor(p)

        l_data.append(data)
    return l_data


def generate_ER(n_data, list_blocks, p, er_param=None):
    """Function generating ER graphs

    Args:
        n_data (int): Number of elements per class
        list_blocks ([int]): List number of elements per clusters for the SBM
        p (2d array): Matrix of probability for block SBM
        er_param (float) : If specified probability parameter ER graphs

    Returns:
        [torch_geometric.data.Data]: Dataset ER
    """
    
    ## Generate ER
    n_graph = sum(list_blocks)

    ## If not specify compute coef such that the expected number of edges is the same
    if er_param is not None:
        er_p = er_param
        if (er_p > 1.) or (er_p < 0.):
            raise Exception("ER parameter is not a probability")
    else:
        er_p = compute_p_from_sbm(np.array(p), list_blocks)
    
    l_data = []

    ## Add condition for clustering or not, by default single cluster here
    for _ in tqdm(range(n_data), desc = "Generate ER graphs..."):
        G = nx.generators.random_graphs.erdos_renyi_graph(n_graph, er_p)

        ###Compute Features
        x = torch.FloatTensor([nx.katz_centrality_numpy(G)[i] for i in range(G.number_of_nodes())]).unsqueeze(1)
        edge_idx = torch.from_numpy(np.array(G.edges).T)
        edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)
        y = torch.ones(1).long()

        ### Compute Data
        data = Data(x=x, edge_index=edge_idx, y = y)
        
        ### Compute clusters
        data.node_community = torch.tensor([0 for _ in range(n_graph)])
        data.node_community = [list(range(n_graph))]
        data.community_size = torch.tensor([n_graph])
        data.community_prob = torch.tensor([[er_p]])
        l_data.append(data)
    return l_data

def generate_synthetic(n_data, list_blocks, p, er_param, save_path, prop_train_test):
    """Generate Synthetic dataset

    Args:
        n_data (int): Number of elements per class
        list_blocks ([int]): List number of elements per clusters for the SBM
        p (2d array): Matrix of probability for block SBM
        er_param (float) : If specified probability parameter ER graphs
        save_path (str): Path to save dataset
        prop_train_test (float): Proportion train/test set

    Returns:
        (list, list): Couple train/test set
    """

    #Check if data already exists
    if os.path.isfile(save_path + "dataset_test"):
        raise Exception("Dataset already exists, delete it beforehand")
       
    l_data_sbm = generate_SBMS(n_data, list_blocks, p)
    l_data_er = generate_ER(n_data, list_blocks, p, er_param)

    # Split train/test
    n_train = int(n_data*prop_train_test)
    l_data_sbm_train, l_data_sbm_test = l_data_sbm[:n_train], l_data_sbm[n_train:]
    l_data_er_train, l_data_er_test = l_data_er[:n_train], l_data_er[n_train:]
    l_data_train = l_data_sbm_train + l_data_er_train
    l_data_test = l_data_sbm_test + l_data_er_test
    
    #Shuffle
    random.shuffle(l_data_train)
    random.shuffle(l_data_test)

    return l_data_train, l_data_test
    
def generate_mutag(save_path, prop_train_test):
    """Generate Mutag dataset

    Args:
        save_path (str): Path to save dataset
        prop_train_test (float): Proportion train/test set

    Returns:
        (list, list): Couple train/test set
    """

    #Check if data already exists
    if os.path.isfile(save_path + "dataset_test"):
        print("Dataset already exists, delete it beforehand")
        return 0

    dataset = TUDataset(root=save_path, name='MUTAG')
    dataset = dataset.shuffle()
    n_data = len(dataset)

    l_data = []
    for datum in tqdm(dataset):
        n_nodes = datum.edge_index.max() + 1
        x = torch.ones((n_nodes, 1))
        ###Compute Features
        data = Data(x=x, edge_index=datum.edge_index, y = datum.y)
        l_data.append(data)

    # Split train/test
    n_train = int(n_data*prop_train_test)
    l_data_train, l_data_test = l_data[:n_train], l_data[n_train:]

    return l_data_train, l_data_test
    

def generate_reddit(save_path, prop_train_test):
    """Generate Reddit dataset

    Args:
        save_path (str): Path to save dataset
        prop_train_test (float): Proportion train/test set

    Returns:
        (list, list): Couple train/test set
    """
    #Check if data already exists
    if os.path.isfile(save_path + "dataset_test"):
        print("Dataset already exists, delete it beforehand")

    dataset = TUDataset(root=save_path, name='REDDIT-BINARY')
    dataset = dataset.shuffle()
    n_data = len(dataset)
    l_data = []
    for datum in tqdm(dataset):
        n_nodes = datum.edge_index.max() + 1
        x = torch.ones((n_nodes, 1))
        ###Compute Features
        data = Data(x=x, edge_index=datum.edge_index, y = datum.y)
        l_data.append(data)


    # Split train/test
    n_train = int(n_data*prop_train_test)
    l_data_train, l_data_test = l_data[:n_train], l_data[n_train:]

    return l_data_train, l_data_test