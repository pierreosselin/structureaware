### FIle to run experiment on server directly
import argparse
import numpy as np
import torch
import json
import networkx as nx
import random
import os.path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
import sknetwork as skn

def generate_data(config):
    """Download and Process selected dataset

    Args:
        config (Dict): Configuration of the run

    Returns:
        None
    """
    if config["dataset"] == "synthetic":
        return generate_synthetic(config)

    if config["dataset"] == "mutag":
        return generate_mutag(config)

    if config["dataset"] == "reddit":
        return generate_reddit(config)
    
    print("Dataset not found")
    return -1
    
def generate_SBMS(config):
    """Function generating SBMS graphs

    Args:
        config (dict): configuration dictionary

    Returns:
        [torch_geometric.data.Data]: list of graphs
    """

    ## Generate SBMS
    n_data = config["n_data_per_class"]
    list_blocks = config["list_blocks"]
    p = config["block_probs"]
    n_graph = sum(list_blocks)

    l_data = []
    
    print("Generate SBMs graphs...")
    for i in tqdm(range(n_data)):
        G = nx.generators.community.stochastic_block_model(list_blocks, p)
        edges = G.edges
        adj = skn.utils.edgelist2adjacency(edges)
        betweenness = skn.ranking.Katz()
        scores = betweenness.fit_transform(adj)
        ### Compute features
        x = torch.zeros((n_graph, 1))
        #d1 = nx.algorithms.centrality.degree_centrality(G)
        #d2 = nx.algorithms.cluster.clustering(G)
        #d3 = nx.algorithms.centrality.closeness_centrality(G)
        #d4 = nx.algorithms.centrality.betweenness_centrality(G)

        x[:, 0] = torch.tensor(scores)
        #x[:, 0] = torch.tensor([d2[i] for i in range(n_graph)])
        #x[:, 0] = torch.tensor([d5[i] for i in range(n_graph)])
        #x[:, 1] = torch.tensor([d2[i] for i in range(n_graph)])
        #x[:, 2] = torch.tensor([d3[i] for i in range(n_graph)])
        #x[:, 3] = torch.tensor([d4[i] for i in range(n_graph)])
        #x[:, 4] = torch.tensor([d5[i] for i in range(n_graph)])

        edge_idx = torch.from_numpy(np.array(G.edges).T)
        edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)

        y = torch.zeros(1).long()

        data = Data(x=x, edge_index=edge_idx, y = y)

        data.node_community = torch.tensor(sum([[i for k in range(el)] for i, el in enumerate(list_blocks)], []))

        cumsum_list_blocks = [0] + list(np.cumsum(list_blocks))

        data.community_node = [list(range(cumsum_list_blocks[i], cumsum_list_blocks[i+1])) for i in range(len(list_blocks))]

        data.community_size = torch.tensor(list_blocks)

        data.community_prob = torch.tensor(p)

        l_data.append(data)

    return l_data


def generate_ER(config):
    """Function generating ER graphs

    Args:
        config (dict): configuration dictionary

    Returns:
        [torch_geometric.data.Data]: list of graphs
    """


    ## Generate ER
    n_data = config["n_data_per_class"]
    list_blocks = config["list_blocks"]
    p = config["block_probs"]
    n_graph = sum(list_blocks)

    ## If not specify compute coef such that the expected number of edges is the same
    if config["er_param"]:
        er_p = config["er_param"]
    else:
        exp_edges_sbm = 0
        n_list_blocks = len(list_blocks)
        for i in range(n_list_blocks):
            for j in range(i):
                exp_edges_sbm += p[i][j] * list_blocks[i] * list_blocks[j]
            exp_edges_sbm += p[i][i] * list_blocks[i] * (list_blocks[i] - 1) / 2
        er_p = 2 * exp_edges_sbm / (n_graph*(n_graph - 1))

    l_data = []

    print("Generate ER graphs...")
    ## Add condition for clustering or not, by default single cluster here
    for i in tqdm(range(n_data)):
        G = nx.generators.random_graphs.erdos_renyi_graph(n_graph, er_p)
        edges = G.edges
        adj = skn.utils.edgelist2adjacency(edges)
        betweenness = skn.ranking.Katz()
        scores = betweenness.fit_transform(adj)
        ###Compute Features
        x = torch.zeros((n_graph, 1))
        #d1 = nx.algorithms.centrality.degree_centrality(G)
        #d2 = nx.algorithms.cluster.clustering(G)
        #d3 = nx.algorithms.centrality.closeness_centrality(G)
        #d4 = nx.algorithms.centrality.betweenness_centrality(G)
        #d5 = nx.algorithms.centrality.katz_centrality(G)

        x[:, 0] = torch.tensor(scores)
        #x[:, 0] = torch.tensor([d2[i] for i in range(n_graph)])
        #x[:, 2] = torch.tensor([d3[i] for i in range(n_graph)])
        #x[:, 3] = torch.tensor([d4[i] for i in range(n_graph)])
        #x[:, 4] = torch.tensor([d5[i] for i in range(n_graph)])


        edge_idx = torch.from_numpy(np.array(G.edges).T)
        edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)

        y = torch.ones(1).long()

        data = Data(x=x, edge_index=edge_idx, y = y)

        data.node_community = torch.tensor([0 for i in range(n_graph)])

        data.community_node = [list(range(n_graph))]

        data.community_size = torch.tensor([n_graph])

        data.community_prob = torch.tensor([[er_p]])

        l_data.append(data)
    return l_data

def generate_synthetic(config):
    #Check if data already exists
    if os.path.isfile(config["save_path"] + "dataset_test"):
        print("Dataset already exists, delete it beforehand")
        return 0
       
    l_data_sbm = generate_SBMS(config)

    l_data_er = generate_ER(config)

    # Split train/test
    n_train = int(config["n_data_per_class"]*config["prop_train_test"])
    l_data_sbm_train, l_data_sbm_test = l_data_sbm[:n_train], l_data_sbm[n_train:]
    l_data_er_train, l_data_er_test = l_data_er[:n_train], l_data_er[n_train:]

    l_data_train = l_data_sbm_train + l_data_er_train
    l_data_test = l_data_sbm_test + l_data_er_test
    
    random.shuffle(l_data_train)
    random.shuffle(l_data_test)

    # Save data
    torch.save(l_data_train, config["save_path"] + "dataset_train")
    torch.save(l_data_test, config["save_path"] + "dataset_test")


def generate_mutag(config):
    #Check if data already exists
    if os.path.isfile(config["save_path"] + "dataset_test"):
        print("Dataset already exists, delete it beforehand, crushing it")
        #return 0

    dataset = TUDataset(root=config["save_path"], name='MUTAG')
    dataset = dataset.shuffle()
    n_data = len(dataset)

    if config["apply_clustering"]:
        print("Applying Clustering to the dataset")
        l_data = []
        param_cluster = config["clustering_parameter"]
        for datum in tqdm(dataset):

            ###Compute Features
            data = Data(x=datum.x, edge_index=datum.edge_index, y = datum.y)

            community_prob, node_community, community_node, community_size = process_clustering(datum, param_cluster, 1)

            data.node_community = torch.tensor(node_community)

            data.community_node = community_node

            data.community_size = torch.tensor(community_size)

            data.community_prob = torch.tensor(community_prob)

            l_data.append(data)

    # Split train/test
    n_train = int(n_data*config["prop_train_test"])
    l_data_train, l_data_test = l_data[:n_train], l_data[n_train:]

    # Save data
    torch.save(l_data_train, config["save_path"] + "dataset_train")
    torch.save(l_data_test, config["save_path"] + "dataset_test")

def generate_reddit(config):
    #Check if data already exists
    if os.path.isfile(config["save_path"] + "dataset_test"):
        print("Dataset already exists, delete it beforehand")

    dataset = TUDataset(root=config["save_path"], name='REDDIT-BINARY')
    dataset = dataset.shuffle()
    n_data = len(dataset)

    if config["apply_clustering"]:
        print("Applying Clustering to the dataset")
        l_data = []
        param_cluster = config["clustering_parameter"]
        for datum in tqdm(dataset):

            n_nodes = datum.edge_index.max() + 1
            x = torch.ones((n_nodes, 1))
            ###Compute Features
            data = Data(x=x, edge_index=datum.edge_index, y = datum.y)

            community_prob, node_community, community_node, community_size = process_clustering(data, param_cluster, 1)

            data.node_community = torch.tensor(node_community)

            data.community_node = community_node

            data.community_size = torch.tensor(community_size)

            data.community_prob = torch.tensor(community_prob)

            l_data.append(data)


    # Split train/test
    n_train = int(n_data*config["prop_train_test"])
    l_data_train, l_data_test = l_data[:n_train], l_data[n_train:]

    # Save data
    torch.save(l_data_train, config["save_path"] + "dataset_train")
    torch.save(l_data_test, config["save_path"] + "dataset_test")