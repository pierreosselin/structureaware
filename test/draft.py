from module.prediction import sample_perturbed_graphs_bernoulli, sample_perturbed_graphs_community
import numpy as np
import torch
import networkx as nx
import pytest


if __name__ == "main":
    ## Test edge probabilities und
    list_blocks = [5,5,5]
    block_probs = [[0.2, 0.02, 0.02], [0.02, 0.3, 0.02], [0.02, 0.02, 0.4]]
    
    G = nx.generators.community.stochastic_block_model(list_blocks, block_probs)
    data_idx = torch.from_numpy(np.array(G.edges).T).long()
    initial_edges = [el for el in zip(data_idx[0], data_idx[1])]

    node_community = torch.tensor(sum([[i for _ in range(el)] for i, el in enumerate(list_blocks)], []))
    cumsum_list_blocks = [0] + list(np.cumsum(list_blocks))
    community_node = [list(range(cumsum_list_blocks[i], cumsum_list_blocks[i+1])) for i in range(len(list_blocks))]
    community_size = torch.tensor(list_blocks)
    community_prob = torch.tensor(block_probs)
    

    n, batch_size, param_noise = 15, 5000, 0.3
    count_matrix = np.zeros((15,15))
    perturbed_graph = sample_perturbed_graphs_community(data_idx, n, batch_size, param_noise, community_node, community_size, community_prob)
    for el1, el2 in zip(perturbed_graph[0], perturbed_graph[1]):
            count_matrix[el1 % 15, el2 % 15] += 1
    count_matrix /= batch_size