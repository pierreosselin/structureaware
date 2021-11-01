from module.prediction import sample_perturbed_graphs_community
import numpy as np
import torch
import networkx as nx
import pytest


if __name__ == "__main__":
## Test edge probabilities under community
        list_blocks = [5,5,5]
        block_probs = [[0.2, 0.05, 0.05], [0.05, 0.3, 0.05], [0.05, 0.05, 0.4]]

        G = nx.generators.community.stochastic_block_model(list_blocks, block_probs)
        data_idx = torch.from_numpy(np.array(G.edges).T).cuda().long()
        initial_edges = [el for el in zip(data_idx[0], data_idx[1])]

        node_community = torch.tensor(sum([[i for _ in range(el)] for i, el in enumerate(list_blocks)], []))
        cumsum_list_blocks = [0] + list(np.cumsum(list_blocks))
        community_node = torch.tensor([list(range(cumsum_list_blocks[i], cumsum_list_blocks[i+1])) for i in range(len(list_blocks))])
        community_size = torch.tensor(list_blocks)
        community_prob = torch.tensor(block_probs)

        n, batch_size, param_noise = 15, 2000, 2.
        count_matrix = np.zeros((15,15))
        perturbed_graph = sample_perturbed_graphs_community(data_idx, n, batch_size, param_noise, community_node, community_size, community_prob)
        for el1, el2 in zip(perturbed_graph[0], perturbed_graph[1]):
                count_matrix[el1 % 15, el2 % 15] += 1
        count_matrix /= batch_size

        ## Check statistics of every edges
        for el1, el2 in initial_edges:
                assert abs(count_matrix[el1, el2] - (1 - param_noise*community_prob[node_community[el1],node_community[el2]])) < 0.1
        for i in range(15):
                for j in range(i):
                        if (j, i) not in initial_edges:
                                assert abs(count_matrix[j,i] - param_noise*community_prob[node_community[j],node_community[i]]) < 0.1