from typing import List, Union
import torch
import numpy as np
import networkx as nx
import igraph

def process_clustering(gr, param, digits):
    """Process a graph to partition in clusters, merge pairwise similar communities

    Args:
        gr (torch_geometric.data.Data): Graph to cluster
        param (float): Parameter for the clustering method
        digits ([type]): Degree of approximation for the blocks probabilities

    Returns:
        community_prob (2D Array): Array of inter and intra community probabilities
        node_community (1D Array): Array of node assignment to commumities
        community_node ([1D Array]): List of nodes assigned to each community
        community_size (1D Array): Size of each community
    """

    # Detect if undirected or not
    edge = gr.edge_index.T[0]
    directed = torch.tensor([edge[1], edge[0]]) in gr.edge_index.T

    g = igraph.Graph()
    n = gr.x.shape[0]
    g.add_vertices(n)
    g.add_edges(np.array(gr.edge_index.T))
    clusters = g.community_leiden(objective_function="modularity", resolution_parameter=param, n_iterations=-1)

    node_community = clusters.membership
    communities, community_size = np.unique(node_community, return_counts=True)
    n_communities = communities.shape[0]
    community_node = [[] for i in range(n_communities)]

    ## Compute community_node
    for i, el in enumerate(node_community):
        community_node[el].append(i)

    ###### Community probabilities
    ## Real Probabilities : replace edge with corresponding communities
    community_prob = np.zeros((n_communities, n_communities))
    list_edges = gr.edge_index.T
    for edge in list_edges:
        c1, c2 = node_community[edge[0]], node_community[edge[1]]
        if c1 != c2:
            community_prob[c1, c2] += 1
            community_prob[c2, c1] += 1
        else:
            community_prob[c1, c2] += 1

    if not directed:
        community_prob *= 2

    for i in range(n_communities):
        for j in range(i):
            community_prob[i, j] /= (community_size[i] * community_size[j])
            if community_prob[i, j] < 10**(-digits):
                community_prob[i, j] = 10**(-digits)
            else:
                community_prob[i, j] = round(community_prob[i, j], digits)

            community_prob[j, i] = community_prob[i, j]

        community_prob[i, i] /= community_size[i]*(community_size[i] - 1)
        if community_prob[i, i] < 10**(-digits):
            community_prob[i, i] = 10**(-digits)
        else:
            community_prob[i, i] = round(community_prob[i, i], digits)

    return community_prob, node_community, community_node, community_size
