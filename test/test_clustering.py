from module.clustering import compute_probabilities
import numpy as np
import torch
import pytest
from torch_geometric.data import Data

"""
Should test for:
- Number of clusters when obvious
- Consistency output (In shapes)
- Good index correspondance (we use igraph)
- Good approximation of probabilities
- Test with both duplicate edges
"""

def test_compute_probabilities():

        edge_idx = torch.Tensor([[0,1,2,3,4], [1, 2, 3, 4, 5]])
        edge_idx = edge_idx.type(torch.LongTensor)
        edge_idx = torch.cat((edge_idx, edge_idx[[1, 0]]), 1)
        data = Data(edge_index=edge_idx)

        n_communities = 2
        node_community = [0,0,0,1,1,1]
        community_size = [3,3]

        community_prob = compute_probabilities(n_communities, data, node_community, community_size)
        assert (community_prob == np.array([[2/3, 1/9], [1/9, 2/3]])).all()

