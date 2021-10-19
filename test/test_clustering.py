from module.clustering import compute_probabilities, approximate_probabilities
import numpy as np
import torch
import pytest
from torch_geometric.data import Data

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

def test_approximate_probabilities():
        community_prob = np.array([[0.2334, 0.4534], [0.4534, 0.2334]])
        n_communities = 2
        digits = 2
        result = approximate_probabilities(community_prob, n_communities, digits)
        assert (result == np.array([[0.23, 0.45], [0.45, 0.23]])).all()
        
        community_prob = np.array([[0.2334, 0.4534], [0.4534, 0.2334]])
        n_communities = 2
        digits = 4
        result = approximate_probabilities(community_prob, n_communities, digits)
        assert (result == community_prob).all()

        community_prob = np.array([[0.2334, 0.4534], [0.4534, 0.2334]])
        n_communities = 4
        digits = 4
        with pytest.raises(Exception):
                result = approximate_probabilities(community_prob, n_communities, digits)
        
        community_prob = np.array([[0.2334, 0.4534], [0.23, 0.2334]])
        n_communities, digits = 2, 2
        with pytest.raises(Exception):
                result = approximate_probabilities(community_prob, n_communities, digits)

def test_number_clusters():
        assert True

def test_node_indexing():
        assert True

def test_input_inconsistency():
        assert True

def test_output_consistency():
        assert True