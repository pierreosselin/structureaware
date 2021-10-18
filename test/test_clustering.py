from module.clustering import process_clustering
import numpy as np
import pytest

"""
Should test for:
- Number of clusters when obvious
- Consistency output (In shapes)
- Good index correspondance (we use igraph)
- Good approximation of probabilities
- Test with both duplicate edges
"""


config = {"n_data_per_class": 300, 
        "list_blocks": [20,20,20],
        "block_probs": [[0.2, 0.02, 0.02], [0.02, 0.3, 0.02], [0.02, 0.02, 0.4]],
        "er_param": None,
        "prop_train_test": 0.7}

def test_sbm_generation():
    assert 1