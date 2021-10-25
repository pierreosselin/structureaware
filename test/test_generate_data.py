
"""
Test for:
- get a dataset already processed
"""
from module.generate_data import generate_ER, generate_SBMS, generate_synthetic
import numpy as np
from torch_geometric.utils import is_undirected
import pytest
import shutil
from pathlib import Path


def test_er_generation():

        # Test undirected graph
        n_data, list_blocks, p, er_param = 10, [10,10], np.array([[0.8, 0.1], [0.1, 0.8]]), None
        data = generate_ER(n_data, list_blocks, p, er_param)
        for el in data:
                assert is_undirected(el.edge_index)

        # Test inconsistent input
        n_data, list_blocks, p, er_param = 10, [10,10], np.array([[1.1, 0.1], [0.1, 0.8]]), None
        with pytest.raises(Exception):
                data = generate_ER(n_data, list_blocks, p, er_param)
        
        n_data, list_blocks, p, er_param = 10, [10,10, 10], np.array([[0.8, 0.1], [0.1, 0.8]]), None
        with pytest.raises(Exception):
                data = generate_ER(n_data, list_blocks, p, er_param)
        
        n_data, list_blocks, p, er_param = 10, [10,10], np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.8]]), None
        with pytest.raises(Exception):
                data = generate_ER(n_data, list_blocks, p, er_param)
        
        n_data, list_blocks, p, er_param = 10, [10,10], np.array([[0.8, 0.1], [0.1, 0.8]]), 1.5
        with pytest.raises(Exception):
                data = generate_ER(n_data, list_blocks, p, er_param)

def test_sbm_generation():

        # Test undirected graph
        n_data, list_blocks, p = 10, [10,10], np.array([[0.8, 0.1], [0.1, 0.8]])
        data = generate_SBMS(n_data, list_blocks, p)
        for el in data:
                assert is_undirected(el.edge_index)

        # Test inconsistent input
        n_data, list_blocks, p = 10, [10,10], np.array([[1.1, 0.1], [0.1, 0.8]])
        with pytest.raises(Exception):
                data = generate_SBMS(n_data, list_blocks, p)
        
        n_data, list_blocks, p = 10, [10, 10, 10], np.array([[0.8, 0.1], [0.1, 0.8]])
        with pytest.raises(Exception):
                data = generate_SBMS(n_data, list_blocks, p)
        
        n_data, list_blocks, p = 10, [10, 10], np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.8]])
        with pytest.raises(Exception):
                data = generate_SBMS(n_data, list_blocks, p)