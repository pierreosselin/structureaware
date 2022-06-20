from functools import lru_cache
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
from torch import LongTensor
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix


def split_data(y: LongTensor, train_size: float, test_size: Union[float, int]) -> dict:
    """Split the data into balanced train, validation, and test sets. The validation set size is inferred.

    Args:
        y (LongTensor): Labels of the data.
        train_size (float): Proportion of the dataset for the training data.
        test_size (Union[float, int], optional): Proportion or the absolute size of the dataset for test data.

    Returns:
        dict: A dictionary with keys 'train', 'valid' and 'test' containing the indices of the corresponding dataset.
    """
    idx = np.arange(len(y))
    train_idx, valid_test_idx = train_test_split(idx, train_size = train_size, stratify=y)
    if isinstance(test_size, float):
        test_size = test_size * len(y)/len(valid_test_idx)
    valid_idx, test_idx = train_test_split(valid_test_idx, test_size = test_size, stratify=y[valid_test_idx])
    split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return split

def assign_graph_ids(data_list):
    for i, graph in enumerate(data_list):
        graph.idx = i

@lru_cache(maxsize=None)
def positional_encoding(edge_index, k=6):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')
    vecs = sp.linalg.eigsh(to_scipy_sparse_matrix(edge_index, edge_weight), k=k, which='SM')[1]
    return torch.tensor(vecs)
