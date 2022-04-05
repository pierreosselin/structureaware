import torch
import numpy as np
from torch_sparse import coalesce
from ..utils import copy_idx, offset_idx

def sample_perturbed_graphs_bernoulli(data_idx, n, batch_size, param_noise, device, **kwargs):
    """Sample pertubed graphs according to bernoulli topological noise

    Args:
        data_idx ([2, n_edges] array): Array of the graph edges
        n (int): Number of nodes in the graph
        batch_size (int): Batch size for the number of graph to sample
        param_noise (float): Bernoulli Parameter

    Returns:
        [2, m] array: Array of batch of perturbed graphs
    """
    # Select only direct edges
    data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    # Copy the graph
    idx_copies = copy_idx(data_idx, n, batch_size)
    w_existing = torch.ones_like(idx_copies[0])

    # Sample edges to delete
    to_del = torch.BoolTensor(idx_copies.shape[1]).bernoulli_(param_noise).to(device)
    w_existing[to_del] = 0

    #To rewrite and test!!!
    nadd_persample_np = np.random.binomial(n * n, param_noise, size=batch_size)  # 6x faster than PyTorch
    nadd_persample = torch.FloatTensor(nadd_persample_np).to(device)
    nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * n))
                                            / np.log(1 - 1 / (n * n))).long()
    nadd_with_repl = nadd_persample_with_repl.sum()
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(n * n)
    to_add[1] = to_add[0] % n
    to_add[0] = to_add[0] // n
    to_add = offset_idx(to_add, nadd_persample_with_repl, n, [0, 1])

    ## All graphs are undirected
    to_add = to_add[:, to_add[0] < to_add[1]]
    w_added = torch.ones_like(to_add[0])
    mb = batch_size * n

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               batch_size * n, mb, 'min')
    per_data_idx = joined[:, weights > 0]
    per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)
    
    return per_data_idx