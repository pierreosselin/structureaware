import time

import numpy as np
import torch
from timebudget import timebudget
from torch_geometric.data import Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_dense_adj

from communityaware.models.positional_encoding import position_encoding_batch

device = 'cpu'

def test_position_encoding_batch():
    _test_position_encoding_batch('MUTAG')
    _test_position_encoding_batch('ENZYMES')
    _test_position_encoding_batch('PROTEINS')

def _test_position_encoding_batch(dataset_name):
    print('Testing', dataset_name)
    dataset = TUDataset(root=f'data/{dataset_name}', name=dataset_name, pre_filter = lambda x: x.num_nodes > 6)

    # filter for simple spectrums so its easier to check decomposition equavalence...
    batch = []
    for i, data in enumerate(dataset):
        edge_index, edge_weight = get_laplacian(data.edge_index, normalization='sym')
        laplacian = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze()
        spectrum = torch.linalg.eigvalsh(laplacian)
        spectrum = torch.sort(spectrum)[0]
        if not torch.isclose(torch.diff(spectrum), torch.tensor(0.), atol=10e-4).any().item():
            batch.append(data)
    batch = Batch.from_data_list(batch).to(device)
    print(f'Using {batch.num_graphs} of {len(dataset)}')

    # batch computation
    batch_time = []
    for _ in range(10):
        start = time.time()
        x = position_encoding_batch(batch.edge_index, batch.batch)
        end = time.time()
        batch_time.append(end-start)

    # single graph computation
    single_time = []
    for _ in range(10):
        start = time.time()
        with torch.no_grad():
            xs = []
            for graph in batch.to_data_list():
                edge_index, edge_weight = get_laplacian(graph.edge_index, normalization='sym')
                laplacian = to_dense_adj(edge_index, edge_attr=edge_weight)
                _, eigvecs = torch.linalg.eigh(laplacian.squeeze())
                xs.append(eigvecs[:, :6])
            xs = torch.vstack(xs)
        end = time.time()
        single_time.append(end-start)

    # check if each eigenvector is the same
    ns = np.cumsum([x.num_nodes for x in batch.to_data_list()])
    for i in range(len(ns)):
        start = 0 if i == 0 else ns[i-1]
        end = ns[i]
        x1 = x[start:end]
        x2 = xs[start:end]

        for i in range(x1.size(1)):
            pivot = x1[:, i].abs().argmax().item() # if first entry is zero the next steps will be unstable, so we choose a pivot instead
            if x1[pivot, i] < 0:
                x1[:, i] *= -1
            if x2[pivot, i] < 0:
                x2[:, i] *= -1

        assert torch.allclose(x1, x2, atol=10e-4)
        assert torch.allclose(torch.nn.CosineSimilarity(dim=0)(x1, x2), torch.tensor(1.))

    print('Test passed.')
    print(f'Batch time = {np.mean(batch_time):.4f}. Non-batch time = {np.mean(single_time):.4f}. Speedup = {np.mean(single_time)/np.mean(batch_time):.4f}x')
