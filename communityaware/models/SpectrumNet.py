from functools import lru_cache

import scipy.sparse as sp
import torch
from torch_geometric.utils import (get_laplacian, to_dense_adj,
                                   to_scipy_sparse_matrix)

from communityaware.models import MLP


class SpectrumNet(torch.nn.Module):

    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(SpectrumNet, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.output_dim = num_classes if num_classes > 2 else 1
        self.mlp = MLP(num_features, hidden_channels, self.output_dim, dropout=dropout)

    def forward(self, batch):
        edge_index, edge_weight = get_laplacian(batch.edge_index, normalization='sym')
        eigenvalues = torch.linalg.eigvalsh(to_dense_adj(edge_index, batch.batch, edge_weight))
        x = eigenvalues[:, :6]
        x = self.mlp(x)
        return x

@lru_cache(maxsize=None)
@torch.no_grad()
def spectrum(edge_index, k=6):
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')
    eigvals = sp.linalg.eigsh(to_scipy_sparse_matrix(edge_index, edge_weight), k=k, which='SM')[0]
    return torch.tensor(eigvals)
