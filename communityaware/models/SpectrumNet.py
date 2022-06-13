from functools import lru_cache

import scipy.sparse as sp
import torch
from torch_geometric.utils import (get_laplacian, to_dense_adj,
                                   to_scipy_sparse_matrix)

from communityaware.models import MLP


class SpectrumNet(torch.nn.Module):

    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5, k=6):
        super(SpectrumNet, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.k = k
        self.output_dim = num_classes if num_classes > 2 else 1
        self.mlp = MLP(num_features, hidden_channels, self.output_dim, dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.spectrum(edge_index, batch, self.k)
        x = self.mlp(x)
        return x

    @staticmethod
    @torch.no_grad()
    def spectrum(edge_index, batch=None, k=6):
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')
        eigenvalues = torch.linalg.eigvalsh(to_dense_adj(edge_index, batch, edge_weight))
        x = eigenvalues[:, :k]
        return x
