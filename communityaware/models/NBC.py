from functools import partial
from itertools import accumulate, product

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric

from .MLP import MLP


class NBCModel(torch.nn.Module):

    def __init__(self, num_features, hidden_channels, num_classes, dropout=0.5):
        super(NBCModel, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.output_dim = num_classes if num_classes > 2 else 1
        self.mlp = MLP(num_features, hidden_channels, self.output_dim, dropout=dropout)

    def forward(self, batch):
        x = non_backtracking_path_count(batch, self.num_features)
        x = self.mlp(x)
        return x

def non_backtracking_path_count(batch, max_walk_length):
    if not isinstance(batch, torch_geometric.data.Batch):
        raise ValueError('Only implemented for input of type torch_geometric.data.Batch.')
    B = non_backtracking_matrix(batch.edge_index)
    B_powers = accumulate([B for _ in range(max_walk_length)], lambda x, y: x @ y) # computes B, B^2, B^3 ... B^max_walk_length (this retains block diaganol batch structure)
    B_powers_diag = np.stack([B.diagonal() for B in B_powers]) # row i is the diagonal of B^i
    split = np.cumsum([batch.get_example(i).num_edges for i in range(batch.num_graphs)]) # work out where to split the rows so they correspond to one graph in the batch
    signatures = map(partial(np.sum, axis=1), np.split(B_powers_diag, split, axis=1)) # split the B_powers_matrix then take the trace
    signatures = torch.tensor(np.stack(list(signatures)), dtype=torch.float) # put into torch format
    return signatures

def non_backtracking_path_count_no_batching(edge_index, max_walk_length):
    # Lemma 1 of https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0147-y.pdf
    B = non_backtracking_matrix(edge_index)
    B_powers = accumulate([B for _ in range(max_walk_length)], lambda x, y: x @ y) # computes B, B^2, B^3 ... B^max_walk_length
    signature = torch.tensor(list(map(sp.coo_matrix.trace, B_powers)))
    return signature

def non_backtracking_matrix(edge_index):
    # defn 1 of https://appliednetsci.springeropen.com/track/pdf/10.1007/s41109-019-0147-y.pdf
    edge_mapping = {tuple(edge.numpy()): i for i, edge in enumerate(edge_index.T)}
    edge_mapping = dict(sorted(edge_mapping.items(), key=lambda edge: max(edge[0]))) # sort by edges so B is block diaganol if graph is disconnected.
    entries = np.array([(i, j) for (i, (k, l)), (j, (u, v)) in product(enumerate(edge_mapping), enumerate(edge_mapping)) if u != l and v == k])
    values = np.ones((entries.shape[0],))
    B = sp.coo_matrix((values,entries.T), shape=(len(edge_mapping), len(edge_mapping)))
    return B
