import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import k_means
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from ..data.utils import positional_encoding


class GCN(torch.nn.Module):
    """Graph Convolutional Neural Network.

    Args:
        n_features (int): Dimension Node Feature
        hidden_channels ([int]): List of layers with corresponding dimension representation
        n_classes (int): Number of classes to predict
        p_dropout (float): Dropout parameter before last layer
        pooling: If True, use global pooling to readout the graph.
    """

    def __init__(self, num_features, hidden_channels, num_classes = 2, dropout=0.5, pooling=False, use_positional_encoding=False):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling
        self.use_positional_encoding = use_positional_encoding

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels,]

        final_dim = num_classes if num_classes > 2 else 1
        dimensions = [num_features, *hidden_channels, final_dim]

        self.convs = []
        for f_in, f_out in zip(dimensions[:-2], dimensions[1:-1]):
            self.convs.append(GCNConv(f_in, f_out))
        self.convs = nn.ModuleList(self.convs)
        self.linear = nn.Linear(dimensions[-2], dimensions[-1])
        self.dropout = dropout

    def forward(self, input_batch):
        # extract information from batch
        x, edge_index, batch = input_batch.x, input_batch.edge_index, input_batch.batch
        if self.use_positional_encoding:
            if input_batch.positional_encoding is None:
                x = torch.vstack([positional_encoding(graph.edge_index) for graph in input_batch.to_data_list()]).to(edge_index.device)
            else:
                x = input_batch.positional_encoding

        # 1. Obtain node embeddings
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()

        # 2. Readout layer if graph classification
        if self.pooling:
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return x
