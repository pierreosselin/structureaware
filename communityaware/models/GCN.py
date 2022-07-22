import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
from torch_geometric.utils import get_laplacian

from .degree_encoding import degree_encoding_batch
from .positional_encoding import position_encoding_batch


class GCN(torch.nn.Module):
    """Graph Convolutional Neural Network.

    Args:
        n_features (int): Dimension Node Feature
        hidden_channels ([int]): List of layers with corresponding dimension representation
        n_classes (int): Number of classes to predict
        p_dropout (float): Dropout parameter before last layer
        pooling: If True, use global pooling to readout the graph.
    """

    def __init__(self, num_features=None, hidden_channels=None, num_classes=2, dropout=0.5, pooling=True,
                    drop_original_features=False,
                    use_positional_encoding=False, use_degree_encoding=False,
                    conv_type='gcn'):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling
        self.drop_original_features = drop_original_features
        self.use_positional_encoding = use_positional_encoding
        self.use_degree_encoding = use_degree_encoding
        self.conv_type = conv_type
        if drop_original_features:
            self.num_features = 0
        if self.use_positional_encoding:
            self.num_features += 6
        if self.use_degree_encoding:
            self.num_features += 10

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels,]

        final_dim = num_classes if num_classes > 2 else 1
        dimensions = [self.num_features, *hidden_channels, final_dim]

        self.convs = []
        for f_in, f_out in zip(dimensions[:-2], dimensions[1:-1]):
            if self.conv_type == 'gcn':
                self.convs.append(GCNConv(f_in, f_out))
            elif self.conv_type == 'gin':
                self.convs.append(GINConv(Sequential(nn.Linear(f_in, f_out), nn.ReLU(), nn.Linear(f_out, f_out))))

        self.convs = nn.ModuleList(self.convs) #+ self.batch_norms)
        self.linear = nn.Linear(dimensions[-2], dimensions[-1])
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # extract information from batch
        if self.drop_original_features:
            x = None

        if self.use_positional_encoding and x is None:
            x = position_encoding_batch(edge_index, batch)
        elif self.use_positional_encoding:
            x = torch.cat((x, position_encoding_batch(edge_index, batch)), dim=1)

        if self.use_degree_encoding and x is None:
            x = degree_encoding_batch(edge_index, batch)
        elif self.use_degree_encoding:
            x = torch.cat((x, degree_encoding_batch(edge_index, batch)), dim=1)

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
