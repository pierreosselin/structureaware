import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool


class GCN(torch.nn.Module):
    """Graph Convolutional Neural Network.

    Args:
        n_features (int): Dimension Node Feature
        hidden_channels ([int]): List of layers with corresponding dimension representation
        n_classes (int): Number of classes to predict
        p_dropout (float): Dropout parameter before last layer
        pooling: If True, use global pooling to readout the graph.
    """

    def __init__(self, num_features, hidden_channels, num_classes = 2, dropout=0.5, pooling=False):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.dropout = dropout
        self.pooling = pooling

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels,]
        dimensions = [num_features, *hidden_channels, num_classes]

        self.convs = []
        for f_in, f_out in zip(dimensions[:-2], dimensions[1:-1]):
            self.convs.append(GCNConv(f_in, f_out))
        self.convs = nn.ModuleList(self.convs)
        self.linear = nn.Linear(dimensions[-2], dimensions[-1])
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
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
