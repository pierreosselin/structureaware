import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN_Classification(torch.nn.Module):
    """Graph Convolutional Implemetation fro graph classification

    Args:
        n_features (int): Dimension Node Feature
        hidden_channels ([int]): List of layers with corresponding dimension representation
        n_classes (int): Number of classes to predict
        p_dropout (float): Dropout parameter before last layer
    """

    def __init__(self, num_features, hidden_channels, num_classes = 2, dropout=0.5):
        super(GCN_Classification, self).__init__()
        
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
        if batch is not None:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return x
