import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def load_model(config):
    if config["optimisation"]["model"] == 'GCN_Classification':
        return GCN_Classification

class GCN_Classification(torch.nn.Module):
    """Graph Convolutional Implemetation fro graph classification

    Args:
        n_features (int): Dimension Node Feature
        hidden_channels ([int]): List of layers with corresponding dimension representation
        n_classes (int): Number of classes to predict
        p_dropout (float): Dropout parameter before last layer
    """

    def __init__(self, n_features, hidden_channels, n_classes = 2, p_dropout=0.5):
        super(GCN_Classification, self).__init__()
        self.convlist = []
        self.convlist.append(GCNConv(n_features, hidden_channels[0]))
        for i, el in enumerate(hidden_channels[1:]):
            self.convlist.append(GCNConv(hidden_channels[i], el))
        self.lin = nn.Linear(hidden_channels[-1], n_classes)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for layer in self.convlist:
            x = layer(x, edge_index)
            x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.lin(x)

        return x