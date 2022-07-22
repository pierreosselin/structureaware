import torch
import torch_geometric
from torch import Tensor


def degree_encoding_batch(edge_index: Tensor, batch: Tensor, max_degree: int=10) -> torch.Tensor:
    """Encodes degree into one hot encoding.

    The one hot encoding is for node degrees between 1 and max_degree inclusive. Otherwise node vector is zero.

    Args:
        edge_index (Tensor): edge_index
        batch (Tensor): batch
        max_degree (int, optional): dimension of the node feature. Defaults to 10.

    Returns:
        torch.Tensor: (n x max_degree) one hot encoding of node degrees.
    """
    degree = torch_geometric.utils.degree(edge_index[1], len(batch))
    degree = torch.min(degree, torch.tensor(max_degree)).long()
    degree = torch.max(degree, torch.tensor(1)).long()
    degree = torch.nn.functional.one_hot(degree-1, max_degree).float()
    return degree
