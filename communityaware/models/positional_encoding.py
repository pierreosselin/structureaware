from lib2to3.pgen2.token import OP
from typing import Optional, Tuple

import torch
from functorch import vmap
from timebudget import timebudget
from torch import Tensor
from torch_geometric.utils import get_laplacian, to_dense_adj


@torch.no_grad()
def position_encoding_batch(edge_index: Tensor, batch: Optional[Tensor] = None, k: int = 6) -> torch.tensor:
    """Computes the position encoding for a batch of graphs.
    Encodes are given by the first k eigenvectors of the symmetric normalised Laplacian matrix.

    Args:
        batch (LongTensor, optional): The batch vector hich assigns each
            value in `edge_index` to a specific example. Must be ordered. Defaults to None.
        k (int, optional): Number of eigenvectors. Defaults to 6.

    Raises:
        ValueError: Raised if smallest graph has less nodes than the number of eigenvectors specified by k.

    Returns:
        Tensor: (n x k) tensor of stacked position encodings for each graph in the batch.
    """
    if batch is None:
        batch = torch.zeros(edge_index.flatten().max()+1, dtype=torch.long)
    smallest_n = torch.bincount(batch).min().item()
    if smallest_n < k:
        raise ValueError(f'Graph sizes are too small for the given k: smallest graph is {smallest_n} but {k=}')
    edge_index, edge_weight = get_laplacian(edge_index, normalization='sym')
    laplacians = to_dense_adj(edge_index, batch, edge_weight)
    laplacians = add_laplacian_jitter(laplacians)
    eigenvectors, masks = eigenvectors_batch(laplacians)
    number_of_nodes = masks.sum(axis=(1))[: , -1] # number of nodes in each graph
    output = torch.vstack([eigvecs[mask].reshape(n, n)[:, :k] for eigvecs, mask, n in zip(eigenvectors, masks, number_of_nodes)])
    return output

@vmap
def add_laplacian_jitter(laplacian: Tensor) -> Tensor:
    """Adds a small amount of jitter to the laplacian matrix.

    This causes the eigenvalues to be shifted by a small value. This makes them distinguiable from zero eigenvalues coming
    from the zero padding.

    Args:
        laplacian (Tensor): A symmetric normalised Laplacian. May be zero padded on the right and bottom.
            e.g. the graph is of size n and laplacian[:n, :n] is the Laplacian. The rest of the tensor is zero.

    Returns:
        Tensor: _description_
    """
    mask = laplacian.diag() == 1
    laplacian = laplacian + torch.diag(mask) * 1e-6
    return laplacian

@vmap
def eigenvectors_batch(laplacian: Tensor) -> Tuple[Tensor, Tensor]:
    """Computes eigenvectors of the input symmetric normalised Laplacian.

    Args:
        laplacian (Tensor): A symmetric normalised Laplacian. May be zero padded on the right and bottom.
            e.g. the graph is of size n and laplacian[:n, :n] is the Laplacian. The rest of the tensor is zero.

    Returns:
        Tuple[Tensor, Tensor]: The padded eigenvector matrix and a boolean mask matrix.
            The eigenvector matrix of the Laplacian is given by eigenvectors[mask].reshape(n, n).
    """
    _, eigenvectors = torch.linalg.eigh(laplacian)
    mask = (laplacian.diag()).unsqueeze(0)
    mask = (mask.T @ mask)
    mask = mask.flip(1)
    mask = mask.bool()
    return (eigenvectors, mask)

def eigenvectors_batch_novap(laplacians: Tensor) -> Tuple[Tensor, Tensor]:
    """An alternative to eigenvectors_batch for if vmap gives issues..."""
    _, eigenvectors = torch.linalg.eigh(laplacians)
    masks = torch.diagonal(laplacians, dim1=1, dim2=2) # batch diag
    masks = torch.bmm(masks.unsqueeze(2), masks.unsqueeze(1)) # batch outerproduct
    masks = masks.flip(2)
    masks = masks.bool()
    return (eigenvectors, masks)
