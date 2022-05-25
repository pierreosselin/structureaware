import copy
import functools
from functools import partial

import functorch
import functorch.compile
import torch
import torch_geometric
from scipy.linalg import block_diag
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj


def perturb_batch(batch: torch_geometric.data.batch.Batch, p: tuple, make_noise_matrix) -> torch_geometric.data.batch.Batch:
    """Used to perturb a batch of graphs (each graph in the batch is perturbed once).

    Args:
        batch (torch_geometric.data.batch.Batch): The batch of graphs
        p (tuple): A noise description that fits the make_noise_matrix function
        make_noise_matrix (_type_): A function that operates on (graph, *p) to create a noise matrix.

    Returns:
        A batch where each graph has been perturbed.
    """
    noise_matrices = [make_noise_matrix(graph, *p) for graph in batch.to_data_list()]
    noise_matrix = torch.tensor(block_diag(*noise_matrices))
    adjacency = _get_adjacency(batch)
    adjacency_perturbed = _perturb_graph(adjacency, noise_matrix)
    edge_index = dense_to_sparse(adjacency_perturbed)[0]
    batch.edge_index = edge_index
    return batch

def perturb_graph(graph: torch_geometric.data.data.Data, p: torch.tensor, repeats: int, batch_size=32, device='cpu'):
    """Takes a single graph and makes a batchloader of perturbed graphs."""
    adjacency = _get_adjacency(graph).to(device)
    adjacency = adjacency.unsqueeze(0).repeat(repeats, 1, 1)
    p = p.unsqueeze(0).repeat(repeats, 1, 1).to(device)
    perturbed_adjacencies = _perturb_graph_vmap(adjacency, p)
    graphs = [copy.copy(graph) for _ in range(repeats)]
    for i, graph in enumerate(graphs):
        graph.edge_index = dense_to_sparse(perturbed_adjacencies[i])[0]
    return DataLoader(graphs, batch_size=batch_size)

@partial(functorch.vmap, randomness='different')
def _perturb_graph_vmap(adjacency: torch.tensor, p: torch.tensor) -> torch.tensor:
    return _perturb_graph(adjacency, p)

def _perturb_graph(adjacency: torch.tensor, p: torch.tensor) -> torch.tensor:
    noise = torch.bernoulli(p)
    noise = torch.triu(noise, diagonal=1)
    noise = noise + noise.transpose(0, 1)
    adjacency = torch.logical_xor(adjacency, noise)
    return adjacency

def _get_adjacency(graph):
    return to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).to(bool).squeeze()
