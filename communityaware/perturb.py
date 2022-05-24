import copy
import functools
from functools import partial

import functorch
import functorch.compile
import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from communityaware.data import Synthetic


@torch.no_grad()
def batch_perturbed_graph(graph: torch_geometric.data.data.Data, p: torch.tensor, repeats: int, batch_size=32, device='cpu'):
    adjacency = get_adjacency(graph).to(device)
    adjacency = adjacency.unsqueeze(0).repeat(repeats, 1, 1)
    p = p.unsqueeze(0).repeat(repeats, 1, 1).to(device)
    perturbed_adjacencies = perturbed_graph(adjacency, p)
    graphs = [copy.copy(graph) for _ in range(repeats)]
    for i, graph in enumerate(graphs):
        graph.edge_index = dense_to_sparse(perturbed_adjacencies[i])[0]
    return DataLoader(graphs, batch_size=batch_size)

@partial(functorch.vmap, randomness='different')
@torch.no_grad()
def perturbed_graph(adjacency: torch.tensor, p: torch.tensor) -> torch.tensor:
    noise = torch.bernoulli(p)
    noise = torch.triu(noise, diagonal=1)
    noise = noise + noise.T
    adjacency = torch.logical_xor(adjacency, noise)
    return adjacency

@functools.lru_cache(maxsize=None)
def get_adjacency(graph):
    return to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes).to(bool).squeeze()
