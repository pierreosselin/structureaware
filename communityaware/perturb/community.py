import jax
import jax.numpy as jnp
from functools import partial
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import copy

def batch_perturbed_graph(graph: torch_geometric.data.data.Data, p: np.ndarray, repeats: int, batch_size=32):
    """Return a batchloader of graphs with perturbed edges.

    Args:
        graph (torch_geometric.data.data.Data): graph to perturb
        p (np.ndarray): p[i][j] is the probability of perturbed an edge between nodes i and j 
        repeats (int): number of times to repeat the perturbation
        batch_size (int, optional): batch size of the returned dataloader. Defaults to 32.

    Returns:
        torch_geometric.loader.DataLoader: dataloader of perturbed graphs
    """
    key = jax.random.PRNGKey(graph.idx.item())
    keys = jax.random.split(key, repeats)
    adjacency = jnp.array(to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes), dtype=bool).squeeze()
    perturbed_adjacencies = _batch_perturbed_graph(adjacency, jnp.array(p), keys)
    new_edge_index = [from_scipy_sparse_matrix(sp.csr_matrix(a))[0] for a in np.array(perturbed_adjacencies)]
    graphs = [copy.copy(graph) for _ in range(repeats)]
    for i, graph in enumerate(graphs):
        graph.edge_index = new_edge_index[i]
    return DataLoader(graphs, batch_size=batch_size)


@partial(jax.jit)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=0)
def _batch_perturbed_graph(adjacency: jnp.DeviceArray, p: jnp.DeviceArray, key: jnp.DeviceArray) -> jnp.DeviceArray:
    """Applies stochastic block model noise to the adjacency matrix.

    Args:
        adjacency (jnp.DeviceArray): (num_nodes, num_nodes) adjacency matrix.
        p (jnp.DeviceArray): (num_nodes, num_nodes) noise matrix
        key (jnp.DeviceArray): (repeats, 2) a key for the random number generator (one per sample). Can be generated using `jax.random.split(key, repeats)`.

    Returns:
        jnp.DeviceArray: (repeats, num_nodes, num_nodes) perturbed adjacency matrices.
    """
    noise = jax.random.bernoulli(key, p)
    noise = jnp.triu(noise, k=1) # zeros the diagonal and everything in the lower triangle
    noise = noise + noise.T
    adjacency = jnp.logical_xor(adjacency, noise) 
    return adjacency
