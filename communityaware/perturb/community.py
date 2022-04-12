import jax
import jax.numpy as jnp
from functools import partial
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import copy

def sample_perturbed_graphs_with_sbm_noise(graph: torch_geometric.data.data.Data, sizes: tuple, p: np.ndarray, repeats: int, batch_size=32):
    """Return a batchloader of graphs with perturbed edges.

    Args:
        graph (torch_geometric.data.data.Data): graph to perturb
        sizes (tuple): sizes of the communities (assumed that nodes 0,...,sizes[0]-1 are in community 0, sizes[1], ... are in community 1 etc.)
        p (np.ndarray): p[i][j] is the probability of perturbed an edge between nodes i and j 
        repeats (int): number of times to repeat the perturbation
        batch_size (int, optional): batch size of the returned dataloader. Defaults to 32.

    Returns:
        torch_geometric.loader.DataLoader: dataloader of perturbed graphs
    """
    key = jax.random.PRNGKey(graph.idx.item())
    keys = jax.random.split(key, repeats)
    adjacency = jnp.array(to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes), dtype=bool).squeeze()
    sizes_cumsum = (0, ) + tuple(np.cumsum(sizes))
    p = jnp.array(p)
    perturbed_adjacencies = stochastic_block_model_noise(adjacency, sizes, sizes_cumsum, p, keys)
    new_edge_index = [from_scipy_sparse_matrix(sp.csr_matrix(a))[0] for a in np.array(perturbed_adjacencies)]
    graphs = [copy.copy(graph) for _ in range(repeats)]
    for i, graph in enumerate(graphs):
        graph.edge_index = new_edge_index[i]
    return DataLoader(graphs, batch_size=batch_size)


@partial(jax.jit, static_argnums=(1,2))
@partial(jax.vmap, in_axes=(None, None, None, None, 0), out_axes=0)
def stochastic_block_model_noise(adjacency: jnp.DeviceArray, sizes: tuple, sizes_cumsum: tuple, p: jnp.DeviceArray, key: jnp.DeviceArray) -> jnp.DeviceArray:
    """Applies stochastic block model noise to the adjacency matrix.

    Args:
        adjacency (jnp.DeviceArray): (repeats, num_nodes, num_nodes) adjacency matrix.
        sizes (tuple): (num_communities) size of each community. E.g. (20, 20, 20).
        sizes_cumsum (tuple): (num_communities+1) cumalative sum of sizes (with 0 at the beginning). e.g. (0, 20, 40, 60).
        p (jnp.DeviceArray): (num_communities, num_communities) probability of inter- and intra-community edges.
        key (jnp.DeviceArray): (repeats, 2) a key for the random number generator (one per sample). Can be generated using `jax.random.split(key, repeats)`.

    Returns:
        jnp.DeviceArray: (repeats, num_nodes, num_nodes) perturbed adjacency matrices.
    """
    noise = jnp.zeros_like(adjacency, dtype=bool)
    for i in range(len(sizes)):
        for j in range(i, len(sizes)):
            row_idx_range = sizes_cumsum[i], sizes_cumsum[i+1]
            col_idx_range = sizes_cumsum[j], sizes_cumsum[j+1]
            entries=jax.random.bernoulli(key, p[i, j], (sizes[i], sizes[j]))
            if i == j:
                entries = entries.at[jnp.tril_indices(sizes[i])].set(0)
            noise = noise.at[row_idx_range[0]:row_idx_range[1], col_idx_range[0]:col_idx_range[1]].set(entries)
    noise = noise + noise.T
    adjacency = jnp.logical_xor(adjacency, noise) 
    return adjacency