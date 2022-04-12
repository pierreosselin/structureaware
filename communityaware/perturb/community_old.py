import torch
import numpy as np
from torch_sparse import coalesce
from ..utils import copy_idx, offset_idx

def sample_perturbed_graphs_community(data_idx, n, batch_size, param_noise, node_community, community_size, community_prob, device):
    """Sample pertubed graphs according to community topological noise

    Args:
        data_idx ([2, n_edges] array): Array of the graph edges
        n (int): Number of nodes in the graph
        batch_size (int): Batch size for the number of graph to sample
        param_noise (float): Coefficient of proportionality for the probability matrix
        node_community (array):  Array representing the list of nodes in each community
        community_size ([int]): List of community sizes
        community_prob (array): Probability of inter and intra clustering

    Returns:
        array: Batch of sampled graph according to the noise distribution
    """
    p_communities = param_noise*community_prob
    n_Clusters = community_size.shape[0]
    data_idx = data_idx[:, data_idx[0] < data_idx[1]]
    idx_copies = copy_idx(data_idx, n, batch_size) # batch the data_idx

    ## Next step: populate a tensor of size number of successes, populate random number size of the corresponding cluster and assign node through mapping
    to_add_total = torch.empty(2, 0).long().to(device)

    # Half the loops are useless here, and not parallelized
    for c1 in range(n_Clusters):
        for c2 in range(n_Clusters): # not c1, n_Clusters ... ?

            # number samples at add in C1, C2
            nadd_persample_np = np.random.binomial(community_size[c1] * community_size[c2], p_communities[c1, c2], size=batch_size)  
            nadd_persample = torch.FloatTensor(nadd_persample_np).to(device) 
            val = torch.log(1 - nadd_persample / (community_size[c1] * community_size[c2])) / np.log(1 - 1 / (community_size[c1] * community_size[c2]))

            if torch.isinf(val.sum()): # why would it be inf? community_size[c1] could be 0... nadd_persample could equal 1... ???
                t = torch.isinf(val).nonzero()
                val[t] = community_size[c1] * community_size[c2] * torch.ones_like(t).float()
                nadd_persample_with_repl = torch.round(val).long()
                nadd_with_repl = nadd_persample_with_repl.sum()
                to_add = data_idx.new_empty([2, nadd_with_repl])
                to_add[0].random_(community_size[c1] * community_size[c2])

                edges_indexing = torch.cumsum(nadd_persample_with_repl, dim=0)
                edges_indexing = torch.cat((torch.tensor([0], device=data_idx.device), edges_indexing), dim = 0)
                for el in t:
                    to_add[0][edges_indexing[el]:edges_indexing[el+1]] = torch.arange(community_size[c1] * community_size[c2], device=data_idx.device)
            else:
                nadd_persample_with_repl = torch.round(val).long()
                nadd_with_repl = nadd_persample_with_repl.sum()
                to_add = data_idx.new_empty([2, nadd_with_repl])
                to_add[0].random_(community_size[c1] * community_size[c2])
            to_add[1] = to_add[0] // community_size[c1]
            to_add[0] = to_add[0] % community_size[c1]
            to_add[0] = node_community[c1][to_add[0]]
            to_add[1] = node_community[c2][to_add[1]]
            to_add = offset_idx(to_add, nadd_persample_with_repl, n, [0, 1])

            to_add = to_add[:, to_add[0] < to_add[1]]
            to_add_total = torch.cat((to_add_total, to_add), dim = 1)
    
    mb = batch_size * n
    
    # We need 2 coalesce, one to filter out the duplicates and another one for the addition modulo 2
    to_add_total, _ = coalesce(to_add_total, torch.ones_like(to_add_total[0]).long(),
                               batch_size * n, mb, 'min')

    to_add_total = torch.cat((to_add_total, idx_copies), dim = 1)
    to_add_total, weights = coalesce(to_add_total, torch.ones_like(to_add_total[0]),
                               batch_size * n, mb)

    per_data_idx = to_add_total[:, weights == 1]
    per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)

    return per_data_idx