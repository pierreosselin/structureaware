import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch_sparse import coalesce


def load_perturbation(name):
    """Load the type of perturbation to use

    Args:
        name (str): Name of the perturbation

    Returns:
        [function]: Pertubation function
    """
    if name == "bernoulli":
        return sample_perturbed_graphs_bernoulli
    
    if name == "community":
        return sample_perturbed_graphs_community
    

def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices = [0]):
    offset = dim_size * torch.arange(len(lens), dtype=torch.long,
                                     device=idx_mat.device).repeat_interleave(lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat

def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int):
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(ncopies, dtype=torch.long,
                                     device=idx.device)[:, None].expand(ncopies, idx.shape[1]).flatten()

    idx_copies += offset[None, :]

    return idx_copies

def sample_perturbed_graphs_bernoulli(data_idx, n, batch_size, param_noise, **args):
    """Sample pertubed graphs according to bernoulli topological noise

    Args:
        data_idx ([2, n_edges] array): Array of the graph edges
        n (Int): Number of nodes in the graph
        batch_size (Int): Batch size for the number of graph to sample
        param_noise (Float): Bernoulli Parameter

    Returns:
        [2, m] array: Array of batch of perturbed graphs
    """
    # Select only direct edges
    data_idx = data_idx[:, data_idx[0] < data_idx[1]]

    # Copy the graph
    idx_copies = copy_idx(data_idx, n, batch_size)
    w_existing = torch.ones_like(idx_copies[0])

    # Sample edges to delete
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(param_noise)
    w_existing[to_del] = 0

    #To rewrite and test!!!
    nadd_persample_np = np.random.binomial(n * n, param_noise, size=batch_size)  # 6x faster than PyTorch
    nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
    nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * n))
                                            / np.log(1 - 1 / (n * n))).long()
    nadd_with_repl = nadd_persample_with_repl.sum()
    to_add = data_idx.new_empty([2, nadd_with_repl])
    to_add[0].random_(n * n)
    to_add[1] = to_add[0] % n
    to_add[0] = to_add[0] // n
    to_add = offset_idx(to_add, nadd_persample_with_repl, n, [0, 1])

    ## All graphs are undirected
    to_add = to_add[:, to_add[0] < to_add[1]]
    w_added = torch.ones_like(to_add[0])
    mb = batch_size * n

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value
    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               batch_size * n, mb, 'min')
    per_data_idx = joined[:, weights > 0]
    per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)
    
    return per_data_idx

def sample_perturbed_graphs_community(data_idx, n, batch_size, param_noise, community_node, community_size, community_prob):
    
    p_communities = param_noise*community_prob
    n_Clusters = community_size.shape[0]
    data_idx = data_idx[:, data_idx[0] < data_idx[1]]
    idx_copies = copy_idx(data_idx, n, batch_size)

    # Vector of size CxC with successes corresponding to the cardinal
    # print("Before data:", idx_copies)

    ## Next step: populate a tensor of size number of successes, populate random number size of the corresponding cluster and assign node through mapping
    to_add_total = torch.empty(2, 0).long().cuda()

    # Half the loops are useless here, and not parallelized
    for c1 in range(n_Clusters):
        for c2 in range(n_Clusters):
            nadd_persample_np = np.random.binomial(community_size[c1] * community_size[c2], p_communities[c1, c2], size=batch_size)  # 6x faster than PyTorch
            nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
            val = torch.log(1 - nadd_persample / (community_size[c1] * community_size[c2])) / np.log(1 - 1 / (community_size[c1] * community_size[c2]))
            if torch.isinf(val.sum()):
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
            to_add[0] = community_node[c1][to_add[0]]
            to_add[1] = community_node[c2][to_add[1]]
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




def predict_smooth_model(attr_idx, edge_idx, pf, model, n, d, nc, n_samples, batch_size=1):

    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=edge_idx.device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size
        for _ in tqdm(range(nbatches)):
            edge_idx_batch = sample_perturbed_graphs(data_idx=edge_idx, n=n, m=n, undirected=True,
                                               pf=pf, nsamples=batch_size, offset_both_idx=True)
            attr_idx_batch = copy_idx(idx=attr_idx, dim_size=n, ncopies=batch_size, offset_both_idx=False)
            predictions = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                n=batch_size * n, d=d).argmax(1)
            preds_onehot = F.one_hot(predictions, int(nc)).reshape(batch_size, n, nc).sum(0)
            votes += preds_onehot
    return votes.cpu().numpy()




def predict_smooth_gnn_classification_community(loader, n_samples, param_noise, dataset_name, model, n, nc, batch_size=1):
    
    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size

        # Loop over the graphs in the dataset
        for i, data in enumerate(loader):
            #Get the graph structure and attributes
            edge_idx = data.edge_index.cuda()
            x = data.x.cuda()
            n_graph = x.shape[0]
            x_batch = x.repeat(batch_size, 1)
            batch_idx = torch.arange(batch_size, device=edge_idx.device).repeat_interleave(n_graph, dim=0)
            node_community = data.node_community
            community_node = [torch.tensor(el).clone().detach().cuda() for el in data.community_node[0]]
            community_size = data.community_size
            community_prob = alpha*data.community_prob

            # Loop over the perturbation graph batches
            for _ in range(nbatches):

                ### First function: should output a batch of perturbed graph the size
                # of batch_size. Including x_batch, edge_idx_batch and batch_idx.
                # Could improve by copying x
                edge_idx_batch = sample_multiple_graphs_classification_community(edge_idx=edge_idx, sample_config = sample_config,
                    node_community=node_community, community_node=community_node, community_size=community_size, community_prob=community_prob,
                    nsamples=batch_size, n_nodes=n_graph)
                
                if dataset_name == "synthetic":
                    ### Recompute the features
                    ## With feature recomputation
                    #adj = skn.utils.edgelist2adjacency(np.array(edge_idx_batch.cpu()).T)
                    #betweenness = skn.ranking.Katz()
                    #scores = betweenness.fit_transform(adj)
                    G = nx.Graph()
                    G.add_nodes_from(range(x_batch.shape[0]))
                    G.add_edges_from(np.array(edge_idx_batch.cpu()).T)
                    d2 = nx.algorithms.cluster.clustering(G)
                    x_batch[:, 0] = torch.tensor([d2[i] for i in range(n_graph)])
                    #x_batch[:, 0] = torch.tensor(scores)

                predictions = model(x=x_batch, edge_index=edge_idx_batch,
                                    batch=batch_idx).argmax(1)
                preds_onehot = F.one_hot(predictions.to(torch.int64), int(nc)).sum(0)

                votes[i] += preds_onehot

            if i%10 == 0:
                print(f'Processed {i}/{n} graphs')
    return votes.cpu().numpy()