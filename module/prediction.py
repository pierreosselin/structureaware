import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from module.utils import offset_idx, copy_idx
from torch_sparse import coalesce


def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices: List[int] = [0]):
    offset = dim_size * torch.arange(len(lens), dtype=torch.long,
                                     device=idx_mat.device).repeat_interleave(lens, dim=0)

    idx_mat[indices, :] += offset[None, :]
    return idx_mat

def copy_idx(idx: torch.LongTensor, dim_size: int, ncopies: int, offset_both_idx: bool):
    idx_copies = idx.repeat(1, ncopies)

    offset = dim_size * torch.arange(ncopies, dtype=torch.long,
                                     device=idx.device)[:, None].expand(ncopies, idx.shape[1]).flatten()

    if offset_both_idx:
        idx_copies += offset[None, :]
    else:
        idx_copies[0] += offset

    return idx_copies

def sample_perturbed_graphs(data_idx, pf_minus, pf_plus, n, m, undirected, nsamples, offset_both_idx):

    if undirected:
        # select only one direction of the edges, ignore self loops
        data_idx = data_idx[:, data_idx[0] < data_idx[1]]


    idx_copies = copy_idx(data_idx, n, nsamples, offset_both_idx)
    w_existing = torch.ones_like(idx_copies[0])
    #batch_existing = torch.arange(nsamples, device=data_idx.device).repeat_interleave(data_idx.shape[1], dim=0)
    to_del = torch.cuda.BoolTensor(idx_copies.shape[1]).bernoulli_(pf_minus)
    w_existing[to_del] = 0

    if offset_both_idx:
        assert n == m
        nadd_persample_np = np.random.binomial(n * m, pf_plus, size=nsamples)  # 6x faster than PyTorch
        nadd_persample = torch.cuda.FloatTensor(nadd_persample_np)
        nadd_persample_with_repl = torch.round(torch.log(1 - nadd_persample / (n * m))
                                               / np.log(1 - 1 / (n * m))).long()
        nadd_with_repl = nadd_persample_with_repl.sum()
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m
        to_add = offset_idx(to_add, nadd_persample_with_repl, m, [0, 1])
        #batch_idx = generate_batch_idx(to_add, nadd_persample_with_repl, m, [0, 1])


        if undirected:
            # select only one direction of the edges, ignore self loops
            #batch_idx = batch_idx[to_add[0] < to_add[1]]
            to_add = to_add[:, to_add[0] < to_add[1]]

    else:
        nadd = np.random.binomial(nsamples * n * m, pf_plus)  # 6x faster than PyTorch
        nadd_with_repl = int(np.round(np.log(1 - nadd / (nsamples * n * m))
                                      / np.log(1 - 1 / (nsamples * n * m))))
        to_add = data_idx.new_empty([2, nadd_with_repl])
        to_add[0].random_(nsamples * n * m)
        to_add[1] = to_add[0] % m
        to_add[0] = to_add[0] // m

    w_added = torch.ones_like(to_add[0])

    if offset_both_idx:
        mb = nsamples * m
    else:
        mb = m

    # if an edge already exists but has been removed do not add it back
    # hence we coalesce with the min value


    joined, weights = coalesce(torch.cat((idx_copies, to_add), 1),
                               torch.cat((w_existing, w_added), 0),
                               nsamples * n, mb, 'min')

    #_, batchs = coalesce(torch.cat((idx_copies, to_add), 1),
    #                           torch.cat((batch_existing, batch_idx), 0),
    #                           nsamples * n, mb, 'min')

    per_data_idx = joined[:, weights > 0]
    #batch_idx = batchs[weights > 0]

    if undirected:
        per_data_idx = torch.cat((per_data_idx, per_data_idx[[1, 0]]), 1)
        #batch_idx = torch.cat((batch_idx, batch_idx), 0)

    return per_data_idx


def predict_smooth_gnn(attr_idx, edge_idx, pf, model, n, d, nc, n_samples, batch_size=1):
    ### Raise error for value perturbation parameter
    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=edge_idx.device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size
        for _ in tqdm(range(nbatches)):
            edge_idx_batch = sparse_perturb_multiple(data_idx=edge_idx, n=n, m=n, undirected=True,
                                               pf=pf, nsamples=batch_size, offset_both_idx=True)
            attr_idx_batch = copy_idx(idx=attr_idx, dim_size=n, ncopies=batch_size, offset_both_idx=False)
            predictions = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                n=batch_size * n, d=d).argmax(1)
            preds_onehot = F.one_hot(predictions, int(nc)).reshape(batch_size, n, nc).sum(0)
            votes += preds_onehot
    return votes.cpu().numpy()
