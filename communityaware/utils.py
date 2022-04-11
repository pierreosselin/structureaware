import torch
import os

def compute_p_from_sbm(p_block, list_blocks):
    """Find the equivalent parameter p from the Erdos-Renyi model that matches the average number of edges from the given SBM graph


    Args:
        p_block (2D Array): Matrix of probability from the SBM model
        list_blocks (1D Array): Number of nodes per block

    Returns:
        float: ER parameter 
    """
    n_graph = sum(list_blocks)
    exp_edges_sbm = 0
    n_list_blocks = len(list_blocks)
    
    if n_list_blocks != p_block.shape[0]:
        raise Exception("The list of blocks does not match the clusters probabilities")
    if p_block.shape[0] != p_block.shape[1]:
        raise Exception("The probability matrix is not square")
    if (p_block > 1.).any() or (p_block < 0.).any():
        raise Exception("The matrix is not a probability matrix")
    
    for i in range(n_list_blocks):
        for j in range(i):
            exp_edges_sbm += p_block[i][j] * list_blocks[i] * list_blocks[j]
        exp_edges_sbm += p_block[i][i] * list_blocks[i] * (list_blocks[i] - 1) / 2
    er_p = 2 * exp_edges_sbm / (n_graph*(n_graph - 1))
    return er_p



def er_parameter_from_sbm(sizes, p):
    """Given parameters of an SBM model returns an ER parameter such that the graphs will have same number of expected edges."""
    expected_edges = 0.0
    for i in range(len(sizes)):
        for j in range(i, len(sizes)):
            if i == j:
                expected_edges += (sizes[i] * (sizes[i] - 1) / 2) * p[i][i]
            else:
                expected_edges += sizes[i] * sizes[j] * p[i][j]
    n = sum(sizes)
    p = (2 * expected_edges) / (n * (n-1))  
    return p


def offset_idx(idx_mat: torch.LongTensor, lens: torch.LongTensor, dim_size: int, indices = [0]):
    """Offset the edge indices according to the number of nodes in the graph to format as a batch

    Args:
        idx_mat (torch.LongTensor): Array of edges
        lens (torch.LongTensor): Number of added edges per sample
        dim_size (int): Numbe of nodes in the graph
        indices (list, optional): Dimensions to offset. Defaults to [0].

    Returns:
        (torch.LongTensor): Offset matrix
    """
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

def mask_other_gpus(gpu_number):
    """Mask all other GPUs than the one specified."""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)
