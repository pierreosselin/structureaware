from .utils import triangle_number, partition
import numpy as np
import gmpy2
from scipy.stats import binom
from itertools import product


# def inner_outer_certificate(inner: int, outer: int, P: np.ndarray, p_A: float, precision: int=1000):
#     """Certifies inner edge flips within communities and outer edge flips between communities"""
#     number_of_communities = P.shape[0]
#     inside_partitions = partition(inner, number_of_communities)
#     outside_partitions = partition(outer, triangle_number(number_of_communities-1))
#     for (inner_partition, outer_partion) in product(inside_partitions, outside_partitions):
#         R = np.zeros_like(P, dtype=np.int32)
#         R[np.triu_indices(R.shape[0], 1)] = outer_partion # put outer_partition in top triangle
#         R = R + R.T # put outer_partition in lower triangle
#         np.fill_diagonal(R, inner_partition) # put inner_partition on diag
#         lower_bound = compute_communities(R, P, p_A, precision)
#         if lower_bound <= 0.5:
#             return False
#     return True



# def compute_communities(R: np.ndarray, P: np.ndarray, p_A: float, precision: int=1000):
#     """
#     Args:
#         R (np.ndarray): R_ij is the max number of flips between C_i and C_j
#         P (np.ndarray): P_ij is the probability of flipping an edge between C_i and C_j
#         P_A: estimated (lower bound) on probability of majority class
#         precision: precision for computations
#     """

#     # check inputs are valid
#     if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
#         raise ValueError('R should be a square array.')
#     if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
#         raise ValueError('P should be a square array.')
#     if not np.allclose(P.shape, R.shape):
#         raise ValueError('P and R should be the same shape.')
#     if not np.allclose(P, P.T):
#         raise ValueError('P should be symmetric.')
#     if not np.allclose(R, R.T):
#         raise ValueError('R should be symmetric.')
#     if not 0 <= p_A <= 1:
#         raise ValueError('p_A should be in [0, 1]')
#     if not np.all(P <= 0.5):
#         raise ValueError('P_ij is assumed to be <= 0.5')
    
#     # extra upper triangle of R and P and flatten into a vector
#     R = R[np.triu_indices(R.shape[0])]
#     P = P[np.triu_indices(P.shape[0])]
    
#     with gmpy2.context(precision=precision):
#         sol = np.zeros((np.prod(R+1) , 2)) # first column is c_k, second is P(phi(X) belonging to region)
#         for reg_idx, Q in enumerate(product(*[list(range(i+1)) for i in R])): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
#             sol[reg_idx,0] = np.prod(((1 - P) / P) ** (R - 2*np.array(Q))) # Compute c_k
#             sol[reg_idx,1] = np.prod(binom.pmf(Q, R, P)) # Compute P(phi(X) belonging to region)
#         sol = sol[sol[:, 0].argsort()] # sort by c_k

#         # find lower bound on probability phi(Y) is c_A
#         final_region = np.where(np.cumsum(sol[:, 0]) > p_A)[0][0]-1 # find largest a* s.t. sum_i^a* c_k <= p_A
#         lower_bound = np.sum(sol[:final_region, 1] / sol[:final_region, 0]) # p(Y) in corresponding region

#     return lower_bound

def compute_certificate(R: np.ndarray, P: np.ndarray, p_A: float, precision: int=1000):
    """Compute certificate for communities. If lower_bound is above 0.5 then the certificate holds.

    Args:
        R (np.ndarray): shape (2,) where entries are budget of intra/outer edge flips
        P (np.ndarray):  shape (2,) where entries are probability of intra/outer edge flips
        p_A (float): estimated (lower bound) on probability of majority class
        precision (int, optional): precision for computations. Defaults to 1000.
    """

    # check inputs are valid
    if np.sum(R) == 0:
        raise ValueError('R should not be identically zero.')
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    if not isinstance(P, np.ndarray):
        P = np.array(P)

    # compute region information.
    sol = np.zeros((np.prod(R+1) , 2)) # first column is c_k, second is P(phi(X) belonging to region)
    for reg_idx, Q in enumerate(product(*[list(range(i+1)) for i in R])): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
        sol[reg_idx,0] = np.prod((P / (1-P)) ** (2*np.array(Q)-R)) # Compute c_k = P(phi(Y))/P(phi(X))
        sol[reg_idx,1] = np.prod(binom.pmf(R-Q, R, P)) # Compute P(phi(X) belonging to region)
    sol = sol[sol[:, 0].argsort()] # sort by c_k

    # find lower bound on probability 
    final_region = np.where(np.cumsum(sol[:, 1]) > p_A)[0][0]  # find largest a* s.t. sum_i^a* c_k <= p_A. + 1 because [: n] is not inclusive of n.
    lower_bound = np.sum(sol[:final_region, 1] * sol[:final_region, 0]) # p(Y) in corresponding region. 
        
    # use partial region if possible TODO: check this is correct.
    p = np.sum(sol[:final_region, 1])
    if p_A > p : 
        lower_bound += (p_A-p) * sol[final_region][0]

    # sanity check on output
    assert 0 <= lower_bound <= 1.0 or np.isclose(lower_bound, 0) or np.isclose(lower_bound, 1.0)
    if not (0 <= lower_bound <= 1.0):
        raise ValueError('lower_bound should be in [0, 1]')
    return lower_bound
