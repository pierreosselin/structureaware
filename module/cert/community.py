import numpy as np
from itertools import product
from scipy.stats import binom
import gmpy2

def compute_communities(R: np.ndarray, P: np.ndarray, p_A: float, precision: int=1000):
    """
    TODO: write tests <- isotropic if P is always the same... 
    TODO: do forward and backward with a heap

    Args:
        R (np.ndarray): R_ij is the max number of flips between C_i and C_j
        P (np.ndarray): P_ij is the probability of flipping an edge between C_i and C_j
        P_A: estimated (lower bound) on probability of majority class
        precision: precision for computations
    """

    # check inputs are valid
    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError('R should be a square array.')
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('P should be a square array.')
    if not np.allclose(P.shape, R.shape):
        raise ValueError('P and R should be the same shape.')
    if not np.allclose(P, P.T):
        raise ValueError('P should be symmetric.')
    if not np.allclose(R, R.T):
        raise ValueError('R should be symmetric.')
    if not 0 <= p_A <= 1:
        raise ValueError('p_A should be in [0, 1]')
    if not np.all(P <= 0.5):
        raise ValueError('P_ij is assumed to be <= 0.5')
    
    # extra upper triangle of R and P and flatten into a vector
    R = R[np.triu_indices(R.shape[0])]
    P = P[np.triu_indices(P.shape[0])] 
    
    with gmpy2.context(precision=precision):
        sol = np.zeros((np.prod(R+1) , 2)) # first column is c_k, second is P(phi(X) belonging to region)
        for reg_idx, Q in enumerate(product(*[list(range(i+1)) for i in R])): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
            sol[reg_idx,0] = np.prod(((1 - P) / P) ** (R - 2*np.array(Q))) # Compute c_k
            sol[reg_idx,1] = np.prod(binom.pmf(Q, R, P)) # Compute P(phi(X) belonging to region)
        sol = sol[sol[:, 0].argsort()] # sort by c_k

        # find lower bound on probability phi(Y) is c_A
        final_region = np.where(np.cumsum(sol[:, 0]) > p_A)[0][0]-1 # find largest a* s.t. sum_i^a* c_k <= p_A
        lower_bound = np.sum(sol[:final_region, 1] / sol[:final_region, 0]) # p(Y) in corresponding region

    return lower_bound
