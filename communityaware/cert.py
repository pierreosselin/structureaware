from itertools import product

import numpy as np
from scipy.stats import binom


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
