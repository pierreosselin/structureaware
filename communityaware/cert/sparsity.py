from collections import deque
from functools import lru_cache
from itertools import accumulate, product
from typing import Tuple

import gmpy2
import numpy as np
from scipy.stats import binom


def sparsity(R: np.ndarray, P: np.ndarray, p_A: float, precision: int=100) -> gmpy2.mpfr:
    """_summary_

    Args:
        R (np.ndarray): (r_delete, r_add)
        P (np.ndarray): (p_delete, p_add)
        p_A (float): _description_
        precision (int, optional): _description_. Defaults to 100.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        gmpy2.mpfr: _description_
    """
    # check inputs are valid
    if np.sum(R) == 0:
        raise ValueError('R should not be identically zero.')
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if len(R) != 2 or len(P) != 2:
        raise ValueError('R and P should be of length 2.')

    # preprocess values
    with gmpy2.context(precision=precision):
        gymp_R = tuple(gmpy2.mpz(float(i)) for i in R)
        gymp_p = tuple(gmpy2.mpfr(i) for i in P)
        likelihood_ratios, region_probability = compute_regions(gymp_R, gymp_p)

        final_region = compute_final_region(region_probability, p_A)

        lower_bound = compute_lower_bound(likelihood_ratios, region_probability, final_region, p_A)

    # sanity check on output
    assert 0 <= lower_bound <= 1.0 or np.isclose(lower_bound, 0) or np.isclose(lower_bound, 1.0)
    if not (0 <= lower_bound <= 1.0):
        raise ValueError('lower_bound should be in [0, 1]')
    return lower_bound

def compute_regions(r, p) -> tuple[np.ndarray, np.ndarray]:
    # r: tuple[gmpy2.mpz, gmpy2.mpz], p: tuple[gmpy2.mpfr, gmpy2.mpfr] not sure hwy mypy doesnt like this...
    # compute region information.
    sol = np.zeros((int(r[0]+r[1]+1) , 2), dtype=gmpy2.mpfr) # first column is c_k, second is P(phi(X) belonging to region)
    for Q in range(r[0]+r[1]+1): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
        qymp_Q = gmpy2.mpz(Q)
        sol[Q][0] = ratio(qymp_Q, r, p) # Compute c_k = P(phi(Y))/P(phi(X))
        sol[Q][1] = P_phi_x_in_region(qymp_Q, r, p) # Compute P(phi(X) belonging to region)
    sol = sol[sol[:, 0].argsort()] # sort by c_k
    likelihood_ratios, region_probability = sol[:, 0], sol[:, 1]
    return (likelihood_ratios, region_probability)

def compute_final_region(region_probability, p_A):
    """find largest a* s.t. sum_i^a* p_k <= p_A where p_k are the region_probabilities.

    Args:
        region_probability (_type_): _description_
        p_A (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.where(np.cumsum(region_probability) > p_A)[0][0]

def compute_lower_bound(likelihood_ratios, region_probability, final_region, p_A, use_partial_region=True):
    lower_bound = np.sum(region_probability[:final_region] * likelihood_ratios[:final_region]) # p(Y) in corresponding region.

    # use partial region if possible.
    p = np.sum(region_probability[:final_region])
    if p_A > p and use_partial_region:
        lower_bound += (p_A-p) * likelihood_ratios[final_region]
    return lower_bound


@lru_cache(maxsize=None)
def ratio(qymp_Q, qymp_R, gymp_p):
    first_term = ((1-gymp_p[0])/(gymp_p[1])) ** (qymp_Q - qymp_R[0])
    second_term = ((1-gymp_p[1])/(gymp_p[0])) ** (qymp_Q - qymp_R[1])
    return first_term * second_term

@lru_cache(maxsize=None)
def P_phi_x_in_region(qymp_Q, qymp_R, gymp_p):
    return PB_R(qymp_R, gymp_p, qymp_Q) * (1-gymp_p[1])**qymp_R[1] * (1-gymp_p[0])**qymp_R[0]

@lru_cache(maxsize=None)
def PB_T(R, P, i):
    return R[1] * ((P[1])/(1-P[1]))**i + R[0] * ((P[0])/(1-P[0]))**i

@lru_cache(maxsize=None)
def PB_R(R, P, q):

    # base case
    if q == 0:
        return gmpy2.mpfr(1.0)

    # recursive case
    sol = gmpy2.mpfr(0.0)
    for i in range(1, q+1):
        sol = sol + ((-1) ** (i+1)) * PB_T(R, P, i) * PB_R(R, P, q-i)
    sol = sol / q
    return sol

if __name__=='__main__':
    sparsity(
        R=np.array((1, 2)),
        P=np.array((0.1, 0.4)),
        p_A=0.5
    )
