from collections import deque
from functools import lru_cache
from itertools import accumulate, product

import gmpy2
import numpy as np
from scipy.stats import binom


def compute_certificate_gmpy(R: np.ndarray, P: np.ndarray, p_A: float, precision: int=100):
    # check inputs are valid
    if np.sum(R) == 0:
        raise ValueError('R should not be identically zero.')
    if not isinstance(R, np.ndarray):
        R = np.array(R)
    if not isinstance(P, np.ndarray):
        P = np.array(P)

    # preprocess values
    with gmpy2.context(precision=precision):
        gymp_p = [gmpy2.mpfr(i) for i in P]
        gymp_p_const = tuple(p/(1-p) for p in gymp_p)
        gymp_p_const_squared = tuple(i**2 for i in gymp_p_const)
        qymp_R = tuple(gmpy2.mpz(float(i)) for i in R)

        # compute region information.
        sol = np.zeros((np.prod(R+1) , 2), dtype=gmpy2.mpfr) # first column is c_k, second is P(phi(X) belonging to region)
        for reg_idx, Q in enumerate(product(*[list(range(i+1)) for i in R])): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
            qymp_Q = tuple(gmpy2.mpz(i) for i in Q)
            sol[reg_idx][0] = ratio(qymp_Q, gymp_p_const, gymp_p_const_squared, qymp_R) # Compute c_k = P(phi(Y))/P(phi(X))
            sol[reg_idx][1] = P_phi_x_in_region(qymp_Q, qymp_R, gymp_p) # Compute P(phi(X) belonging to region)
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

@lru_cache(maxsize=None)
def ratio(Q, gymp_p_const, gymp_p_const_squared, qymp_R):
    del_i = [i for i, j in enumerate(Q) if j > 0] # indicies i where Q_i > 0

    # base case
    # compute the ratio when Q is all zeros
    if len(del_i) == 0:
        to_multiply = [i ** -j for i, j in zip(gymp_p_const, qymp_R)]
        prod = gmpy_prod(to_multiply)
        return prod

    # recursive case
    # This works by finding where Q is non-zero, and subtracting 1 from this.
    # We can then compute recursively the function value by first computing the function with this new Q
    # and then multipling this by (p_i / (1-p_i))^2 where i is where the non-zero is.

    del_i = del_i[0] # take the first entry in the list to be i
    Q = list(Q)
    Q[del_i] -=  1
    Q = tuple(Q)
    return gymp_p_const_squared[del_i] * ratio(Q, gymp_p_const, gymp_p_const_squared, qymp_R)

def P_phi_x_in_region(Q, R, P):
    terms = [gmpy2.bincoef(r, r-q) * p**(r-q) * (1-p)**(q) for p, r, q in zip(P, R, Q)]
    prod = gmpy_prod(terms)
    return prod

def gmpy_prod(iterable):
    return deque(accumulate(iterable, gmpy2.mul), maxlen=1).pop()
