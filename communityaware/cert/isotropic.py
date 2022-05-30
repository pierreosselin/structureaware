from typing import Tuple

import gmpy2
import numpy as np


def isotropic(r: int, p: float, p_A: float, precision: int=100) -> gmpy2.mpfr:
    """Compute certificate for Isotropic noise.

    Given p_A, the lower bound of the majority class A of the unperturbed input, as well as P the value of noise
    used in the randomised smoothing. This method will return a float (in gmpy2.mpfr format) which if
    above 0.5 means we can certify to radius R.

    Args:
        r (int): Radius at what to certify at.
        p (float): The binomial noise paremeter used for the smoothed classifier.
        p_A (float): Lower bound on the majority class A.
        precision (int, optional): precision of computation. Defaults to 100.

    Returns:
        gmpy2.mpfr: lower_bound of the smoothed classifier predicting class A for the perturbed input.
    """

    # preprocess values
    with gmpy2.context(precision=precision):

        # compute region information
        likelihood_ratios, region_probability = compute_regions(gmpy2.mpz(r), gmpy2.mpfr(p))

        # find lower bound on probability.
        final_region = compute_final_region(region_probability, p_A)

        # lower_bound
        lower_bound = compute_lower_bound(likelihood_ratios, region_probability, final_region, p_A)

        # sanity check and return output
        assert 0 <= lower_bound <= 1.0 or np.isclose(lower_bound, 0) or np.isclose(lower_bound, 1.0)
        if not (0 <= lower_bound <= 1.0):
            raise ValueError('lower_bound should be in [0, 1]')
        return lower_bound

def compute_regions(r: gmpy2.mpz, p: gmpy2.mpfr) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the likelihood ratio of each region as well as the probability the smoothed input belongs to the region.

    Args:
        r (gmpy2.mpz): maximum radius.
        p (gmpy2.mpfr): noise parameter.

    Returns:
        np.ndarray: A (r+1, 2) array. The first column is likelihood ratios and the second is probability the smoothed input belongs to the region.
                    The array is sorted in ascending order of the likelihood ratios.
    """
    p_const = p / (1-p)
    sol = np.zeros((r+1 , 2), dtype=gmpy2.mpfr) # first column is c_k, second is P(phi(X) belonging to region)
    for Q in range(r+1): # all possible combinations of Q s.t. 0 <= Q_i <= R_i
        qymp_Q = gmpy2.mpz(Q)
        sol[Q][0] = p_const ** (2 * qymp_Q - r) # Compute c_k = P(phi(Y))/P(phi(X))
        sol[Q][1] = gmpy2.bincoef(r, r-qymp_Q) * p**(r-qymp_Q) * (1-p)**(qymp_Q) # Compute P(phi(X) belonging to region)
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
    return np.where(np.cumsum(region_probability) > p_A)[0][0]  #

def compute_lower_bound(likelihood_ratios, region_probability, final_region, p_A, use_partial_region=True):
    lower_bound = np.sum(region_probability[:final_region] * likelihood_ratios[:final_region]) # p(Y) in corresponding region.

    # use partial region if possible.
    p = np.sum(region_probability[:final_region])
    if p_A > p and use_partial_region:
        lower_bound += (p_A-p) * likelihood_ratios[final_region]
    return lower_bound
