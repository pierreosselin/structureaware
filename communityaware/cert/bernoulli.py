import numpy as np
import gmpy2
from scipy import special
from scipy.stats import binom
from functools import partial

def max_bernoulli_radius(p_A: float, p_noise: float, p_B: float=None, precision: int=1000):
    """Find the largest radius such that certify_bernoulli(p_A, p_noise, radius, p_B, precision) is True

    Returns:
        int: largest radius 
    """
    certificate = partial(certify_bernoulli, p_A=p_A, p_noise=p_noise, p_B=p_B, precision=precision)
    radius = 1
    while certificate(radius=radius):
        radius += 1
    return radius - 1

def certify_bernoulli(p_A: float, p_noise: float, radius: int, p_B: float=None, precision: int=1000) -> bool:
    """Return True if can certify around the point x.

    Args:
        p_A (float): lower bound on the probability of the predicted class
        p_noise (float): noise parameter used for edge flips
        radius (int): number of edge flips to certify with respect to
        p_B (float, optional): upper bound on the probability of the runner-up class. Defaults to 1-p_A.
        precision (int, optional): precision used for calculations. Defaults to 1000.

    Raises:
        ValueError: If p_noise >= 0.5 (we assume it to be less).

    Returns:
        bool: True if certified, else False.
    """
    # check inputs are valid
    if p_noise >= 0.5:
        raise ValueError('p_noise is assumed to be <= 0.5')
    if radius <= 0:
        raise ValueError('Expect radius to be non-negative integer.')
    if p_B is None:
        p_B = 1 - p_A
    
    with gmpy2.context(precision=precision):
        ## Compute certificate
        # compute likelihood ratios
        c = ((1-p_noise)/p_noise) ** np.arange(-radius, radius+1, 2)

        # compute P(X in regions) and find a* and b* with this
        prob_X_in_H = binom.pmf(k=np.arange(0, radius+1), n=radius, p=p_noise)
        a_star = np.where(np.cumsum(prob_X_in_H) > p_A)[0][0]
        b_star = radius-np.where(np.cumsum(prob_X_in_H[::-1]) > p_B)[0][0]

        # compute P(Y in regions) and find lower bound on p_A and upper bound on p_B for Y
        prob_Y_in_H = c * prob_X_in_H
        lower_bound = np.sum(prob_Y_in_H[:a_star+1])
        upper_bound = np.sum(prob_Y_in_H[b_star:])

        # certified if lower_bound > upper_bound
        certified = lower_bound > upper_bound

        ## sanity check calculations
        # Regions are disjoint unions of space
        assert np.isclose(sum(prob_X_in_H), 1) 
        assert np.isclose(sum(prob_Y_in_H), 1)

        # check a_star is correct and the smallest possible
        assert np.sum(prob_X_in_H[:a_star+1]) >= p_A  
        assert np.sum(prob_X_in_H[:a_star]) < p_A 

        # check b_star is correct and the largest possible          
        assert np.sum(prob_X_in_H[b_star:]) >= p_B 
        assert np.sum(prob_X_in_H[b_star+1:]) < p_B 

    return certified