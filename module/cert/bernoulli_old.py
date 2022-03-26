import numpy as np
import gmpy2
from scipy import special
from scipy.stats import binom


def certify_bernoulli(pstar, pprime, pnoise, l_max = 10):
    """Compute certification radius

    Args:
        pstar (float): Probability best label
        pprime (float): Probability second best label
        pnoise (float): Noise Bernoulli parameter
        l_max (int, optional): Maximal Radius to certify. Defaults to 10.

    Returns:
        int: Radius
    """
    l_max += 1 # why?

    #Precompute table of probabilities
    table_regions = pre_compute_regions_bernoulli(l_max, pnoise)
    reversed_table_regions = table_regions[:,::-1] # reorder columns 

    # Compute the cumulative sums of the probability that X are in curly H and curly Q
    table_regions_cum_sum = np.cumsum(table_regions,axis=1) # [i, j] entry is sum_k=0^j of Bin(k; i, p) (CDF of the binomial distribution)
    reversed_table_regions_cum_sum = np.cumsum(reversed_table_regions,axis=1) # This might be flawed because of all the zeros... 

    # Compute lower bound on right label
    y_lower = compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar) # lower bound for each l 

    # Compute upper bound on wrong label
    y_upper = compute_upper_bound_bernoulli(l_max, table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pprime)  # upper bound for each l 
    
    return np.argmax(y_lower < y_upper) - 1 # best l


def pre_compute_regions_bernoulli(max_l, p):
    """Pre compute the regions probabilities

    Args:
        max_l (int): maximum radius to certify
        p (float): noise probability

    Returns:
        table_regions (2d array): Table of regions probability table_regions[i, j] corresponds to H_{j}^{i} 
    """
    table_regions = np.zeros((max_l, max_l))
    for i in range(max_l):
        for j in range(i+1):
            table_regions[i,j] = special.comb(i, j,exact=True) * (p**j) * ((1-p)**(i-j))  # H[i,j] = Bin(j; i, p)
            assert np.isclose(table_regions[i, j], binom.pmf(j, i, p)) 
    return table_regions



def compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar):
    # Compute the threshold of regions for X
    k_star = np.argmax((table_regions_cum_sum > pstar), axis = 1) # a_star, one per row (radius)

    # Compute difference cum_sum at kstar and before to have proportion in Hij then keep this proportion
    a = np.zeros((l_max,l_max+1)) 
    a[:,1:] = table_regions_cum_sum # table_regions_cum_sum with an entra row/column of zeros?
    diff = pstar - a[np.arange(l_max), k_star] #Probability volume remaining in the last unaccepted region
    prop = diff / table_regions[np.arange(l_max),k_star] #Proportion of this region that needs to be included

    # Proba Y lower bounds
    b = np.zeros((l_max, l_max+1))
    b[:,1:] = reversed_table_regions_cum_sum
    indices = np.array(range(1, l_max+1))
    y_lower = b[np.arange(l_max), l_max - indices + k_star] + prop*reversed_table_regions[np.arange(l_max), l_max - indices + k_star]

    return y_lower

def compute_upper_bound_bernoulli(l_max, table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pprime):
    
    indices = np.array(range(1, l_max+1))
    a = np.zeros((l_max,l_max+1))
    a[:,1:] = table_regions_cum_sum

    # Compute the threshold of regions for X
    c = np.zeros((l_max,l_max+1))
    c[:,:-1] = reversed_table_regions_cum_sum[:, ::-1]
    k_prime = np.argmax((c < pprime), axis = 1) # >0

    # Compute difference cum_sum at kstar and before to have proportion in Hij then keep this proportion
    diff = pprime - c[np.arange(l_max), k_prime] #Probability volume remaining in the last unaccepted region
    prop = diff / table_regions[np.arange(l_max), k_prime - 1] #Proportion of this region that needs to be included

    # Proba Y upper bounds
    y_upper = a[np.arange(l_max), indices - k_prime] + prop*table_regions[np.arange(l_max), indices - k_prime]

    return y_upper