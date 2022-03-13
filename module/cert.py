import numpy as np
from scipy import special



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
            table_regions[i,j] = special.comb(i, j,exact=True) * (p**j) * ((1-p)**(i-j))
    return table_regions


def compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar):
    # Compute the threshold of regions for X
    k_star = np.argmax((table_regions_cum_sum > pstar), axis = 1)

    # Compute difference cum_sum at kstar and before to have proportion in Hij then keep this proportion
    a = np.zeros((l_max,l_max+1))
    a[:,1:] = table_regions_cum_sum
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
    l_max += 1
    #Precompute table of probabilities
    table_regions = pre_compute_regions_bernoulli(l_max, pnoise)
    reversed_table_regions = table_regions[:,::-1]

    # Compute the cumulative sums of the probability that X are in curly H and curly Q
    table_regions_cum_sum = np.cumsum(table_regions,axis=1)
    reversed_table_regions_cum_sum = np.cumsum(reversed_table_regions,axis=1)

    # Compute lower bound on right label
    y_lower = compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar)

    # Compute upper bound on wrong label
    y_upper = compute_upper_bound_bernoulli(l_max, table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pprime)
    
    return np.argmax(y_lower < y_upper) - 1

def certify_community(pstar, pprime, pnoise, l_max = 10):
    #pnoise is an array
    # For every possible separation of lmax into C communities
    # Compute ck 
    # Sort ck
    # Select Rgions such that pstar stay superior
    # COmpute lower bound
    # Compute upper bound with pprime
    # 


    raise NotImplementedError