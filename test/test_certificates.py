from communityaware.cert import pre_compute_regions_bernoulli, compute_lower_bound_bernoulli, compute_upper_bound_bernoulli
import numpy as np

def test_lower_bound():
    pstar, pprime, pnoise, l_max  = 0.99, 0.01, 0.1, 5
    l_max += 1
    #Precompute table of probabilities
    table_regions = pre_compute_regions_bernoulli(l_max, pnoise)
    reversed_table_regions = table_regions[:,::-1]

    # Compute the cumulative sums of the probability that X are in curly H and curly Q
    table_regions_cum_sum = np.cumsum(table_regions,axis=1)
    reversed_table_regions_cum_sum = np.cumsum(reversed_table_regions,axis=1)
    y_lower = compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar)
    
    assert (y_lower - np.array([0.99, 0.91, 0.19, 0.19, 0.046, 0.0084]) < 0.001).all()

def test_symmetry_binary():

    pstar, pprime, pnoise, l_max  = 0.8, 0.2, 0.15, 9
    l_max += 1
    #Precompute table of probabilities
    table_regions = pre_compute_regions_bernoulli(l_max, pnoise)
    reversed_table_regions = table_regions[:,::-1]

    # Compute the cumulative sums of the probability that X are in curly H and curly Q
    table_regions_cum_sum = np.cumsum(table_regions,axis=1)
    reversed_table_regions_cum_sum = np.cumsum(reversed_table_regions,axis=1)
    y_lower = compute_lower_bound_bernoulli(l_max, table_regions, reversed_table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pstar)
    y_upper = compute_upper_bound_bernoulli(l_max, table_regions, table_regions_cum_sum, reversed_table_regions_cum_sum, pprime)
    
    assert (1 - y_lower-y_upper < 0.001).all()