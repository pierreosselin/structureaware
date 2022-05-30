from math import comb

import gmpy2
import numpy as np

from communityaware.cert.isotropic import compute_regions


def isotropic_compute_regions_test():
    r, p = 4, 0.25
    likelihood_ratios, region_probability = compute_regions(gmpy2.mpz(r), gmpy2.mpfr(p))
    expected_likelihood_ratios = np.array([1/81, 1/9, 1, 9, 81]) # \eta^{R_4}, ..., \eta^{R_0}
    expected_region_probabilities = np.array([
        comb(4, 0) * p ** 0 * (1-p) ** 4, # P(Phi(x)) \in R_4
        comb(4, 1) * p ** 1 * (1-p) ** 3, # P(Phi(x)) \in R_3
        comb(4, 2) * p ** 2 * (1-p) ** 2, # P(Phi(x)) \in R_2
        comb(4, 3) * p ** 3 * (1-p) ** 1, # P(Phi(x)) \in R_1
        comb(4, 4) * p ** 4 * (1-p) ** 0  # P(Phi(x)) \in R_0
    ])
    assert np.allclose(likelihood_ratios.astype(float), expected_likelihood_ratios)
    assert np.allclose(region_probability.astype(float), expected_region_probabilities)
