import numpy as np
import gmpy2
from tqdm.autonotebook import tqdm
from statsmodels.stats.proportion import proportion_confint
from itertools import product
from collections import defaultdict
from scipy.stats import binom
import scipy.special as sp
import time

def load_certificate(experiment):
    if experiment == "bernoulli":
        return binary_certificate_grid

    if experiment == "SBM":
        return binary_certificate_grid_community


def compute_rho_for_many(regions, p_emps, is_sorted=True, reverse=False):
    """
    Compute the worst-case probability of the adversary for many p_emps at once.

    Parameters
    ----------
    regions : array-like, [?, 3]
        Regions of constant probability under p_x and p_y,
    p_emps : array-like [?]
        Empirical probabilities per node.
    is_sorted : bool
        Whether the regions are sorted, e.g. regions from `regions_binary` are automatically sorted.
    reverse : bool
        Whether to consider the sorting in reverse order.

    Returns
    -------
    p_adver : array-like [?]
        The worst-case probability of the adversary.
    """
    sort_direction = -1 if reverse else 1
    if not is_sorted:
        o = regions[:, 2].argsort()[::-sort_direction]
        regions = regions[o]
    else:
        regions = regions[::sort_direction]

    # add one empty region to have easier indexing
    regions = np.row_stack(([0, 0, 0], regions))

    #print("Here are the sorted regions")
    #print(regions)

    cumsum = np.cumsum(regions[:, :2], 0)
    h_stars = (cumsum[:, 0][:, None] >= p_emps).argmax(0)
    h_stars[h_stars > 0] -= 1

    h_star_cumsums = cumsum[h_stars]

    acc_p_clean = h_star_cumsums[:, 0]
    acc_p_adver = h_star_cumsums[:, 1]

    #print("Proba with missing element")
    #print(acc_p_adver)
    # add the missing probability for those that need it
    flt = (p_emps - acc_p_clean > 0) & (h_stars + 1 < len(regions))
    addition = (p_emps[flt] - acc_p_clean[flt]) * \
        regions[h_stars[flt] + 1, 1] / regions[h_stars[flt] + 1, 0]
    acc_p_adver[flt] += addition

    acc_p_adver[h_stars == -1] = 0
    #print("final probability")
    #print(acc_p_adver)
    return acc_p_adver.astype('float')

def min_p_emp_for_radius_1(pf_plus, pf_minus, which, lower=0.5, verbose=False):
    """
    Find the smallest p_emp for which we can certify a radius of 1 using bisection.


    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    which : string
        'ra': find min_{p_emp, ra=1, rd=0}
        'rd': find min_{p_emp, rd=1, ra=0}
    lower : float
        A lower bound on the minimum p_emp.
    verbose : bool
        Verbosity.

    Returns
    -------
    min_p_emp : float
        The minimum p_emp.
    """
    initial_lower = lower
    upper = 1
    p_emp = 0

    if which == 'ra':
        ra = 1
        rd = 0
    elif which == 'rd':
        ra = 0
        rd = 1
    else:
        raise ValueError('which can only be "ra" or "rd"')

    while lower < upper:
        p_emp = lower + (upper - lower) / 2

        cur_rho = compute_rho(regions_binary(
            ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus), p_emp)
        if verbose:
            print(p_emp, float(cur_rho))

        if cur_rho < 0.5:
            if lower == p_emp:
                break
            lower = p_emp
        elif abs(cur_rho - 0.5) < 1e-10:
            break
        else:
            upper = p_emp

    if p_emp <= initial_lower:
        if verbose:
            print(
                'p_emp <= initial_lower, restarting the search with a smaller lower bound')
        return min_p_emp_for_radius_1(
            pf_plus=pf_plus, pf_minus=pf_minus, which=which, lower=lower*0.5, verbose=verbose)

    return p_emp

def max_radius_for_p_emp(pf_plus, pf_minus, p_emp, which, upper=100, verbose=False):
    """
    Find the maximum radius we can certify individually (either ra or rd) using bisection.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emp : float
        Empirical probability of the majority class
    which : string
        'ra': find max_{ra, rd=0}
        'rd': find max_{rd, ra=0}
    upper : int
        An upper bound on the maximum radius
    verbose : bool
        Verbosity.

    Returns
    -------
    max_r : int
        The maximum certified radius s.t. the probability of the adversary is above 0.5.

    """
    initial_upper = upper
    lower = 1
    r = 1

    while lower < upper:
        r = lower + (upper - lower) // 2
        if which == 'ra':
            ra = r
            rd = 0
        elif which == 'rd':
            ra = 0
            rd = r
        else:
            raise ValueError('which can only be "ra" or "rd"')

        cur_rho = compute_rho(regions_binary(
            ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus), p_emp)
        if verbose:
            print(r, float(cur_rho))

        if cur_rho > 0.5:
            if lower == r:
                break
            lower = r
        else:
            upper = r

    if r == initial_upper or r == initial_upper - 1:
        if verbose:
            print('r = upper, restart the search with a larger upper bound')
        return max_radius_for_p_emp(pf_plus=pf_plus, pf_minus=pf_minus,
                                    p_emp=p_emp, which=which, upper=2*upper, verbose=verbose)

    return r


def regions_binary(ra, rd, pf_plus, pf_minus, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    ra: int
        Number of ones y has added to x
    rd : int
        Number of ones y has deleted from x
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    regions: array-like, [None, 3]
        Regions of constant probability under px and px_tilde,
    """

    pf_plus, pf_minus = gmpy2.mpfr(pf_plus), gmpy2.mpfr(pf_minus)
    with gmpy2.context(precision=precision):
        if pf_plus == 0:
            px = pf_minus ** rd
            px_tilde = pf_minus ** ra

            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0]
                             ])

        if pf_minus == 0:
            px = pf_plus ** ra
            px_tilde = pf_plus ** rd
            return np.array([[1 - px, 0, float('inf')],
                             [px, px_tilde, px / px_tilde],
                             [0, 1 - px_tilde, 0],
                             ])
        max_q = ra + rd
        i_vec = np.arange(0, max_q + 1)

        T = ra * ((pf_plus / (1 - pf_plus)) ** i_vec) + \
            rd * ((pf_minus / (1 - pf_minus)) ** i_vec)

        ratio = np.zeros_like(T)
        px = np.zeros_like(T)
        px[0] = 1

        for q in range(0, max_q + 1):
            ratio[q] = (pf_plus/(1-pf_minus)) ** (q - rd) * \
                (pf_minus/(1-pf_plus)) ** (q - ra)
            if q == 0:
                continue

            for i in range(1, q + 1):
                px[q] = px[q] + ((-1) ** (i + 1)) * T[i] * px[q - i]
            px[q] = px[q] / q

        scale = ((1-pf_plus) ** ra) * ((1-pf_minus) ** rd)

        px = px * scale

        regions = np.column_stack((px, px / ratio, ratio))
        if pf_plus+pf_minus > 1:
            # reverse the order to maintain decreasing sorting
            regions = regions[::-1]
        return regions

def binary_certificate_grid(parameter, p_emps, reverse=False, regions=None, progress_bar=True, **kwargs):
    """
    Compute rho for all given p_emps and for all combinations of radii up to the maximum radii.

    Parameters
    ----------
    pf_plus : float, 0 <= p_plus <= 1
        The probability to flip a zero to a one
    pf_minus: float, 0 <= p_plus <= 1
        The probability to flip a one to a zero
    p_emps : array-like [n_nodes]
        Empirical probabilities per node.
    reverse : bool
        Whether to consider the sorting in reverse order.
    regions : dict
        A pre-computed dictionary of regions
    progress_bar : bool
        Whether to show a tqdm progress bar

    Returns
    -------
    radii : array-like, [n_nodes, max_ra, max_rd]
        Probabilities of the adversary. Node is certified if [:, :, :]>0.5
    regions : dict
        A pre-computed dictionary of regions
    max_ra : int
        Maximum certified addition radius
    max_rd : int
        Maximum certified deletion radius
    """
    if progress_bar:
        def bar(loop):
            return tqdm(loop)
    else:
        def bar(loop):
            return loop

    pf_plus = parameter
    pf_minus = parameter
    max_ra = kwargs["max_ra"]
    max_rd = kwargs["max_rd"]


    if regions is None:
        # compute the maximum possible ra and rd we can certify for the largest p_emp
        if max_ra is None or max_rd is None:
            max_p_emp = p_emps.max()
            max_ra = max_radius_for_p_emp(
                pf_plus=pf_plus, pf_minus=pf_minus, p_emp=max_p_emp, which='ra', upper=100)
            max_rd = max_radius_for_p_emp(
                pf_plus=pf_plus, pf_minus=pf_minus, p_emp=max_p_emp, which='rd', upper=100)
            min_p_emp = min(min_p_emp_for_radius_1(pf_plus, pf_minus, 'ra'),
                            min_p_emp_for_radius_1(pf_plus, pf_minus, 'rd'))

            print(f'max_ra={max_ra}, max_rd={max_rd}, min_p_emp={min_p_emp:.4f}')

        regions = {}
        print("Computing Regions...")
        for ra in bar(range(max_ra + 2)):
            for rd in range(max_rd + 2):
                regions[(ra, rd)] = regions_binary(
                    ra=ra, rd=rd, pf_plus=pf_plus, pf_minus=pf_minus)

    n_nodes = len(p_emps)
    arng = np.arange(n_nodes)
    radii = np.zeros((n_nodes, max_ra + 2, max_rd + 2))

    print("Computing Radii...")
    for (ra, rd), regions_ra_rd in bar(regions.items()):
        if ra + rd == 0:
            radii[arng, ra, rd] = 1
        else:
            radii[arng, ra, rd] = compute_rho_for_many(
                regions=regions_ra_rd, p_emps=p_emps, is_sorted=True, reverse=reverse)

    return radii, regions, max_ra, max_rd

def binary_certificate_grid_community(parameter, p_emps, reverse=False, regions=None, progress_bar=True, **kwargs):
    """
    Compute rho for all given p_emps and for all combinations of radii up to the maximum radii.

    Parameters
    ----------
    p_emps : array-like [n_graphs]
        Empirical probabilities per graph.
    reverse : bool
        Whether to consider the sorting in reverse order.
    regions : dict
        A pre-computed dictionary of regions
    progress_bar : bool
        Whether to show a tqdm progress bar

    Returns
    -------
    radii : array-like, [n_nodes, max_ra, max_rd]
        Probabilities of the adversary. Node is certified if [:, :, :]>0.5
    regions : dict
        A pre-computed dictionary of regions
    max_r : array like
        Maximum certified perturbation radius per community
    """

    if progress_bar:
        def bar(loop):
            return tqdm(loop)
    else:
        def bar(loop):
            return loop

    loader = kwargs["loader"]
    alpha = parameter
    undirected = kwargs["undirected"]
    max_l0 = kwargs["max_l0"]
    l0_list = list(range(1, max_l0+1))
    radii = np.zeros((len(loader), len(l0_list)))

    for i, data in bar(enumerate(loader)):
        print(f"Processed Graph: {i}")
        beta_vector = alpha*data.community_prob.cpu() ## Matrix of size CxC
        p_emp = p_emps[i]
        com_size = (np.outer(data.community_size, data.community_size) - np.diag(data.community_size)) / 2
        
        if undirected:
            beta_vector = beta_vector[np.tril_indices(beta_vector.shape[0], k = 0)]
            com_size = com_size[np.tril_indices(com_size.shape[0], k = 0)]
        else:
            beta_vector = beta_vector.flatten()
            com_size = com_size.flatten()
        
        beta_vector_unique = np.unique(beta_vector).astype(float)
        final_com_size = []
        for el in beta_vector_unique:
            if com_size.shape[0] == 1:
                final_com_size.append(com_size[0])
            else:
                final_com_size.append(com_size[beta_vector == el].sum())
        beta_vector = beta_vector_unique
        final_com_size = np.array(final_com_size)

        n_com = beta_vector.shape[0]
        ### Try a defaultdict
        d_ratio = defaultdict(int)
        d_bin = defaultdict(int)
        d_prod = defaultdict(int)
        #print(f"The number of community is {beta_vector.shape[0]}")
        for j, l0 in enumerate(l0_list):
            #print(f"Certification for {l0} l0 norm")
            list_disk = generate_disk(n_com, l0)
            list_disk = [el for el in list_disk if (np.array(el) <= final_com_size).all()]
            
            #print(f"Elapsed time for generating disk: {end_time - current_time}, length disk: {len(list_disk)}")
            l_min = []
            for el in list_disk:
                #print("Regions for element:", el)
                l_min.append(regions_binary_community(p_emp, el, beta_vector, d_ratio, d_bin, d_prod))
            #print(l_min)
            radii[i, j] = min(l_min)
            #print(f"Elapsed time for radii computation: {end_time - current_time}")

    return radii

def generate_disk(d, l0):
    """
    Generate elements of the disk if radius l0 and dimension d
    to change to integrate a max element per dimension
    """
    if d == 1:
        return [[l0]]

    if l0 == 0:
        return [[0 for i in range(d)]]

    return sum([[[i] + el for el in generate_disk(d-1, l0 - i)] for i in range(l0+1)], [])


def regions_binary_community(p_emp, r, p_com, d_ratio, d_bin, d_prod, precision=1000):
    """
    Construct (px, px_tilde, px/px_tilde) regions used to find the certified radius for binary data.

    Intuitively, pf_minus controls rd and pf_plus controls ra.

    Parameters
    ----------
    p_emps : float
        Empirical probability of the graph.
    r: array [K]
        Number of perturbation per pair of selected edge set (symmetry accounted)
    p_com : array [K]
        The probability to flip an edge between two communities
    precision: int
        Numerical precision for floating point calculations

    Returns
    -------
    p_adv:
        Probability of the adversary for the given ma   x ra
    """
    a_current = (1-p_com)/p_com
    a_current = np.array([gmpy2.mpfr(el) for el in a_current])
    with gmpy2.context(precision=precision):
        a = r
        #### Compute regions
        regions  = []
        list_permutation = product_flexible(a) #Size product of the maximal capacity

        for el in list_permutation:
            current_state = r - 2*el
            key = tuple(current_state)
            ratio = d_ratio[key]
            if not ratio:
                ratio = np.prod(a_current**(current_state))
                d_ratio[key] = ratio

            #px = np.prod(np.array([gmpy2.mpfr(get_binom(d_bin, a, b)*get_prod(i, a, b)    binom.pmf(el2, a[i], p_com[i])) for i, el2 in enumerate(el)]))
            px = np.prod(np.array([gmpy2.mpfr(get_binom(d_bin, el2, a[i]) * get_prod(d_prod, i, el2, a[i], p_com[i]) ) for i, el2 in enumerate(el)]))
            #print(f"px probability is {px}, for the following situation el : {el} | a {a} | pcom {p_com}")
            px_tilde = px/ratio
            regions.append([px, px_tilde, ratio])

        #print(f"Regions for radius {r}:")
        #print(regions)
        #print('Edge flipping probability:', p_com)

        #### Select first propability p of sorted regions in decreasing order
        sorted_regions = sorted(list(regions), key=lambda a: a[2], reverse=True)
        #print(f"Regions for radius {r}:")
        #print(sorted_regions)

        acc_p_clean = 0.0
        acc_p_adver = 0.0

        for i, (p_clean, p_adver, _) in enumerate(sorted_regions):
            # break early so the sums only reflect up to H*-1
            if acc_p_clean + p_clean >= p_emp:
                break
            if p_clean > 0:
                acc_p_clean += p_clean
                acc_p_adver += p_adver

        rho = acc_p_adver

        #print("Probability with missing element")
        #print(rho)
        
        # there is some probability left
        if p_emp - acc_p_clean > 0 and i < len(regions):
            addition = (p_emp - acc_p_clean) * (p_adver / p_clean)
            rho += addition

        #print("Probability without missing element")
        #print(rho)

        return rho