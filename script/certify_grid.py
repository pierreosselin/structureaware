import argparse
from itertools import product
from os.path import join

import numpy as np
import torch
import yaml
from statsmodels.stats.proportion import binom_test, proportion_confint
from tqdm import tqdm, trange

from communityaware.cert import compute_certificate_gmpy as compute_certificate
from communityaware.data import Synthetic
from communityaware.utils import (load_dataset, make_noise_grid,
                                  make_radius_grid)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='synthetic')
args = parser.parse_args()

# load parameters
config = yaml.safe_load(open(f'config/{args.config}.yaml'))

# load test set labels
dataset = load_dataset(config)
test_set_labels = dataset.testset_labels.numpy()

# load noise values 
noise_values = make_noise_grid(config)
radius = make_radius_grid(config)
confidence_level = config['confidence_level']

# compute certificate.
certificates = {} # maps alpha_pair to certificate grid 
for noise in tqdm(noise_values, desc="Loop over values of noise."):

    # load votes
    vote_path = join('output', config['data']['name'], 'votes')
    votes = torch.load(join(vote_path, '_'.join(map(str, np.round(noise, 8))))).numpy()

    smoothed_prediction =  votes.argmax(1)
    p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * confidence_level, method="beta")[0]
    abstain = np.array([binom_test(na, total, 0.5) for na, total in zip(votes.max(1), votes.sum(1))]) >= confidence_level # abstain_i is True if smoothed classifier abstains

    certificate_grid = np.zeros((len(votes), *config['radius']['R_max'])) # (i, r_1, r_2,...) entry says if sample i is robust at radius r_1, r_2,...
    correctly_classified = np.zeros((len(votes),)) # ith entry says if smoothed classifier predicts ith sample correctly. 
    for sample_idx in trange(len(votes), leave=False, desc="Loop over test set."):

        # check if certify abstained
        if abstain[sample_idx]:
            continue

        # check if smoothed prediction is correct or not
        if smoothed_prediction[sample_idx] == test_set_labels[sample_idx]:
            correctly_classified[sample_idx] = True
        else:
            correctly_classified[sample_idx] = False

        # computer certificate grid
        for R in radius:
            idx = (sample_idx, *(R-1)) 
            if np.sum(R) == 0:
                certificate_grid[idx] = 1.0 # has already been certifed correct and we know we didnt abstain. 
            else:
                certificate_grid[idx] = float(compute_certificate(R=R, P=noise, p_A=p_A[sample_idx]))
    
    certificates[noise] = {'certificate_grid': certificate_grid,
                           'correctly_classified': correctly_classified,
                           'abstain': abstain}

torch.save(certificates, join('output', config['data']['name'], 'certificates'))
