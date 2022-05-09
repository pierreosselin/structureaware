import argparse
import torch
import yaml
from communityaware.cert.community import compute_certificate
from communityaware.data import Synthetic
from statsmodels.stats.proportion import proportion_confint, binom_test
from os.path import join
import numpy as np
from tqdm import tqdm
from itertools import product

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='synthetic')
args = parser.parse_args()

# load parameters
config = yaml.safe_load(open(f'config/{args.config}.yaml'))
vote_path = join('output', config['dataset'], 'votes')
certificate_path = join('output', config['dataset'], 'certificates')

# load test set
dataset = Synthetic('data')
test_set = dataset.dataloader('test', batch_size=1).dataset

# load votes
alpha_min = config['certification']['alpha_parameter_min']
alpha_max = config['certification']['alpha_parameter_max']
alpha_step = config['certification']['alpha_parameter_step']
alphas = np.arange(alpha_min, alpha_max + 10e-8, alpha_step) # 10e-8 makes the arange inclusive when alpha_step divides alpha_max.  
alpha_pairs = list(product(alphas, alphas)) # one for each p
r_max = config['certification']['r_max']
confidence_level = config['certification']['confidence_level']

# compute certificate.
certificates = {} # maps alpha_pair to certificate grid 
for alpha_pair in tqdm(alpha_pairs, desc="Loop over values of alpha.", total=len(alphas)**2):
    vote_path = join('output', config['dataset'], 'votes')
    votes = torch.load(join(vote_path, '_'.join(map(str, np.round(alpha_pair, 8))))).numpy()

    smoothed_prediction =  votes.argmax(1)
    p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * confidence_level, method="beta")[0]
    abstain = np.array([binom_test(na, total, 0.5) for na, total in zip(votes.max(1), votes.sum(1))]) >= confidence_level # abstain_i is True if smoothed classifier abstains

    certificate_grid = np.zeros((len(test_set), r_max, r_max)) # (i, j, k) entry says if sample i is robust to j and k. 
    smoothed_accuracy = np.zeros((len(test_set),)) # ith entry says if smoothed classifier predicts ith sample correctly. 
    for sample_idx in range(len(test_set)):

        # check if certify abstained
        if abstain[sample_idx]:
            continue

        # check if smoothed prediction is correct or not
        if smoothed_prediction[sample_idx] != test_set[sample_idx].y.item():
            continue

        # computer certificate grid
        for inner_idx, inner in enumerate(range(r_max)):
            for outer_idx, outer in enumerate(range(r_max)):
                if inner + outer == 0.0:
                    certificate_grid[sample_idx, inner_idx, outer_idx] = 1.0 # has already been certifed correct and we know we didnt abstain. 
                else:
                    certificate_grid[sample_idx, inner_idx, outer_idx] = compute_certificate(R=np.array((inner, outer)), P=np.array(alpha_pair), p_A=p_A[sample_idx])
    
    certificates[alpha_pair] = certificate_grid

torch.save(certificates, join('output', config['dataset'], 'certificates'))