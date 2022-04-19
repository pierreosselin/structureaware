import argparse
import torch
import yaml
from communityaware.cert.community import compute_certificate
from communityaware.data import Synthetic
from statsmodels.stats.proportion import proportion_confint
from os.path import join
import numpy as np
import os
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='synthetic_community')
args = parser.parse_args()

# load parameters
config = yaml.safe_load(open(f'config/{args.config}.yaml'))
vote_path = join('output', config['dataset'], 'votes')
certificate_path = join('output', config['dataset'], 'certificates')

# load test set
dataset = Synthetic('data')
test_set = dataset.dataloader('test', batch_size=1).dataset

# load votes
noise = 0.001
alpha = 0.99
votes = torch.load(join(vote_path, str(round(noise, 8)))).numpy()
votes = np.ones_like(votes)*500
p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * alpha, method="beta")[0]

# compute certificate.
n = 5
certs = np.zeros((n, n))
for sample_p_A, graph in zip(p_A, test_set):
    for inner in range(n):
        for outer in range(n):
            if graph.y.item() == 1:
                P = np.array([(noise)])
                R = np.array((inner+outer,))
            else:
                R = np.array((inner, outer))
                P = np.array((0.1, noise))
            certificate = compute_certificate(R, P, sample_p_A)
            certs[inner, outer] = certificate

    if graph.y.item() == 0:
        print(certs)
        print()

