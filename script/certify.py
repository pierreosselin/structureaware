import argparse
import torch
import yaml
from module.cert.bernoulli import max_bernoulli_radius
from statsmodels.stats.proportion import proportion_confint
from os.path import join
import numpy as np
import os

def certify(vote_path, noise, certificate_path, alpha=0.99):
    votes = torch.load(join(vote_path, 'votes')).numpy()
    p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * alpha, method="beta")[0]
    radius = []
    for sample_p_A in p_A:
        radius.append(max_bernoulli_radius(sample_p_A, noise))
    os.makedirs(certificate_path)
    np.save(join(certificate_path, 'certificate'), radius)


if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    vote_path = join('output', config['dataset'], 'votes')
    certificate_path = join('output', config['dataset'], 'certificates')
    noise = config['certification']['parameter_list'][-1]
    certify(vote_path, noise, certificate_path)
