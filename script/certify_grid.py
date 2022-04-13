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

def certify(vote_path, noise, sbm_parameters, certificate_path, alpha=0.99):

    data_path = 'data'
    dataset = Synthetic(data_path)
    test_set = dataset.dataloader('test', batch_size=1).dataset

    votes = torch.load(join(vote_path, str(round(noise, 8)))).numpy()
    p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * alpha, method="beta")[0]

    n = 20
    certs = np.zeros((n, n))
    for sample_p_A, graph in zip(p_A, test_set):
        for inner in range(n):
            for outer in range(n):
                if graph.y.item() == 1:
                    P = np.array([(noise * sbm_parameters[0][0])])
                    R = np.array((inner+outer,))
                else:
                    R = np.array((inner, outer))
                    P = np.array((noise * sbm_parameters[0][0], noise * sbm_parameters[0][1]))
                certificate = compute_certificate(R, P, sample_p_A)
                certs[inner, outer] = certificate
        if np.any(certs < 0.5):
            pass

    #os.makedirs(certificate_path)
    #np.save(join(certificate_path, 'certificate'), results)


if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic_community')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    vote_path = join('output', config['dataset'], 'votes')
    certificate_path = join('output', config['dataset'], 'certificates')
    noise = 0.05
    certify(vote_path, noise, np.array(config['sbm_parameters']), certificate_path)
