import argparse
import torch
import yaml
from communityaware.cert.community import inner_outer_certificate
from statsmodels.stats.proportion import proportion_confint
from os.path import join
import numpy as np
import os
from tqdm import tqdm

def certify(vote_path, noise, sbm_parameters, certificate_path, alpha=0.99):
    votes = torch.load(join(vote_path, str(round(noise, 8)))).numpy()
    noise = noise * sbm_parameters
    p_A = proportion_confint(votes.max(1), votes.sum(1), alpha=2 * alpha, method="beta")[0]
    results = []
    # for i, sample_p_A in tqdm(enumerate(p_A)):
    #     certified = True
    #     inner = 1
    #     outer = 0
    #     while certified:
    #         certificate = inner_outer_certificate(inner, outer, noise, sample_p_A)
    #         if not certificate:
    #             if inner == 0: # no certificates for the largest value of output
    #                 break
    #             else:
    #                 outer += 1
    #                 inner = 0
    #         else:
    #             inner += 1
    #             results.append((i, inner, outer))
    #             print(i, inner, outer)

    certs = np.zeros((6, 6))
    for i, sample_p_A in tqdm(enumerate(p_A)):
        for inner in range(6):
            for outer in range(6):
                certificate = inner_outer_certificate(inner, outer, noise, sample_p_A)
                certs[inner, outer] = certificate
        print(certs)
        print()

    print(results)
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
    noise = 0.02
    certify(vote_path, noise, np.array(config['sbm_parameters']), certificate_path)