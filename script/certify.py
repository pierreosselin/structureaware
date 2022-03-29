import argparse
import torch
import yaml
from module.cert import certify_bernoulli


def certify(vote_path):
    votes = torch.load(path_votes + "votes"))
    pstar, pprime, pnoise  = 0.99, 0.01, 0.1
    l = certify_bernoulli(pstar, pprime, pnoise, l_max = 5)
    print(l)
    return l

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    certify(config.vote_path)