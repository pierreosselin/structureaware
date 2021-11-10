import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
from module.cert_copy import certify_bernoulli


def certify(config):
    pstar, pprime, pnoise  = 0.8, 0.2, 0.1
    certify_bernoulli(pstar, pprime, pnoise)
    return 0

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    certify(config)