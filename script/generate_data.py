import argparse
import yaml
from module.generate_data import generate_synthetic_data
from module.utils import er_parameter_from_sbm
from os.path import join

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic')
    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    er_nodes = sum(config['community_sizes'])
    er_parameter = er_parameter_from_sbm(config['community_sizes'],  config['sbm_parameters'])

    generate_synthetic_data(config['graphs_per_class'], config['community_sizes'], config['sbm_parameters'], er_nodes, er_parameter, 'data', config['data_split'])