import argparse
import yaml
from module.generate_data import generate_synthetic_data
from module.utils import er_parameter_from_sbm

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    er_nodes = sum(config['list_blocks'])
    er_parameter = er_parameter_from_sbm(config['list_blocks'],  config['block_probs'])
    generate_synthetic_data(config['n_data_per_class'], config['list_blocks'], config['block_probs'], er_nodes, er_parameter, config['save_path'], config['data_split'])