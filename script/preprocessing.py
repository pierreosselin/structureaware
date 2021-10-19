from module.data_processing import generate_data
import argparse
import yaml

def preprocessing(config):
    """Preprocessing routine saving the data and creating clusters

    Args:
        config ([Dict]): Configuration parameters for the experiments
    """

    generate_data(config)
    return 0

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    preprocessing(config)