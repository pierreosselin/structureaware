from module.data_processing import generate_data
import torch
import argparse
import yaml
from tqdm import tqdm
from clustering import process_clustering


def preprocessing(config):
    """Preprocessing routine saving the data and creating clusters

    Args:
        config ([Dict]): Configuration parameters for the experiments
    """

    l_data_train, l_data_test = generate_data(config)

    # Apply clustering (only for real-world data)
    if config["dataset"] != "synthetic":
        if config["clustering"]["apply_clustering"]:
            param_cluster = config["clustering"]["clustering_parameter"]
            digits = config["clustering"]["digits"]
            for datum in tqdm(l_data_train + l_data_test, desc='Applying Clustering to the dataset'):
                community_prob, node_community, community_node, community_size = process_clustering(datum, param_cluster, digits)
                datum.node_community = torch.tensor(node_community)
                datum.community_node = community_node
                datum.community_size = torch.tensor(community_size)
                datum.community_prob = torch.tensor(community_prob)

    # Save data
    torch.save(l_data_train, config["save_path"] + "dataset_train")
    torch.save(l_data_test, config["save_path"] + "dataset_test")
    return 0

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    preprocessing(config)