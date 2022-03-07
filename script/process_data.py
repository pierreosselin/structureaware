import argparse
import shutil
from pathlib import Path

import torch
import yaml
from module.clustering import process_clustering
from module.generate_data import generate_data
from tqdm import tqdm


def preprocessing(config):
    """Preprocessing routine saving the data and creating clusters

    Args:
        config ([Dict]): Configuration parameters for the experiments
    """
    name_dataset = config["dataset"]
    n_data = config["n_data_per_class"]
    list_blocks = config["list_blocks"]
    p = config["block_probs"]
    er_param = config["er_param"]
    save_path = config["vote_path"]
    prop_train_test = config["prop_train_test"]
    override = config["override_weight"]

    ###Create folders for output, overide if necessary
    path_creation = Path(f"./{save_path}")
    if path_creation.exists():
        if override:
            print("The dataset folder already exists and the override option is on, dataset deleted.")
            shutil.rmtree(path_creation)
        else:
            raise Exception("The dataset folder already exists and the override option is off, dataset generation aborted.")
    path_creation.mkdir(parents=True, exist_ok=True)

    l_data_train, l_data_test = generate_data(name_dataset=name_dataset, n_data=n_data, 
                            list_blocks=list_blocks, p=p, er_param=er_param, save_path=save_path, prop_train_test=prop_train_test)

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