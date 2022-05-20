import os
from itertools import product

import numpy as np

from communityaware.data import HIV, CoraML, Synthetic


def mask_other_gpus(gpu_number):
    """Mask all other GPUs than the one specified."""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)


def load_dataset(config):
    config = config['data']
    if config['name'] == 'synthetic':
        dataset = Synthetic('data', config['graphs_per_class'], config['size_of_community'], config['number_of_communities'], config['split_proportions'])
    elif config['name'] == 'hiv':
        dataset = HIV('data', config['min_required_edge_flips'], config['split_proportions'])
    elif config['name'] == 'cora_ml':
        dataset = CoraML('data', config['training_nodes_per_class'])
    else:
        raise ValueError('Dataset {} not supported'.format(dataset))
    return dataset

def make_noise_grid(config):
    P_min = map(float, config['noise']['P_min'])
    P_max = map(float, config['noise']['P_max'])
    P_step = map(float, config['noise']['P_step'])
    ranges = [np.arange(p_min, p_max + 10e-8 + p_step) for (p_min, p_max, p_step) in zip(P_min, P_max, P_step)]
    return list(product(*ranges))

def make_radius_grid(config):
    R_max = map(int, config['radius']['R_max'])
    ranges = [np.arange(1, r_max+1) for r_max in R_max]
    return [np.array(i) for i in product(*ranges)]