import os
from itertools import product

import numpy as np
import scipy.sparse as sp

from communityaware.data import HIV, Synthetic
from communityaware.models import GCN, SpectrumNet


def mask_other_gpus(gpu_number):
    """Mask all other GPUs than the one specified."""
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_number)


def load_dataset(config):
    config = config['data']
    if config['name'] == 'synthetic':
        dataset = Synthetic('data', config['graphs_per_class'], config['size_of_community'], config['number_of_communities'], config['split_proportions'])
    elif config['name'] == 'hiv':
        dataset = HIV('data', config['min_required_edge_flips'], config['split_proportions'])
    else:
        raise ValueError('Dataset {} not supported'.format(dataset))
    return dataset

def load_model(config):
    dataset_name = config['data']['name'].lower()
    graph_classification_task = True if dataset_name in ['synthetic', 'hiv'] else False
    if config['model']['architecture'].lower() == 'spectrumnet':
        return SpectrumNet(config['model']['number_of_eigenvalues'], config['model']['hidden_channels'], config['data']['num_classes'], config['model']['dropout'])
    elif config['model']['architecture'].lower() == 'gcn':
        use_positional_encoding = True if dataset_name == 'synthetic' else False
        return GCN(config['data']['num_features'], config['model']['hidden_channels'], config['data']['num_classes'], config['model']['dropout'], pooling=graph_classification_task, use_positional_encoding=use_positional_encoding)


def make_noise_grid(config):
    P_min = list(map(float, config['noise']['P_min']))
    P_max = list(map(float, config['noise']['P_max']))
    P_step = list(map(float, config['noise']['P_step']))
    ranges = [inclusive_arange(p_min, p_max + 10e-8, p_step) for (p_min, p_max, p_step) in zip(P_min, P_max, P_step)]
    return list(product(*ranges))

def inclusive_arange(x_min, x_max, step):
    if np.isclose(x_min, x_max):
        return np.array([x_min,])
    else:
        return np.arange(x_min, x_max + 10e-8, step)

def make_radius_grid(config):
    R_max = map(int, config['radius']['R_max'])
    ranges = [np.arange(0, r_max+1) for r_max in R_max]
    return [np.array(i) for i in product(*ranges)]
