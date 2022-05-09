from communityaware.utils import mask_other_gpus
mask_other_gpus(1)

import argparse
import torch
import torch.nn.functional as F
import yaml
from communityaware.models import GCN_Classification
from communityaware.perturb import batch_perturbed_graph
from communityaware.data import Synthetic, HIV
from tqdm import tqdm
from os.path import join
import os
import numpy as np
from itertools import product


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='synthetic')
args = parser.parse_args()
config = yaml.safe_load(open(f'config/{args.config}.yaml'))

data_path = 'data'
model_path = join('output', config['dataset'], 'weights')
votes_path = join('output', config['dataset'], 'votes')

alpha_min = config['certification']['alpha_parameter_min']
alpha_max = config['certification']['alpha_parameter_max']
alpha_step = config['certification']['alpha_parameter_step']
alphas = np.arange(alpha_min, alpha_max + 10e-8, alpha_step) # 10e-8 makes the arange inclusive when alpha_step divides alpha_max.  
alpha_pairs = list(product(alphas, alphas)) # one for each p

# load data
if config['dataset'].lower() == 'synthetic':
    dataset = Synthetic('data')
elif config['dataset'].lower() == 'hiv':
    dataset = HIV('data', min_required_edge_flips=20)
test_loader = dataset.dataloader('test', batch_size=1)

# Load config parameters
model = GCN_Classification(hidden_channels=config['optimisation']['hidden_channels'], num_features=dataset.num_features, num_classes=dataset.num_classes)
model.load_state_dict(torch.load(join(model_path, "weights.pt")))
model.eval()

for alpha_pair in tqdm(alpha_pairs, desc="Loop over values of alpha.", total=len(alphas)**2):
    votes = torch.zeros((len(test_loader), dataset.num_classes), dtype=torch.long)
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader.dataset), leave=False, total=len(test_loader))
        for i, graph in pbar:
            pbar.set_description(f'Perturbing graph.')
            noise = dataset.make_noise_matrix(graph, *alpha_pair)
            perturbed_graphs = batch_perturbed_graph(graph, noise, repeats=10000)
            pbar.set_description(f'Predicting labels.')
            for batch in perturbed_graphs:
                predictions = model(batch.x, batch.edge_index, batch.batch).argmax(axis=1)
                votes[i] += torch.bincount(predictions, minlength=2)
            pass

    os.makedirs(votes_path, exist_ok=True)
    torch.save(votes, join(votes_path, '_'.join(map(str, np.round(alpha_pair, 8))))) # round(, 8) is to get rid of floating point errors. E.g. 0.30000000000000004 -> 0.3
