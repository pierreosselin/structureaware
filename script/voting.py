from communityaware.utils import mask_other_gpus
mask_other_gpus(0)

import argparse
import torch
import torch.nn.functional as F
import yaml
from communityaware.models import GCN_Classification
from communityaware.perturb import sample_perturbed_graphs_with_sbm_noise
from communityaware.data import Synthetic
from torch_geometric.data import DataLoader
from tqdm import tqdm
from os.path import join
import os
import numpy as np
from communityaware.utils import er_parameter_from_sbm

def vote(data_path, model_path, votes_path, hidden_channels, sbm_parameters, alphas):
    """_summary_

    Args:
        data_path (_type_): _description_
        weight_path (_type_): _description_
        hidden_channels (_type_): _description_
        device (_type_): device for the model 
        sample_eval (_type_): _description_
        batch_size (_type_): _description_
        vote_path (_type_): _description_
        sbm_parameters (_type_): _description_
        parameter_list (_type_): _description_
    """

    # Load dataset
    dataset = Synthetic(data_path)
    test_loader = dataset.dataloader('test', batch_size=1)
    
    # Load config parameters
    model = GCN_Classification(hidden_channels=hidden_channels, num_features=dataset.num_features, num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(join(model_path, "weights.pt")))
    model.eval()
    
    for alpha in tqdm(alphas, desc="Loop over values of alpha."):
        votes = torch.zeros((len(test_loader), dataset.num_classes), dtype=torch.long)
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader.dataset), leave=False)
            for i, graph in pbar:
                pbar.set_description(f'Processing graph ({i}/{len(test_loader)}) (perturbing)')
                sizes = graph.sizes
                p = alpha * sbm_parameters
                repeats = 10000
                perturbed_graphs = sample_perturbed_graphs_with_sbm_noise(graph, sizes, p, repeats)
                pbar.set_description(f'Processing graph ({i}/{len(test_loader)}) (predicting)')
                for batch in perturbed_graphs:
                    predictions = model(batch.x, batch.edge_index, batch.batch).argmax(axis=1)
                    votes[i] += torch.bincount(predictions, minlength=2)
                pass
    
        os.makedirs(votes_path, exist_ok=True)
        torch.save(votes, join(votes_path, str(round(alpha, 8)))) # round is to get rid of floating point errors. E.g. 0.30000000000000004 -> 0.3

if __name__ == '__main__':

    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic_community')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    data_path = 'data'
    model_path = join('output', config['dataset'], 'weights')
    votes_path = join('output', config['dataset'], 'votes')
    
    alpha_min = config['certification']['alpha_parameter_min']
    alpha_max = config['certification']['alpha_parameter_max']
    alpha_step = config['certification']['alpha_parameter_step']
    alphas = np.arange(alpha_min, alpha_max + 10e-8, alpha_step) # 10e-8 makes the arange inclusive when alpha_step divides alpha_max.  

    vote(data_path=data_path,
         model_path=model_path,
         votes_path=votes_path,
         hidden_channels=config['optimisation']['hidden_channels'],
         sbm_parameters=np.array(config['sbm_parameters']),
         alphas=alphas)