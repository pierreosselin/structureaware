import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

from communityaware.models import GCN
from communityaware.perturb import perturb_graph
from communityaware.utils import load_dataset, load_model, make_noise_grid


def votes_node_classification(model, alpha_pair, dataset, repeats=10000, batch_size=50, device='cpu'):

    # check input
    if repeats % batch_size != 0:
        raise ValueError(f'Batch size should divide repeats. {batch_size=} {repeats=}')

    with torch.no_grad():
        votes = torch.zeros((len(dataset.testset_labels), dataset.num_classes), dtype=torch.long).to(device)
        pbar = tqdm(total=repeats)
        for _ in range(repeats // batch_size):
            # repeat and perturb the graph batch_size times
            pbar.set_description(f'Perturbing graph.')
            noise = dataset.make_noise_matrix(*alpha_pair)
            perturbed_graphs = perturb_graph(dataset.data, noise, repeats=batch_size, batch_size=batch_size, device=device)
            batch = list(perturbed_graphs)[0].to(device)

            # predict the labels of the perturbed graphs
            pbar.set_description(f'Predicting labels.')
            output = model(batch.x, batch.edge_index, batch.batch)
            output, _ = to_dense_batch(output, batch.batch) # output is a tensor of shape (batch_size, num_nodes, num_classes)
            output = output[:, dataset.data.test_mask, :] # output is a tensor of shape (batch_size, num_test_nodes, num_classes)
            output = output.argmax(dim=2).long() # output is a tensor of shape (batch_size, num_test_nodes) (value at i,j is the predicted class of node j in batch i)

            # thse next three lines make a tensor output of size (num_test_nodes, num_classes) where the value at i,j is the number of votes for class j for node i.
            target = torch.zeros((dataset.num_classes, output.size(1)), dtype=torch.long).to(device)
            src = torch.ones_like(output, dtype=torch.long).to(device)
            output = target.scatter_add_(0, output, src).T

            votes += output
            pbar.update(batch_size)
        return votes.cpu()

def votes_graph_classification(model, alpha_pair, dataset, repeats=10000, batch_size=32, device='cpu'):
    with torch.no_grad():
        votes = torch.zeros((len(dataset.testset_labels), dataset.num_classes), dtype=torch.long).to(device)
        test_loader = dataset.dataloader('test', batch_size=1)
        pbar = tqdm(enumerate(test_loader), leave=False, total=len(test_loader))
        for i, graph in pbar:
            pbar.set_description(f'Perturbing graph.')
            noise = torch.tensor(dataset.make_noise_matrix(graph, *alpha_pair))
            perturbed_graphs = perturb_graph(graph, noise, repeats=repeats, batch_size=batch_size, device=device)
            pbar.set_description(f'Predicting labels.')
            for batch in perturbed_graphs:
                batch = batch.to(device)
                if dataset.num_classes > 2:
                    predictions = model(batch).argmax(axis=1)
                    votes[i] += torch.bincount(predictions, minlength=dataset.num_classes)
                else:
                    predictions = model(batch) > 0
                    votes[i] += torch.bincount(predictions.long().squeeze(), minlength=2)
    return votes.cpu()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device

    data_path = 'data'
    model_path = join('output', config['data']['name'], 'weights')
    votes_path = join('output', config['data']['name'], 'votes')

    # load noise values
    noise_values = make_noise_grid(config)

    # load data
    dataset = load_dataset(config)
    dataset_name = config['data']['name'].lower()

    # determine if the dataset is graph or node classification
    graph_classification_task = True if dataset_name in ['synthetic', 'hiv'] else False

    # determine if we should use positional_encoding
    use_positional_encoding = True if dataset_name == 'synthetic' else False

    # Load config parameters
    model = load_model(config).to(device)
    model = model.to(device)
    model.load_state_dict(torch.load(join(model_path, 'weights.pt')))
    model.eval()

    for noise in tqdm(noise_values, desc='Loop over values of noise.'):
        if graph_classification_task:
            votes = votes_graph_classification(model, noise, dataset, config['voting']['repeats'], config['voting']['batch_size'], device)
        else:
            votes = votes_node_classification(model, noise, dataset, config['voting']['repeats'], config['voting']['batch_size'], device)
        assert votes.sum(1).unique().item() == config['voting']['repeats'] # this also implicitly checks unique is size 1.
        os.makedirs(votes_path, exist_ok=True)
        torch.save(votes, join(votes_path, '_'.join(map(str, np.round(noise, 8))))) # round(, 8) is to get rid of floating point errors. E.g. 0.30000000000000004 -> 0.3
