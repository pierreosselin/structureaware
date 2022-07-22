import argparse
import os
from functools import partial
from os.path import join
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch_geometric.data import InMemoryDataset

from communityaware.perturb import perturb_batch
from communityaware.utils import load_dataset, load_model


def sparsity_aware_noise_function(graph, p0, p1):
    noise = torch.ones((graph.num_nodes, graph.num_nodes)) * p1
    for i, j in graph.edge_index.T:
        noise[i.item(),j.item()] = p0
    noise.fill_diagonal_(0)
    return noise

# helper functions for graph classification
def train_epoch(model, dataset, criterion, optimiser, scheduler=None, noise_function=None, noise=None) -> tuple:
    #model: nn.Module, dataset: InMemoryDataset, criterion: _Loss, optimiser: Optimizer, scheduler: Optional[Callable]==None, noise_function: Optional[Callable]=None, noise: Tuple=None) -> tuple:
    model.train()
    train_loader = dataset.dataloader('train', config['training']['batch_size'])
    accuracy = torchmetrics.Accuracy(threshold=0.0)
    epoch_loss = torchmetrics.MeanMetric()
    for batch in train_loader:  # Iterate in batches over the training dataset.
        if noise_function is not None:
            batch = perturb_batch(batch, noise, noise_function)
        out = model(batch.x, batch.edge_index, batch.batch)  # Perform a single forward pass.
        loss = criterion(out, batch.y)  # Compute the loss.
        loss.mean().backward()  # Derive gradients.
        epoch_loss.update(loss.cpu())
        accuracy.update(out.squeeze().cpu(), batch.y.cpu())
        optimiser.step()  # Update parameters based on gradients.
        optimiser.zero_grad()  # Clear gradients.
    if scheduler is not None:
        scheduler.step()
    return epoch_loss.compute().item(), accuracy.compute().item()

def evaluate(model: nn.Module, dataset: InMemoryDataset, criterion: _Loss, split: str) -> tuple:
    with torch.no_grad():
        loader = dataset.dataloader(split, config['training']['batch_size'])
        model.eval()
        accuracy = torchmetrics.Accuracy(threshold=0.0)
        epoch_loss = torchmetrics.MeanMetric()
        for batch in loader:  # Iterate in batches over the training/test dataset.
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            epoch_loss.update(loss.cpu())
            accuracy.update(out.cpu().squeeze(), batch.y.cpu())
        return epoch_loss.compute().item(), accuracy.compute().item()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='mutag')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device

    # load data
    dataset = load_dataset(config)
    dataset.data.to(device)
    dataset_name = config['data']['name']

    # instanciate model
    model = load_model(config, num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)

     # instanciate optimiser and scheduler
    optimiser = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    if config['training'].get('decay', False):
        scheduler = StepLR(optimiser, step_size=50, gamma=0.5)
    else:
        scheduler = None

    # instanciate loss function
    if dataset.num_classes == 2:
        _criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion = lambda x, y: _criterion(x.squeeze(), y.float())
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # early stopping parameters
    best_model = None
    best_loss = np.inf

    # train with noise
    train_with_noise = config['training'].get('train_with_noise', None)
    noise_function = sparsity_aware_noise_function if train_with_noise else None
    p = config['training'].get('p', None)

    # training
    for epoch in range(config['training']['max_epochs']):
        train_loss, train_accuracy = train_epoch(model, dataset, criterion, optimiser, scheduler, sparsity_aware_noise_function, p)
        valid_loss, valid_accuracy = evaluate(model, dataset, criterion, 'valid')
        print(epoch, round(train_loss, 3), round(valid_loss, 3), round(train_accuracy, 2), round(valid_accuracy, 2))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model = model.state_dict()

    # test model
    model.load_state_dict(best_model)
    test_loss, test_accuracy = evaluate(model, dataset, criterion, 'test')
    print(f'Test loss: {round(test_loss, 3)}. Test accuracy: {round(test_accuracy, 2)}.')

    # save model
    print('Saving model...')
    model_path = join('output', config['data']['name'], 'weights')
    os.makedirs(model_path, exist_ok=True)
    torch.save(model.cpu().state_dict(), join(model_path, 'weights.pt'))
