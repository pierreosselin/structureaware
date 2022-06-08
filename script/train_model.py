import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch_geometric.data import InMemoryDataset

from communityaware.utils import load_dataset, load_model


# helper functions for graph classification
def train_epoch(model: nn.Module, dataset: InMemoryDataset, criterion: _Loss, optimiser: Optimizer, smoothing: int=None) -> tuple:
    if smoothing is not None:
        raise NotImplementedError('Smoothing not implemented for graph classification.')
    model.train()
    train_loader = dataset.dataloader('train', config['training']['batch_size'])
    accuracy = torchmetrics.Accuracy()
    epoch_loss = torchmetrics.MeanMetric()
    for batch in train_loader:  # Iterate in batches over the training dataset.
        out = model(batch.x, batch.edge_index, batch.batch)  # Perform a single forward pass.
        loss = criterion(out, batch.y)  # Compute the loss.
        loss.mean().backward()  # Derive gradients.
        epoch_loss.update(loss.cpu())
        accuracy.update(out.squeeze().cpu(), batch.y.cpu())
        optimiser.step()  # Update parameters based on gradients.
        optimiser.zero_grad()  # Clear gradients.
    return epoch_loss.compute().item(), accuracy.compute().item()

def evaluate(model: nn.Module, dataset: InMemoryDataset, criterion: _Loss, split: str) -> tuple:
    with torch.no_grad():
        loader = dataset.dataloader(split, config['training']['batch_size'])
        model.eval()
        accuracy = torchmetrics.Accuracy()
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
    parser.add_argument('--config', type=str, default='synthetic')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device

    # load data
    dataset = load_dataset(config)
    dataset.data.to(device)
    dataset_name = config['data']['name'].lower()

    # determine if the dataset is graph or node classification
    graph_classification_task = True if dataset_name in ['synthetic', 'hiv'] else False

    # determine if we should use positional_encoding
    use_positional_encoding = True if dataset_name == 'synthetic' else False

    # Instanciate model, optimiser and loss
    model = load_model(config).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    if dataset.num_classes == 2:
        _criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        criterion = lambda x, y: _criterion(x.squeeze(), y.float())
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')


    # early stopping
    best_model = None
    best_loss = np.inf

    # training
    for epoch in range(config['training']['max_epochs']):
        train_loss, train_accuracy = train_epoch(model, dataset, criterion, optimiser)
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
