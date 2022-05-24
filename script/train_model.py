import argparse
import os
from os.path import join

import numpy as np
import torch
import torchmetrics
import yaml

from communityaware.models import GCN
from communityaware.utils import load_dataset


# helper functions for graph classification
def train_epoch_graph_classification(model, dataset, criterion, optimiser):
    model.train()
    train_loader = dataset.dataloader('train', config['training']['batch_size'])
    accuracy = torchmetrics.Accuracy()
    epoch_loss = torchmetrics.MeanMetric()
    for batch in train_loader:  # Iterate in batches over the training dataset.
        out = model(batch.x, batch.edge_index, batch.batch)  # Perform a single forward pass.
        loss = criterion(out, batch.y)  # Compute the loss.
        loss.mean().backward()  # Derive gradients.
        epoch_loss.update(loss.cpu())
        accuracy.update(out.cpu(), batch.y.cpu())
        optimiser.step()  # Update parameters based on gradients.
        optimiser.zero_grad()  # Clear gradients.
    return epoch_loss.compute().item(), accuracy.compute().item()

def evaluate_graph_classification(model, dataset, criterion, split):
    with torch.no_grad():
        loader = dataset.dataloader(split, config['training']['batch_size'])
        model.eval()
        accuracy = torchmetrics.Accuracy()
        epoch_loss = torchmetrics.MeanMetric()
        for batch in loader:  # Iterate in batches over the training/test dataset.
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            epoch_loss.update(loss.cpu())
            accuracy.update(out.cpu(), batch.y.cpu())
        return epoch_loss.compute().item(), accuracy.compute().item()

# helper functions for node classification
def train_epoch_node_classification(model, dataset, criterion, optimiser):
    model.train()
    train_mask = dataset.data.train_mask
    out = model(dataset.data.x, dataset.data.edge_index)[train_mask]
    loss = criterion(out, dataset.data.y[train_mask]).mean()  # Compute the loss.
    loss.backward()  # Derive gradients.
    accuracy = torchmetrics.Accuracy()(out.cpu(), dataset.data.y[train_mask].cpu())
    optimiser.step()  # Update parameters based on gradients.
    optimiser.zero_grad()  # Clear gradients.
    return loss.item(), accuracy.item()


def evaluate_node_classification(model, dataset, criterion, split):
    with torch.no_grad():
        model.eval()
        mask = dataset.data.test_mask if split=='test' else dataset.data.val_mask
        out = model(dataset.data.x, dataset.data.edge_index)[mask]
        loss = criterion(out, dataset.data.y[mask]).mean()
        accuracy = torchmetrics.Accuracy()(out.cpu(), dataset.data.y[mask].cpu())
        return loss.item(), accuracy.item()

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device

    # load data
    dataset = load_dataset(config)
    dataset.data.to(device)

    # determine if the dataset is graph or node classification
    graph_classification_task = True if config['data']['name'].lower() in ['synthetic', 'hiv'] else False

    # Instanciate model, optimiser and loss
    model = GCN(num_features=dataset.num_features,
        hidden_channels=config['model']['hidden_channels'],
        num_classes=dataset.num_classes,
        pooling =graph_classification_task
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # early stopping
    best_model = None
    best_loss = np.inf

    # training loop functions
    train_epoch = train_epoch_graph_classification if graph_classification_task else train_epoch_node_classification
    evaluate = evaluate_graph_classification if graph_classification_task else evaluate_node_classification

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
