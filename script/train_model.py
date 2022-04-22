import argparse

import torchmetrics
import torch
import yaml
from communityaware.models import GCN_Classification
from os.path import join
from communityaware.data import Synthetic, HIV
import os
import numpy as np


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='synthetic')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()
config = yaml.safe_load(open(f'config/{args.config}.yaml'))
device = args.device

# helper functions 
def train_epoch(model, train_loader, criterion, optimiser):
    model.train()
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

def evaluate(model, loader, criterion):
    with torch.no_grad():
        model.eval()
        accuracy = torchmetrics.Accuracy()
        epoch_loss = torchmetrics.MeanMetric()
        for batch in loader:  # Iterate in batches over the training/test dataset.
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            epoch_loss.update(loss.cpu())
            accuracy.update(out.cpu(), batch.y.cpu())
        return epoch_loss.compute().item(), accuracy.compute().item()

# load data
if config['dataset'].lower() == 'synthetic':
    dataset = Synthetic('data')
elif config['dataset'].lower() == 'hiv':
    dataset = HIV('data', min_required_edge_flips=20)
dataset.data.to(device)

# dataloaders
train_loader = dataset.dataloader('train', config['optimisation']['batch_size'])
valid_loader = dataset.dataloader('valid', config['optimisation']['batch_size'])
test_loader = dataset.dataloader('test', config['optimisation']['batch_size'])

# Instanciate model, optimiser and loss
model = GCN_Classification(num_features=dataset.num_features, hidden_channels=config['optimisation']['hidden_channels'], num_classes=dataset.num_classes).to(device)
optimiser = torch.optim.Adam(model.parameters(), lr=config['optimisation']['lr'])
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# early stopping
best_model = None
best_loss = np.inf

# training
for epoch in range(config['optimisation']['max_epochs']):
    train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimiser)
    valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)
    print(epoch, round(train_loss, 3), round(valid_loss, 3), round(train_accuracy, 2), round(valid_accuracy, 2))
    if valid_loss < best_loss:
        best_loss = valid_loss
        best_model = model.state_dict()

# test model
model.load_state_dict(best_model)
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test loss: {round(test_loss, 3)}. Test accuracy: {round(test_accuracy, 2)}.')

# save model
print("Saving model...")
model_path = join('output', config['dataset'], 'weights')
os.makedirs(model_path, exist_ok=True)
torch.save(model.cpu().state_dict(), join(model_path, "weights.pt"))
