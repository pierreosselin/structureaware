import argparse

import torch
import yaml
from module.models import GCN_Classification
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def train_model(max_epochs, lr, hidden_channels, batch_size, save_path, weight_path, device):

    # Load data
    l_data_train = torch.load(save_path + "dataset_train")
    l_data_test = torch.load(save_path + "dataset_test")

    # Process data such that removing attributes not important for batching
    l_data_train = [Data(x=el.x, edge_index=el.edge_index, y=el.y).to(device) for el in l_data_train]
    l_data_test = [Data(x=el.x, edge_index=el.edge_index, y=el.y).to(device) for el in l_data_test]

    # Instanciate model
    model = GCN_Classification(n_features=l_data_train[0].x.shape[1], hidden_channels=hidden_channels, n_classes=2).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(l_data_train, batch_size = batch_size)
    valid_loader = DataLoader(l_data_test, batch_size = batch_size)
    test_loader = DataLoader(l_data_test, batch_size = batch_size)

    for epoch in range(max_epochs):
        train_epoch(model, train_loader, criterion, optimiser)
        evaluate(model, valid_loader)

    print("Saving model...")
    torch.save(model.state_dict(), weight_path + "GCN.pth")


def train_epoch(model, train_loader, criterion, optimiser):
    model.train()
    for batch in train_loader:  # Iterate in batches over the training dataset.
        out = model(batch.x, batch.edge_index, batch.batch)  # Perform a single forward pass.
        loss = criterion(out, batch.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimiser.step()  # Update parameters based on gradients.
        optimiser.zero_grad()  # Clear gradients.


def evaluate(model, loader):
    model.eval()
    correct = 0
    for batch in loader:  # Iterate in batches over the training/test dataset.
        out = model(batch.x, batch.edge_index, batch.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == batch.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device
    train_model(config["optimisation"]["n_epochs"], config["optimisation"]['lr'], config["optimisation"]["hidden_channels"], config["optimisation"]["batch_size"], config["save_path"], config["weight_path"], device)

