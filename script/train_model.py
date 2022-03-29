import argparse
import shutil
from pathlib import Path

import torch
import yaml
from module.models import GCN_Classification
from torch_geometric.data import Data, DataLoader


def pre_training(config, device):
    n_epochs = config["optimisation"]["n_epochs"]
    hidden_channels = config["optimisation"]["hidden_channels"]
    batch_size = config["optimisation"]["batch_size"]
    lr=config["optimisation"]["lr"]
    override = config["optimisation"]["model"]

    save_path, weight_path = config["save_path"], config["weight_path"]

    # Load data
    l_data_train = torch.load(save_path + "dataset_train")
    l_data_test = torch.load(save_path + "dataset_test")

    # Process data such that removing attributes not important for batching
    l_data_train = [Data(x=el.x, edge_index=el.edge_index, y=el.y).to(device) for el in l_data_train]
    l_data_test = [Data(x=el.x, edge_index=el.edge_index, y=el.y).to(device) for el in l_data_test]

    # Instanciate model
    model = GCN_Classification(n_features=l_data_train[0].x.shape[1], hidden_channels=hidden_channels, n_classes=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(l_data_train, batch_size = batch_size)
    test_loader = DataLoader(l_data_test, batch_size = batch_size)

    def train():
        model.train()
        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()
        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(1, n_epochs):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print("Saving model...")
    torch.save(model.state_dict(), weight_path + "GCN.pth")
    return 0


if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    parser.add_argument('--device', type=str, default='cuda:2')
    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))
    device = args.device
    pre_training(config, device)

