import argparse
import yaml
import torch
from module.models import load_model
from torch_geometric.data import DataLoader, Data

def pre_training(config):
    n_epochs = config["optimisation"]["n_epochs"]
    hidden_channels = config["optimisation"]["hidden_channels"]
    batch_size = config["optimisation"]["batch_size"]
    model_name = config["optimisation"]["model"]

    # Load data
    l_data_train = torch.load(config["save_path"] + "dataset_train")
    l_data_test = torch.load(config["save_path"] + "dataset_test")

    # Process data such that removing attributes not important for batching
    l_data_train = [Data(x=el.x, edge_index=el.edge_index, y=el.y) for el in l_data_train]
    l_data_test = [Data(x=el.x, edge_index=el.edge_index, y=el.y) for el in l_data_test]

    # Instanciate model
    model = load_model(model_name)(n_features=l_data_train[0].x.shape[1], hidden_channels=hidden_channels, n_classes=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimisation"]["lr"])
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
    torch.save(model.state_dict(), config["weights_path"])
    return 0


if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    pre_training(config)

