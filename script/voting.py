import argparse
import yaml
import torch
from module.models import load_model
from torch_geometric.data import DataLoader
from module.prediction import predict_smooth_model

def vote(config):

    # Load dataset
    l_data_test = torch.load(config["save_path"] + "dataset_train")

    # Load config parameters
    hidden_layers, n_class, d_features, n_data = config["hidden_channels"], config["n_classes"], l_data_test[0].x.shape[1], len(l_data_test)
    loader_test = DataLoader(l_data_test, batch_size = 1, shuffle = False)

    model = load_model(config["model"])(hidden_channels=hidden_layers, n_features=d_features, n_classes=n_class).cuda()
    model.load_state_dict(torch.load(config["weight_path"]))

    n_samples_eval = config["sample_eval"] # Number of samples to predict label probabilities
    n_samples_pre_eval = config["sample_pre_eval"] # Number of samples to predict label class
    batch_size = config["batch_size"] # Number of graph perturbations before prediction
    path_votes = config["path_votes"] # Path save votes

    lp = config["lp"] # List of parameters to try
    votes = torch.zeros((n, nc), dtype=torch.long, device=next(model.parameters()).device)

    predict_smooth_model(attr_idx, edge_idx, pf, model, n, d, nc, n_samples, batch_size=1)
            
    return votes.cpu().numpy()

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    vote(config)