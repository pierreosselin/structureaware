import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
from tqdm import tqdm
import numpy as np
from module.models import load_model
from torch_geometric.data import DataLoader
from module.prediction import load_perturbation
import torch.nn.functional as F

def vote(config):
    """Make prediction of the smoothed models

    Args:
        config ([Dic]): Dictionary of configuration
    """
    # Load dataset
    l_data_test = torch.load(config["save_path"] + "dataset_train")
    n, nc = l_data_test[0].x.shape[0], config["optimisation"]["n_classes"]
    loader_test = DataLoader(l_data_test, batch_size = 1, shuffle = False)
    
    # Load config parameters
    hidden_layers, n_class, d_features = config["hidden_channels"], config["n_classes"], l_data_test[0].x.shape[1]
    model = load_model(config["model"])(hidden_channels=hidden_layers, n_features=d_features, n_classes=n_class).cuda()
    model.load_state_dict(torch.load(config["weight_path"]))

    # Load voting parameters    
    n_samples = config["sample_eval"] # Number of samples to predict label probabilities
    batch_size = config["batch_size"] # Number of graph perturbations before prediction
    path_votes = config["path_votes"] # Path save votes
    noise_type = config["perturbation"] # Type of perturbation
    lp = config[noise_type] # List of parameters to try

    # Load type of perturbation
    perturbation_function = load_perturbation(noise_type)

    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=next(model.parameters()).device)
    
    ## Todo: Batch size might depend on the parameter (if many edges are created the memory might not be enough)
    for param_noise in tqdm(lp, desc="Loop over the parameter"):
        with torch.no_grad():
            assert n_samples % batch_size == 0
            nbatches = n_samples // batch_size

            # Loop over the graphs in the dataset
            for i, data in tqdm(enumerate(loader_test), desc="Loop over the data"):

                #Get the graph structure and attributes
                edge_idx = data.edge_index.cuda()
                x = data.x.cuda()
                n_graph = x.shape[0]
                x_batch = x.repeat(batch_size, 1)
                batch_idx = torch.arange(batch_size, device=edge_idx.device).repeat_interleave(n_graph, dim=0)
                if noise_type == "community":
                    community_node = [torch.tensor(el).clone().detach().cuda() for el in data.community_node[0]]
                    community_size = data.community_size
                    community_prob = data.community_prob

                # Loop over the perturbation graph batches
                for _ in range(nbatches):
                    if noise_type == "community":
                        edge_idx_batch = perturbation_function(edge_idx, n, batch_size, param_noise, community_node=community_node, community_size=community_size, community_prob=community_prob)
                    else:
                        edge_idx_batch = perturbation_function(edge_idx, n, batch_size, param_noise)
                    predictions = model(x=x_batch, edge_index=edge_idx_batch, batch=batch_idx).argmax(1)
                    preds_onehot = F.one_hot(predictions.to(torch.int64), int(nc)).sum(0)
                    votes[i] += preds_onehot

    np.save(path_votes, votes)

if __name__ == '__main__':

    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    args = parser.parse_args()

    # Load Config
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    vote(config)