import argparse
import torch
import torch.nn.functional as F
import yaml
from module.models import GCN_Classification
from module.perturb import load_perturbation
from module.data import Synthetic
from torch_geometric.data import DataLoader
from tqdm import tqdm
from os.path import join
import os

def vote(data_path, weight_path, hidden_channels, device, sample_eval, batch_size, vote_path, perturbation, parameter_list):
    """Make prediction of the smoothed models

    Args:
        config ([Dic]): Dictionary of configuration
    """

    # Load dataset
    dataset = Synthetic(data_path)
    test_loader = dataset.dataloader('test', batch_size=1)
    n_graph = len(test_loader)
    n, nc, d_features = dataset[0].num_nodes, dataset.num_classes, dataset.num_features

    
    # Load config parameters
    hidden_layers, n_class = hidden_channels, nc
    model = GCN_Classification(hidden_channels=hidden_layers, num_features=d_features, num_classes=n_class)
    model.load_state_dict(torch.load(join(weight_path, "weights.pt")))
    model.to(device)

    # Load voting parameters    
    n_samples = sample_eval # Number of samples to predict label probabilities
    batch_size = batch_size # Number of graph perturbations before prediction
    path_votes = vote_path # Path save votes
    noise_type = perturbation # Type of perturbation
    lp = parameter_list # List of parameters to try

    # Load type of perturbation
    perturbation_function = load_perturbation(noise_type)
    model.eval()
    votes = torch.zeros((n_graph, nc), dtype=torch.long, device=next(model.parameters()).device)
    ## Todo: Batch size might depend on the parameter (if many edges are created the memory might not be enough)
    for param_noise in tqdm(lp, desc="Loop over the parameter"):
        with torch.no_grad():
            assert n_samples % batch_size == 0
            nbatches = n_samples // batch_size

            # Loop over the graphs in the dataset
            for i, data in tqdm(enumerate(test_loader), desc="Loop over the data"):

                #Get the graph structure and attributes
                edge_idx = data.edge_index.to(device)
                x = data.x.to(device)
                n_graph = x.shape[0]
                x_batch = x.repeat(batch_size, 1)
                batch_idx = torch.arange(batch_size, device=edge_idx.device).repeat_interleave(n_graph, dim=0)
                if noise_type == "community":
                    community_node = [torch.tensor(el).clone().detach().to(device) for el in data.community_node[0]]
                    community_size = data.community_size
                    community_prob = data.community_prob

                # Loop over the perturbation graph batches
                for _ in range(nbatches):
                    if noise_type == "community":
                        edge_idx_batch = perturbation_function(edge_idx, n, batch_size, param_noise, community_node=community_node, community_size=community_size, community_prob=community_prob, device=device)
                    else:
                        edge_idx_batch = perturbation_function(edge_idx, n, batch_size, param_noise, device=device)
                    predictions = model(x=x_batch, edge_index=edge_idx_batch, batch=batch_idx).argmax(1)
                    preds_onehot = F.one_hot(predictions.to(torch.int64), int(nc)).sum(0)
                    votes[i] += preds_onehot

    os.makedirs(path_votes, exist_ok=True)
    torch.save(votes.cpu(), join(path_votes, "votes"))

if __name__ == '__main__':

    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='synthetic')
    args = parser.parse_args()
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    data_path = 'data'
    model_path = join('output', config['dataset'], 'weights')
    vote_path = join('output', config['dataset'], 'votes')

    vote(data_path=data_path,
         weight_path=model_path,
         hidden_channels=config['optimisation']['hidden_channels'],
         device='cuda:1',
         sample_eval=config['certification']['sample_eval'],
         batch_size=config['certification']['batch_size'],
         vote_path=vote_path,
         perturbation=config["certification"]["perturbation"],
         parameter_list=config["certification"]["parameter_list"])