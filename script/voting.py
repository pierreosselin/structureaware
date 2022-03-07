import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from module.models import load_model
from module.prediction import load_perturbation
from torch_geometric.data import DataLoader
from tqdm import tqdm


def vote(config):
    """Make prediction of the smoothed models

    Args:
        config ([Dic]): Dictionary of configuration
    """
    # Load dataset
    l_data_test = torch.load(config["save_path"] + "dataset_train")
    n_graph = len(l_data_test)
    n, nc = l_data_test[0].x.shape[0], config["optimisation"]["n_classes"]
    loader_test = DataLoader(l_data_test, batch_size = 1, shuffle = False)
    
    # Load config parameters
    hidden_layers, n_class, d_features = config["optimisation"]["hidden_channels"], config["optimisation"]["n_classes"], l_data_test[0].x.shape[1]
    model = load_model(config["optimisation"]["model"])(hidden_channels=hidden_layers, n_features=d_features, n_classes=n_class)
    model.load_state_dict(torch.load(config["weight_path"] + "GCN.pth"))
    model.to("cuda")

    # Load voting parameters    
    n_samples = config["certification"]["sample_eval"] # Number of samples to predict label probabilities
    batch_size = config["certification"]["batch_size"] # Number of graph perturbations before prediction
    path_votes = config["vote_path"] # Path save votes
    noise_type = config["certification"]["perturbation"] # Type of perturbation
    lp = config["certification"]["parameter_list"] # List of parameters to try
    override = config["override_vote"]

    ###Create folders for output, overide if necessary
    path_creation = Path(f"./{path_votes}")
    if path_creation.exists():
        if override:
            print("The vote folder already exists and the override option is on, votes deleted.")
            shutil.rmtree(path_creation)
        else:
            raise Exception("The vote folder already exists and the override option is off, vote generation aborted.")
    path_creation.mkdir(parents=True, exist_ok=True)

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

    torch.save(votes, path_votes + "votes")

if __name__ == '__main__':

    # Argument definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='default_config')
    args = parser.parse_args()

    # Load Config
    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    vote(config)