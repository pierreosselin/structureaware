import argparse
import yaml
import torch
from module.models import load_model
from torch_geometric.data import DataLoader
from module.prediction import sample_perturbed_graphs

def vote(config):

    # Load dataset
    l_data_test = torch.load(config["save_path"] + "dataset_train")

    # Load config parameters
    nc, d, n = config["n_classes"], l_data_test[0].x.shape[1], len(l_data_test)
    loader_test = DataLoader(l_data_test, batch_size = 1, shuffle = False)

    model = load_model(config["model"])(hidden_channels=config["hidden_channels"], n_features=l_data_test[0].x.shape[1], n_classes=config["n_classes"]).cuda()
    model.load_state_dict(torch.load(config["weight_path"]))

    n_samples_eval = config["sample_eval"]
    n_samples_pre_eval = config["sample_pre_eval"]
    batch_size = config["batch_size"]
    path_votes = config["path_votes"]

    lp = config["lp"]
    n_samples = sample_config.get('n_samples', 1)
    dataset = sample_config.get('dataset', 1)

    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():

        assert n_samples % batch_size == 0
        # n_samples : number of samples for votes
        # batch_size : number of these samples forwarded in gnn model at a time
        nbatches = n_samples // batch_size

        # Loop over the graphs in the dataset
        for i, data in enumerate(loader_test):
            #Get the graph structure and attributes
            edge_idx = data.edge_index.cuda()
            x = data.x.cuda()
            n_graph = x.shape[0]
            x_batch = x.repeat(batch_size, 1)
            batch_idx = torch.arange(batch_size, device=edge_idx.device).repeat_interleave(n_graph, dim=0)
            # Loop over the perturbation graph batches
            for _ in range(nbatches):

                ### First function: should output a batch of perturbed graph the size
                # of batch_size. Including x_batch, edge_idx_batch and batch_idx.
                # Could improve by copying x
                edge_idx_batch = sample_perturbed_graphs(
                        edge_idx=edge_idx,
                        sample_config=sample_config, nsamples=batch_size, n_nodes = n_graph)
                
                predictions = model(x=x_batch, edge_index=edge_idx_batch,
                                    batch=batch_idx).argmax(1)
                preds_onehot = F.one_hot(predictions.to(torch.int64), int(nc)).sum(0)
                votes[i] += preds_onehot

            if i%10 == 0:
                print(f'Processed {i}/{n} graphs')
            
    return votes.cpu().numpy()

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    vote(config)