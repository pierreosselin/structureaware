import argparse
import yaml
import torch
from module.models import load_model
from torch_geometric.data import DataLoader


def vote(config):

    # Load dataset

    l_data_test = torch.load(config["save_path"] + "dataset_train")
    
    loader_test = DataLoader(l_data_test, batch_size = 1, shuffle = False)
    dataset=config["dataset"]

    model = load_model(config["model"])(hidden_channels=64, n_features=l_data_test[0].x.shape[1], n_classes=2).cuda()
    model.load_state_dict(torch.load(config["model_weights"]))

    n_samples_eval = config["sample_eval"]
    n_samples_pre_eval = config["sample_pre_eval"]
    batch_size = config["batch_size"]
    path_votes = config["path_votes"]

    lp = config["lp"]
    path_certificates = config["path_certificates"]

    nc = 2
    d = l_data_test[0].x.shape[1]
    n = len(l_data_test)
    conf_alpha = config["conf_alpha"]

    predictor = load_predictor(config["experiment"])
    certificate = load_certificate(config["experiment"])
    certificate_parameters = config["certificate_parameters"]
    certificate_parameters["loader"] = loader_test



    ### Raise error for value perturbation parameter
    model.eval()
    votes = torch.zeros((n, nc), dtype=torch.long, device=edge_idx.device)
    with torch.no_grad():
        assert n_samples % batch_size == 0
        nbatches = n_samples // batch_size
        for _ in tqdm(range(nbatches)):
            edge_idx_batch = sparse_perturb_multiple(data_idx=edge_idx, n=n, m=n, undirected=True,
                                               pf=pf, nsamples=batch_size, offset_both_idx=True)
            attr_idx_batch = copy_idx(idx=attr_idx, dim_size=n, ncopies=batch_size, offset_both_idx=False)
            predictions = model(attr_idx=attr_idx_batch, edge_idx=edge_idx_batch,
                                n=batch_size * n, d=d).argmax(1)
            preds_onehot = F.one_hot(predictions, int(nc)).reshape(batch_size, n, nc).sum(0)
            votes += preds_onehot
    return votes.cpu().numpy()

if __name__ == '__main__':
    # Argument definition
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='default_config')

    args = parser.parse_args()

    config = yaml.safe_load(open(f'config/{args.config}.yaml'))

    vote(config)