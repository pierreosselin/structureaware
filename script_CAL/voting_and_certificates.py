import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from CAL.datasets import get_dataset
from CAL.opts import get_model, parse_args
from CAL.train_smooth import eval_acc_causal, smoothbatch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

import wandb
from communityaware.models import GCN
from communityaware.perturb import perturb_graph
from communityaware.utils import load_dataset, load_model, make_noise_grid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--model', type=str, default='CausalGAT')
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--p_attention_cutoff', type=float, default=0.0)
    parser.add_argument('--p_large_attention', type=float, default=0.0)
    parser.add_argument('--p_small_attention', type=float, default=0.0)
    parser.add_argument('--repeats', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sweep', type=str, default=None)
    parser.add_argument('--use_wandb', type=bool, default=False)

    args = parser.parse_args()
    device = args.device

    if args.sweep is not None and args.use_wandb:
        with open(args.sweep) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        wandb.init(project='smooth_CAL', entity='kenlay', config=config)
        args.p_attention_cutoff = float(wandb.config.p_attention_cutoff)
        args.p_large_attention = float(wandb.config.p_large_attention)
        args.p_small_attention = float(wandb.config.p_small_attention)
        wandb.config = {
            'p_attention_cutoff': args.p_attention_cutoff,
            'p_large_attention': args.p_large_attention,
            'p_small_attention': args.p_small_attention
        }
        wandb.run.name = f'{args.p_attention_cutoff}_{args.p_large_attention}_{args.p_small_attention}'
        wandb.run.save()
    elif args.use_wandb:
        wandb.init(project='smooth_CAL', entity='kenlay')

    data_path = 'data'
    model_path = join('output', args.dataset, 'weights')
    votes_path = join('output', args.dataset, 'votes')

    # load data
    dataset = get_dataset(args.dataset, sparse=True, feat_str='', root='CAL/data/')
    dataset_name = args.dataset.lower()

    # Load model
    fname = f'script_CAL/CAL/models/{args.p_attention_cutoff}_{args.p_large_attention}_{args.p_small_attention}.pt'
    loaded_data = torch.load(fname)
    model_weights, test_idx = loaded_data['weights'], loaded_data['test_idx']
    model_args = parse_args()
    model_args.layers = args.layers
    model_args.model = args.model
    model_args.dataset = args.dataset
    model_func = get_model(model_args)
    model = model_func(dataset.num_features, dataset.num_classes).to(device)
    model.load_state_dict(model_weights)
    model.eval()

    # evaluate the test set accuracy
    #test_data = list(DataLoader(dataset[test_idx], len(test_idx), shuffle=True))[0] #indexing dataset directly didnt work for some reason
    #accuracy = []
    #for _ in range(10):
    #    test_data = smoothbatch(test_data, model, args.p_attention_cutoff, args.p_large_attention, args.p_small_attention)
    #    _, _, co_logs = model(test_data, eval_random=False)
    #    pred = co_logs.max(1)[1]
    #    accuracy.append(pred.eq(test_data.y.view(-1)).sum().item() / test_data.num_graphs)

    # compute certificates
    if args.p_large_attention > 0 and args.p_small_attention > 0:
        with torch.no_grad():
            votes = torch.zeros((len(test_idx), dataset.num_classes), dtype=torch.long).to(device)
            test_loader = DataLoader(dataset[test_idx], 1, shuffle=False)
            for i, graph in tqdm(enumerate(test_loader), leave=False, total=len(test_loader)):
                perturbed_graphs = DataLoader([graph for _ in range(args.repeats)], batch_size=args.batch_size)
                for batch in perturbed_graphs:
                    batch = batch.to(device)
                    batch = smoothbatch(batch, model, args.p_attention_cutoff, args.p_large_attention, args.p_small_attention)
                    _, _, co_logs = model(batch)
                    predictions = co_logs.max(1)[1]
                    votes[i] += torch.bincount(predictions, minlength=dataset.num_classes)

                # select single graph in batch
                perturbed_graphs = DataLoader([graph,])
                for batch in perturbed_graphs:
                    batch = batch.to(device)

                # count number of flips we can certify up to (i.e. number of pairwise nodes which are below or above cutoff)
                batch, pairwise_att = smoothbatch(batch, model, args.p_attention_cutoff, args.p_large_attention, args.p_small_attention, return_pairwise_att=True)
                row, col = torch.tril_indices(graph.num_nodes, graph.num_nodes)
                pairwise_att[row, col]=0.0
                small_attention_flips = torch.bitwise_and(pairwise_att < args.p_attention_cutoff, ~torch.isclose(pairwise_att, torch.tensor(0.0))).flatten().sum()
                large_attention_flips = torch.bitwise_and(pairwise_att >= args.p_attention_cutoff, ~torch.isclose(pairwise_att, torch.tensor(0.0))).flatten().sum()

        assert votes.sum(1).unique().item() == args.repeats # this also implicitly checks unique is size 1.
        votes_path = os.path.join('output', args.dataset, 'cal_votes')
        os.makedirs(votes_path, exist_ok=True)
        noise = np.array((args.p_attention_cutoff, args.p_large_attention, args.p_small_attention))
        fname = join(votes_path, '_'.join(map(str, np.round(noise, 8))))

        torch.save({
            'votes': votes,
            'small_attention_flips': small_attention_flips,
            'large_attention_flips': large_attention_flips
        }, join(votes_path, '_'.join(map(str, np.round(noise, 8))))) # round(, 8) is to get rid of floating point errors. E.g. 0.30000000000000004 -> 0.3
