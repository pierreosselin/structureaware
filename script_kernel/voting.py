import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pickle
from os.path import join

import grakel
import networkx as nx
import numpy as np
import torch
from grakel.kernels import (GraphletSampling,
                            NeighborhoodSubgraphPairwiseDistance)
from sklearn.svm import SVC
from timebudget import timebudget
from tqdm import tqdm

from communityaware.perturb import _perturb_graph_vmap

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, required=True, choices=['graphlet', 'nspd'])
parser.add_argument('--p1', type=float, required=True)
parser.add_argument('--p2', type=float, default=None)
parser.add_argument('--repeats', type=int, default=10_000)
parser.add_argument('--suppress_output', action='store_true', default=False)
args = parser.parse_args()

# choose kernel
if args.kernel == 'graphlet':
    kernel = GraphletSampling()
elif args.kernel == 'nspd':
    kernel = NeighborhoodSubgraphPairwiseDistance()

# determine noise
if args.p2 is None: # isotropic noise
    noise_matrix = np.ones((10, 10)) * args.p1
else:
    noise_matrix = np.ones((10, 10)) * args.p2
    noise_matrix[:5, :5] = args.p1


# no self-loops and put into torch format
np.fill_diagonal(noise_matrix, 0)
noise_matrix = torch.tensor(noise_matrix)

# load data
data = pickle.load(open('data/kernel/data.pkl', 'rb'))
G_train = data['G_train']
G_test = data['G_test']
y_test = data['y_test']

# load the kernel
K_train = kernel.fit_transform(G_train) # this is deterministic so its the same as in fit_model
assert np.allclose(K_train, pickle.load(open(f'output/kernel/training_{args.kernel}.pkl', 'rb'))['K_train'])
clf = SVC(kernel='precomputed')
clf.fit(K_train, data['y_train'])

# compute certificate for each graph in the test set
votes = []
pbar = tqdm(zip(G_test, y_test), total=len(G_test), disable=args.suppress_output)
for test_sample, label in pbar:

    # put graph into torch format
    graph_dict = test_sample[0]
    graph = nx.Graph()
    graph.add_nodes_from(sorted(graph_dict.keys()))
    for node in graph_dict:
        for neighbour in graph_dict[node]:
            graph.add_edge(node, neighbour)
    A = torch.tensor(np.array(nx.adjacency_matrix(graph).todense()))

    # perturb graph and put into grakel format
    perturbed_graphs = _perturb_graph_vmap(A.unsqueeze(0).repeat(args.repeats, 1, 1), noise_matrix.unsqueeze(0).repeat(args.repeats, 1, 1)).numpy()
    perturbed_graphs = [nx.from_numpy_matrix(perturbed_graph) for perturbed_graph in perturbed_graphs]
    for perturbed_graph in perturbed_graphs:
        nx.set_node_attributes(perturbed_graph, {i: 1 for i in perturbed_graph.nodes()}, 'node_label')
        nx.set_edge_attributes(perturbed_graph, {i: 1 for i in perturbed_graph.edges()}, 'edge_label')
    perturbed_graphs = grakel.graph_from_networkx(perturbed_graphs, node_labels_tag='node_label', edge_labels_tag='edge_label')

    # use kernel/svm to predict the label of the perturbed graphs
    K_test = kernel.transform(perturbed_graphs)
    y_pred = clf.predict(K_test)
    test_accuracy = np.mean(y_pred == label)

    votes.append(torch.bincount(torch.tensor(y_pred).transpose(-1, 0), minlength=2))
    pbar.set_description(f'Accuracy of last sample: {test_accuracy}')

# save votes
votes = torch.stack(votes)
if args.p2 is None:
     fname = '_'.join((str(args.p1), args.kernel))
else:
    fname = '_'.join((str(args.p1), str(args.p2), args.kernel))
votes_path = 'output/kernel/votes'
os.makedirs(votes_path, exist_ok=True)
torch.save(votes, join(votes_path, fname))
