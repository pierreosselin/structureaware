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
                            NeighborhoodSubgraphPairwiseDistance,
                            VertexHistogram)
from sklearn.svm import SVC
from timebudget import timebudget
from tqdm import tqdm

from communityaware.perturb import _perturb_graph_vmap

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, default='vh', choices=['graphlet', 'nspd', 'vh'])
parser.add_argument('--motif_n', type=int, default=10)
parser.add_argument('--random_n', type=int, default=10)
parser.add_argument('--train_n', type=int, default=1000)
parser.add_argument('--valid_n', type=int, default=500)
parser.add_argument('--test_n', type=int, default=10)
parser.add_argument('--p1', type=float, default=0.05)
parser.add_argument('--p2', type=float, default=None)
parser.add_argument('--repeats', type=int, default=100_000)
parser.add_argument('--suppress_output', action='store_true', default=False)
args = parser.parse_args()

# choose kernel
if args.kernel == 'graphlet':
    kernel = GraphletSampling()
elif args.kernel == 'nspd':
    kernel = NeighborhoodSubgraphPairwiseDistance()
elif args.kernel == 'vh':
    kernel = VertexHistogram()

# determine noise
number_of_nodes = args.motif_n + args.random_n
if args.p2 is None: # isotropic noise
    noise_matrix = np.ones((number_of_nodes, number_of_nodes)) * args.p1
else:
    noise_matrix = np.zeros((number_of_nodes, number_of_nodes))
    for i in range(args.motif_n):
        for j in range(i+1, args.motif_n):
            if i == 0 and j == args.motif_n-1:
                continue
            elif i == j - 1:
                continue
            else:
                noise_matrix[i, j] = args.p1

    noise_matrix = noise_matrix + noise_matrix.T
    noise_matrix[args.motif_n:, args.motif_n:] = args.p2


# no self-loops and put into torch format
np.fill_diagonal(noise_matrix, 0)
noise_matrix = torch.tensor(noise_matrix)

# load data
fname = f'{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
data = pickle.load(open(f'data/kernel/{fname}.pkl', 'rb'))
G_train = data['G_train']
G_test = data['G_test']
y_test = data['y_test']

# load the kernel
K_train = kernel.fit_transform(G_train) # this is deterministic so its the same as in fit_model
fname = f'{args.kernel}_{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
assert np.allclose(K_train, pickle.load(open(f'output/kernel/{fname}.pkl', 'rb'))['K_train'])
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
    A = torch.tensor(np.array(nx.adjacency_matrix(graph, nodelist=range(graph.number_of_nodes())).todense()))

    # perturb graph and put into grakel format
    perturbed_graphs = _perturb_graph_vmap(A.unsqueeze(0).repeat(args.repeats, 1, 1), noise_matrix.unsqueeze(0).repeat(args.repeats, 1, 1)).numpy()
    perturbed_graphs = [nx.from_numpy_matrix(perturbed_graph) for perturbed_graph in perturbed_graphs]
    for perturbed_graph in perturbed_graphs:
        #nx.set_node_attributes(perturbed_graph, {i: 1 for i in perturbed_graph.nodes()}, 'node_label')
        nx.set_node_attributes(perturbed_graph, {i: perturbed_graph.degree(i) for i in perturbed_graph.nodes()}, 'node_label')
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
