import argparse
import os
import pickle

import grakel
import networkx as nx
import numpy as np
import torch
from grakel.kernels import (GraphletSampling,
                            NeighborhoodSubgraphPairwiseDistance)
from sklearn.svm import SVC
from tqdm import tqdm

from communityaware.perturb import _perturb_graph_vmap

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, choices=['graphlet', 'nspd'], default='nspd')
parser.add_argument('--motif_n', type=int, default=15)
parser.add_argument('--random_n', type=int, default=17)
parser.add_argument('--train_n', type=int, default=1000)
parser.add_argument('--valid_n', type=int, default=1000)
parser.add_argument('--test_n', type=int, default=10)
parser.add_argument('--p1', type=float, default=0.05)
parser.add_argument('--p2', type=float, default=0.3)
args = parser.parse_args()

# parse arguments
if args.kernel == 'graphlet':
    kernel = GraphletSampling()
elif args.kernel == 'nspd':
    kernel = NeighborhoodSubgraphPairwiseDistance()

# load data
fname = f'{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
data = pickle.load(open(f'data/kernel/{fname}.pkl', 'rb'))
G_train, G_valid, G_test = data['G_train'], data['G_valid'], data['G_test']
y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']


def perturb_gataset(graphs, noise_matrix):
    perturbed_graphs = []
    for graph in tqdm(graphs, desc='Perturbing graphs'):
        perturbed_graphs.append(perturb_single_graph(graph, noise_matrix))
    return list(grakel.graph_from_networkx(perturbed_graphs, node_labels_tag='node_label', edge_labels_tag='edge_label'))

def perturb_single_graph(graph, noise_matrix) -> nx.Graph:
    # graph is grakel foramt
    # noise_matrix is torch format
    # put graph into torch format
    graph_dict = graph[0]
    graph = nx.Graph()
    graph.add_nodes_from(sorted(graph_dict.keys()))
    for node in graph_dict:
        for neighbour in graph_dict[node]:
            graph.add_edge(node, neighbour)
    A = torch.tensor(np.array(nx.adjacency_matrix(graph).todense()))

    # perturb graph and put into grakel format
    perturbed_graph = _perturb_graph_vmap(A.unsqueeze(0), noise_matrix.unsqueeze(0)).numpy().squeeze()
    perturbed_graph = nx.from_numpy_matrix(perturbed_graph)
    nx.set_node_attributes(perturbed_graph, {i: 1 for i in perturbed_graph.nodes()}, 'node_label')
    nx.set_edge_attributes(perturbed_graph, {i: 1 for i in perturbed_graph.edges()}, 'edge_label')
    return perturbed_graph

number_of_nodes = args.motif_n + args.random_n - 2
if args.p1 is not None:  # apply noise
    if args.p2 is not None:  # apply anisotropic noise
        noise_matrix = np.ones((number_of_nodes, number_of_nodes)) * args.p1
        noise_matrix[:args.motif_n, :args.motif_n] = args.p2
    else:  # apply isotropic noise
        noise_matrix = np.ones((number_of_nodes, number_of_nodes)) * args.p1
    np.fill_diagonal(noise_matrix, 0)
    noise_matrix = torch.tensor(noise_matrix)
    G_train = perturb_gataset(G_train, noise_matrix)

# fit the kernel
print('Fitting kernel')
K_train = kernel.fit_transform(G_train)
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

print('Evaluating kernel')
# evaluation function
def evaluate(G, y):
    K = kernel.transform(G)
    y_pred = clf.predict(K)
    return np.mean(y_pred==y)

# compute accuracy on three datasets.
output = {
    'K_train': K_train,
    'train_accuracy': evaluate(G_train, y_train),
    'valid_accuracy': evaluate(G_valid, y_valid),
    'test_accuracy': evaluate(G_test, y_test)
}

# save output
os.makedirs('output/kernel', exist_ok=True)
fname = f'{args.kernel}_{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}_{args.p1}_{args.p2}'
pickle.dump(output, open(f'output/kernel/{fname}.pkl', 'wb'))
print(output['train_accuracy'], output['valid_accuracy'], output['test_accuracy'])
