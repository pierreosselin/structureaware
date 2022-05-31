import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pickle
from os.path import join

import grakel
import networkx as nx
import numpy as np
import torch
from grakel.kernels import GraphletSampling
from sklearn.svm import SVC
from timebudget import timebudget
from tqdm import tqdm

from communityaware.perturb import _perturb_graph_vmap

P = (0.05, 0.45)
repeats = 100000

# noise matrix
noise_matrix = np.zeros((10, 10))
noise_matrix[0, 2] = P[0]
noise_matrix[0, 3] = P[0]
noise_matrix[1, 3] = P[0]
noise_matrix[1, 4] = P[0]
noise_matrix[2, 4] = P[0]
noise_matrix = noise_matrix + noise_matrix.T
noise_matrix[5:, 5:] = P[1]
np.fill_diagonal(noise_matrix, 0)
noise_matrix = torch.tensor(noise_matrix)

# load data
data = pickle.load(open('data/kernel/data.pkl', 'rb'))
G_train = data['G_train']
y_train = data['y_train']
G_test = data['G_test']
y_test = data['y_test']

# fit the kernel
kernel = GraphletSampling(k=5)
K_train = kernel.fit_transform(G_train)
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

votes = []
pbar = tqdm(zip(G_test, y_test), total=len(G_test))
for test_sample, label in pbar:

    # extract the graph
    graph_dict = test_sample[0]
    graph = nx.Graph()
    graph.add_nodes_from(sorted(graph_dict.keys()))
    for node in graph_dict:
        for neighbour in graph_dict[node]:
            graph.add_edge(node, neighbour)
    A = torch.tensor(np.array(nx.adjacency_matrix(graph).todense()))

    # perturb graphs
    perturbed_graphs = _perturb_graph_vmap(A.unsqueeze(0).repeat(repeats, 1, 1), noise_matrix.unsqueeze(0).repeat(repeats, 1, 1)).numpy()
    perturbed_graphs = [nx.from_numpy_matrix(perturbed_graph) for perturbed_graph in perturbed_graphs]
    perturbed_graphs = grakel.graph_from_networkx(perturbed_graphs)

    K_test = kernel.transform(perturbed_graphs) # this line takes the longest inside the loop...
    y_pred = clf.predict(K_test)
    test_accuracy = np.mean(y_pred == label)

    votes.append(torch.bincount(torch.tensor(y_pred).transpose(-1, 0), minlength=2))
    pbar.set_description(f'Accuracy of last sample: {test_accuracy}')

votes = torch.stack(votes)
fname = '_'.join(map(str, np.round(P, 8)))
votes_path = 'output/kernel/votes'
os.makedirs(votes_path, exist_ok=True)
torch.save(votes, join(votes_path, fname)) # round(, 8) is to get rid of floating point errors. E.g. 0.30000000000000004 -> 0.3
