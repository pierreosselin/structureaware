import argparse
import os
import pickle

import grakel
import networkx as nx
import numpy as np
import torch
from grakel.kernels import (GraphletSampling,
                            NeighborhoodSubgraphPairwiseDistance,
                            VertexHistogram)
from sklearn.svm import SVC
from tqdm import tqdm

from communityaware.perturb import _perturb_graph_vmap

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, choices=['graphlet', 'nspd', 'vh'], default='vh')
parser.add_argument('--motif_n', type=int, default=10)
parser.add_argument('--random_n', type=int, default=10)
parser.add_argument('--train_n', type=int, default=1000)
parser.add_argument('--valid_n', type=int, default=1000)
parser.add_argument('--test_n', type=int, default=100)
args = parser.parse_args()

# parse arguments
if args.kernel == 'graphlet':
    kernel = GraphletSampling()
elif args.kernel == 'nspd':
    kernel = NeighborhoodSubgraphPairwiseDistance()
elif args.kernel == 'vh':
    kernel = VertexHistogram()

# load data
fname = f'{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
data = pickle.load(open(f'data/kernel/{fname}.pkl', 'rb'))
G_train, G_valid, G_test = data['G_train'], data['G_valid'], data['G_test']
y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']


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
fname = f'{args.kernel}_{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
pickle.dump(output, open(f'output/kernel/{fname}.pkl', 'wb'))
print(output['train_accuracy'], output['valid_accuracy'], output['test_accuracy'])
