import argparse
import os
import pickle

import numpy as np
from grakel.kernels import (GraphletSampling,
                            NeighborhoodSubgraphPairwiseDistance)
from sklearn.svm import SVC

parser = argparse.ArgumentParser()
parser.add_argument('--kernel', type=str, required=True, choices=['graphlet', 'nspd'])
args = parser.parse_args()

# parse arguments
if args.kernel == 'graphlet':
    kernel = GraphletSampling()
elif args.kernel == 'nspd':
    kernel = NeighborhoodSubgraphPairwiseDistance()

# load data
data = pickle.load(open('data/kernel/data.pkl', 'rb'))
G_train, G_valid, G_test = data['G_train'], data['G_valid'], data['G_test']
y_train, y_valid, y_test = data['y_train'], data['y_valid'], data['y_test']

# fit the kernel
K_train = kernel.fit_transform(G_train)
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)

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
pickle.dump(output, open(f'output/kernel/training_{args.kernel}.pkl', 'wb'))
