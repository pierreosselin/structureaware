import argparse
import os
import pickle

import grakel
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--motif_n', type=int, default=10)
parser.add_argument('--random_n', type=int, default=10)
parser.add_argument('--train_n', type=int, default=1000)
parser.add_argument('--valid_n', type=int, default=1000)
parser.add_argument('--test_n', type=int, default=100)
args = parser.parse_args()


def make_er_motif(n, motif):
    while True:
        #target_degree = (2 + len(motif)-1) / 2
        graph = nx.erdos_renyi_graph(n, 0.5)
        connected = nx.is_connected(graph)
        if connected:
            break

    graph = nx.relabel_nodes(graph, {i: i+motif.number_of_nodes() for i in range(n)})
    graph.add_edges_from(motif.edges())
    graph.add_edge(motif.number_of_nodes()-1, motif.number_of_nodes())
    graph = add_attributes(graph)
    return graph

def add_attributes(graph, use_degree=True):
    if use_degree:
        nx.set_node_attributes(graph, {i: graph.degree(i) for i in graph.nodes()}, 'node_label')
    else:
        nx.set_node_attributes(graph, {i: 1 for i in graph.nodes()}, 'node_label')
    nx.set_edge_attributes(graph, {i: 1 for i in graph.edges()}, 'edge_label')
    return graph


if __name__ == '__main__':

    # specify the two motifs
    motif1 = nx.cycle_graph(args.motif_n)
    motif2 = nx.complete_graph(args.motif_n)

    # generate the dataset
    dataset = []
    labels = []
    total_n = args.train_n + args.valid_n + args.test_n

    for _ in range(int(total_n // 2)):
        dataset.append(make_er_motif(args.random_n, motif1))
        labels.append(0)
        dataset.append(make_er_motif(args.random_n, motif2))
        labels.append(1)

    labels = np.array(labels)
    dataset = list(grakel.graph_from_networkx(dataset, node_labels_tag='node_label', edge_labels_tag='edge_label'))
    G_train, G_valid_test, y_train, y_valid_test = train_test_split(dataset, labels, train_size=args.train_n, stratify=labels)
    G_test, G_valid, y_test, y_valid = train_test_split(G_valid_test, y_valid_test, train_size=args.test_n, stratify=y_valid_test) # train size here is actually test size.

    assert len(G_train) == args.train_n
    assert len(G_valid) == args.valid_n
    assert len(G_test) == args.test_n
    assert np.sum(y_test)==int(args.test_n/2)

    os.makedirs('data/kernel', exist_ok=True)
    output = {
        'G_train': G_train,
        'G_valid': G_valid,
        'G_test': G_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test
    }

    fname = f'{args.motif_n}_{args.random_n}_{args.train_n}_{args.valid_n}_{args.test_n}'
    pickle.dump(output, open(f'data/kernel/{fname}.pkl', 'wb'))
