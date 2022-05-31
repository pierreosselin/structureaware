import os
import pickle

import grakel
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split


def make_er_motif(n, motif):
    while True:
        g = nx.erdos_renyi_graph(n, np.log(n)/n)
        connected = nx.is_connected(g)
        if connected and g.has_edge(0, 1):
            break
    g = nx.relabel_nodes(g, {i: i+3 for i in range(n)})
    g.add_edges_from(motif.edges())
    nx.set_node_attributes(g, {i: 1 for i in g.nodes()}, 'node_label')
    nx.set_edge_attributes(g, {i: 1 for i in g.edges()}, 'edge_label')
    return g

if __name__ == '__main__':

    # specify the two motifs
    motif1 = nx.cycle_graph(5)
    motif2 = nx.complete_graph(5)

    # generate the dataset
    dataset = []
    labels = []
    for _ in range(30):
        dataset.append(make_er_motif(7, motif1))
        dataset.append(make_er_motif(7, motif2))
        labels.append(0)
        labels.append(1)
    labels = np.array(labels)
    dataset = list(grakel.graph_from_networkx(dataset, node_labels_tag='node_label', edge_labels_tag='edge_label'))
    G_train, G_test, y_train, y_test = train_test_split(dataset, labels, train_size=50, random_state=42)

    os.makedirs('data/kernel', exist_ok=True)
    output = {
        'G_train': G_train,
        'G_test': G_test,
        'y_train': y_train,
        'y_test': y_test
    }
    pickle.dump(output, open('data/kernel/data.pkl', 'wb'))
