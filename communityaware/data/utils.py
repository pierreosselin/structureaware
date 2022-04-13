from sklearn.model_selection import train_test_split
import numpy as np

def split_data(y, train_size, val_size, test_size):
    idx = np.arange(len(y))
    train_idx, valid_test_idx = train_test_split(idx, train_size = train_size, stratify=y)
    valid_idx, test_idx = train_test_split(valid_test_idx, train_size = val_size/(val_size+test_size), stratify=y[valid_test_idx])
    split = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return split

def assign_graph_ids(data_list):
    for i, graph in enumerate(data_list):
        graph.idx = i
