import numpy as np
import torch

from communityaware.data.utils import split_data


def test_split_data():
    labels = torch.tensor([0 for _ in range(50)] + [1 for _ in range(50)] + [2 for _ in range(50)] + [3 for _ in range(50)])
    split = split_data(labels, 0.5, 0.25)
    assert disjoint_union(split)
    assert balanced_labels(split, labels)
    assert len(split['valid']) == 50

    split = split_data(labels, 0.3, 0.1)
    assert disjoint_union(split)
    assert balanced_labels(split, labels)
    assert len(split['valid']) == 120

    split = split_data(labels, 0.3, 20)
    assert disjoint_union(split)
    assert balanced_labels(split, labels)
    assert len(split['train']) == 60
    assert len(split['valid']) == 120
    assert len(split['test']) == 20

def disjoint_union(split):
    train, valid, test = set(split['train']), set(split['valid']), set(split['test'])
    if not train.intersection(valid).intersection(test) == set():
        return False
    elif not train.union(valid).union(test) == set(range(len(train)+len(valid)+len(test))):
        return False
    else:
        return True

def balanced_labels(split, labels):
    train, valid, test = split['train'], split['valid'], split['test']
    if not np.bincount(labels[train]).max() - np.bincount(labels[train]).min() in [0, 1]:
        return False
    elif not np.bincount(labels[valid]).max() - np.bincount(labels[valid]).min() in [0, 1]:
        return False
    elif not np.bincount(labels[valid]).max() - np.bincount(labels[valid]).min() in [0, 1]:
        return False
    else:
        return True


test_split_data()
