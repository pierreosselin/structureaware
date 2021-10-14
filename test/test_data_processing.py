from module.data_processing import *

config = {"n_data_per_class": 300, 
        "list_blocks": [20,20,20],
        "block_probs": [[0.2, 0.02, 0.02], [0.02, 0.3, 0.02], [0.02, 0.02, 0.4]],
        "er_param": None,
        "prop_train_test": 0.7}

def test_sbm_generation():

    assert 1