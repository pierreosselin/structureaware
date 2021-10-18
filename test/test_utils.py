from module.utils import compute_p_from_sbm
import numpy as np
import pytest

def test_compute_p_from_sbm():
        
        ## Test equivalent sbm
        p_block = np.array([[0.1, 0.1], [0.1, 0.1]])
        list_blocks = np.array([10,20])
        er_p = compute_p_from_sbm(p_block, list_blocks)
        assert er_p == 0.1

        ## Test different SBM
        p_block = np.array([[0.4, 0.1], [0.1, 0.7]])
        list_blocks = np.array([10,20])
        er_p = compute_p_from_sbm(p_block, list_blocks)
        assert er_p == 171/435

        ##Test inputs sizes
        p_block = np.array([[0.4, 0.1], [0.1, 0.7]])
        list_blocks = np.array([10,20, 40])
        with pytest.raises(Exception):
                er_p = compute_p_from_sbm(p_block, list_blocks)

        ##Test inputs sizes
        p_block = np.array([[0.4, 0.1], [0.1, 0.7], [0.1, 0.2]])
        list_blocks = np.array([10,20])
        with pytest.raises(Exception):
                er_p = compute_p_from_sbm(p_block, list_blocks)
        