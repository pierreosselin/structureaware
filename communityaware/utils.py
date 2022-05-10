import os

import torch


def mask_other_gpus(gpu_number):
    """Mask all other GPUs than the one specified."""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)