import os

from communityaware.data import HIV, CoraML, Synthetic


def mask_other_gpus(gpu_number):
    """Mask all other GPUs than the one specified."""
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_number)


def load_dataset(dataset):
    if dataset.lower() == 'synthetic':
        dataset = Synthetic('data')
    elif dataset.lower() == 'hiv':
        dataset = HIV('data', min_required_edge_flips=20)
    elif dataset.lower() == 'cora_ml':
        dataset = CoraML('data')
    else:
        raise ValueError('Dataset {} not supported'.format(dataset))
    return dataset