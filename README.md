# CommunityAwareRS
Community Aware Randomized smoothing for graph classification


All undirected graph are represented with double edges in pytorch geometric

TODO:
- Incorporate GPU option


## Usage

1. Generate data using `python script/process_data.py` (this generates synthetic data, splits and communities).
2. Train a model using `python script/train_model.py`.
3. Generate votes from smoothe classifier using `python script/voting.py`.
3. Generate certificates using `python script/certify.py`.