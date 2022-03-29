# Community Aware Randomised smoothing
Community Aware Randomised smoothing for graph classification.

## Usage

Steps 1 and 2 are agnostic of the certificate. Steps 3/4 require the same certificate parameters.

1. Generate data using `python script/process_data.py` (this generates synthetic data, splits and communities).
2. Train a model using `python script/train_model.py`.
3. Generate votes from smoothe classifier using `python script/voting.py`.
4. Generate certificates using `python script/certify.py`.