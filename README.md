# Community Aware Randomised smoothing
Community Aware Randomised smoothing for graph classification.

## Usage

Steps 1 and 2 are agnostic of the certificate. Steps 3/4 require the same certificate parameters.

1. Generate data using `python script/generate_data.py` (this generates synthetic data, splits and communities).
2. Train a model using `python script/train_model.py`.
3. Generate votes from smoothe classifier using `python script/voting.py`.
4. Generate certificates using `python script/certify.py`.

## TODO

- [ ] Work out how voting works when using community (see cert/temp/communityaware)
- [ ] Get certificates for synthetic example and plot like in the Gunnemann paper 
- [ ] Merge generate_data.py and data.py so download generates the data. Use torch_lightning seed everywhere. Save data generating parameters in raw and check in download if theyve changed?
- [ ] Perturb.py can be split into Bernoulli and community
- [ ] Understand how the Gunnemann code works (including the correction and batching). Try a JAX perturb vs Gunnemann implementation for Bernoulli
- [ ] Add argparseconfig support 

