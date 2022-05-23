# Community Aware Randomised smoothing
Community Aware Randomised smoothing for graph classification.

## Usage

1. Train a model using `python script/train_model.py`.
2. Generate votes from smoothe classifier using `python script/voting.py`.
3. Generate certificates using `python script/certify_grid.py`.

## TODO

Code
- [x] Work out how voting works when using community (see cert/temp/communityaware)
- [x] Refactor voting so its simpler
- [x] Get certificates for synthetic example and plot like in the Gunnemann paper
- [x] Merge generate_data.py and data.py so download generates the data. Use torch_lightning seed everywhere. Save data generating parameters in raw and check in download if theyve changed?
- [x] Perturb.py can be split into Bernoulli and community
- [x] Understand how the Gunnemann code works (including the correction and batching). Try a JAX perturb vs Gunnemann implementation for Bernoulli
- [x] Get arbitary precision working!!
- [ ] Do an experiment on sparsity aware
- [ ] Exhaustive search on adversarial examples?
- [ ] Add argparseconfig support and print args for scripts
- [ ] Seperate configs into common and experiment specific
- [x] Certificates in JAX? - No because need arbitary precision.
- [ ] More sophisticated model adn the load_model util
- [ ] Tidy code using black and add it as a pre-commit hook

Theory
- [x] Is there a nicer way to make framework more general and reduce compute? No.
- [x] What happens if the radius is 0? PREDICT instead of CERTIFY?
- [ ] Can we prove monoticity? Proof could be to show certificate (r_1, r_2, ...,r_n) holds then so does (r_1 - 1, r_2, ...,r_n).
- [x] Subregions - Comes from the Lee paper.
