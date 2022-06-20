python script/train_model.py --config synthetic --device cuda:1
python script/train_model.py --config hiv --device cuda:1

python script/voting.py --config synthetic
python script/voting.py --config hiv

python script/certify_grid.py --config synthetic
python script/certify_grid.py --config hiv
