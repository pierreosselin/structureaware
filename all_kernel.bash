#!/bin/bash

repeats=10000
python script_kernel/data.py
python script_kernel/fit_model.py --kernel vh

for p1 in $(seq 0.02 0.02 0.2)
do
    echo "vh" $p1
    python script_kernel/voting.py --kernel vh --p1 $p1 --repeats $repeats &
    for p2 in $(seq 0.05 0.05 0.45)
    do
        python script_kernel/voting.py --kernel vh --p1 $p1 --p2 $p2 --repeats $repeats  &
    done
    wait
done
