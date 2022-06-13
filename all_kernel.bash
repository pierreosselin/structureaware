#!/bin/bash

repeats=10000

python script_kernel/fit_model.py --kernel graphlet

for p1 in $(seq 0. 0.05 0.45)
do
    echo "graphlet" $p1
    python script_kernel/voting.py --kernel graphlet --p1 $p1 --repeats $repeats &
    for p2 in $(seq 0. 0.05 0.45)
    do
        python script_kernel/voting.py --kernel graphlet --p1 $p1 --p2 $p2 --repeats $repeats &
    done
    wait
done

python script_kernel/fit_model.py --kernel nspd

for p1 in $(seq 0. 0.05 0.45)
do
    echo "nspd" $p1
    python script_kernel/voting.py --kernel nspd --p1 $p1 --repeats $repeats &
    for p2 in $(seq 0. 0.05 0.45)
    do
        python script_kernel/voting.py --kernel nspd --p1 $p1 --p2 $p2 --repeats $repeats &
    done
    wait
done
