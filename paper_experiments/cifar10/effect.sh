#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1

alpha_list=(1000000)

for alpha in "${alpha_list[@]}"; do
    # run FedAvg
    echo "Run FedAvg"
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.01 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedAvg + local adaption
    echo "run FedAvg + local adaption"
    python run_experiment.py cifar10 FedAvg --n_learners 1 --n_tasks 80 --locally_tune_clients --n_rounds 200 --bz 128 \
    --lr 0.001 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run training using local data only
    echo "Run Local"
    python run_experiment.py cifar10 local --n_learners 1 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run Clustered FL
    echo "Run Clustered FL"
    python run_experiment.py cifar10 clustered --n_learners 1 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.003 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedProx
    echo "Run FedProx"
    python run_experiment.py cifar10 FedProx --n_learners 1 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.01 --mu 1.0\
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --alpha $alpha

    # Run pFedME
    echo "Run pFedME"
    python run_experiment.py cifar10 pFedMe --n_learners 1 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.001 --mu 1.0 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedEM
    echo "Run FedEM"
    python run_experiment.py cifar10 FedEM --n_learners 3 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run pFedMoE
    echo "Run pFedMoE"
    python run_experiment.py cifar10 pFedMoE --n_learners 2 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run pFedMM
    echo "Run pFedMM"
    python run_experiment.py cifar10 pFedMM --n_learners 3 --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha
done

# n_learners=(2 4 5 6 7)

# for n in "${n_learners[@]}"; do
#     # run pFedMM
#     echo "Run pFedMM with ${n} learners"
#     python run_experiment.py cifar10 pFedMM --n_learners $n --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
#     --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1
# done

# n_learners=(2 4 5 6 7)
# for n in "${n_learners[@]}"; do
#     # run FedEM
#     echo "Run FedEM with ${n} learners"
#     python run_experiment.py cifar10 FedEM --n_learners $n --n_tasks 80 --n_rounds 200 --bz 128 --lr 0.03 \
#     --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1
# done
