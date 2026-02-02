#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

# run FedAvg
echo "Run FedAvg"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "run FedAvg + local adaption"
python run_experiment.py cifar100 FedAvg --n_learners 1 --n_tasks 100 --locally_tune_clients --n_rounds 200 --bz 128 \
 --lr 0.001 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py cifar100 local --n_learners 1 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py cifar100 clustered --n_learners 1 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py cifar100 FedProx --n_learners 1 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.01 --mu 0.1 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run Richtarek's Formulation
echo "Run Personalized (Richtarek's Formulation)"
python run_experiment.py cifar100 pFedMe --n_learners 1 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.001 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py cifar100 FedEM --n_learners 3 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run pFedMoE
echo "Run pFedMoE"
python run_experiment.py cifar100 pFedMoE --n_learners 2 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run pFedMM
echo "Run pFedMM"
python run_experiment.py cifar100 pFedMM --n_learners 3 --n_tasks 100 --n_rounds 200 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1