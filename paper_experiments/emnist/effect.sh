#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

alpha_list=(1000000)
for alpha in "${alpha_list[@]}"; do
    # run FedAvg
    echo "Run FedAvg"
    python run_experiment.py emnist FedAvg --n_learners 1 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.01 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedAvg + local adaption
    echo "Run FedAvg + local adaption"
    python run_experiment.py emnist FedAvg --n_learners 1 --n_tasks 100 --locally_tune_clients --n_rounds 100 --bz 128 \
    --lr 0.01 --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1  --alpha $alpha

    # run training using local data only
    echo "Run Local"
    python run_experiment.py emnist local --n_learners 1 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.01 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run Clustered FL
    echo "Run Clustered FL"
    python run_experiment.py emnist clustered --n_learners 1 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.01 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedProx
    echo "Run FedProx"
    python run_experiment.py emnist FedProx --n_learners 1 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.01 --mu 0.1 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --alpha $alpha

    # Run Richtarek's Formulation
    echo "Run Personalized (Richtarek's Formulation)"
    python run_experiment.py emnist pFedMe --n_learners 1 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.01 --mu 1.0 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1 --alpha $alpha

    # run FedEM
    echo "Run FedEM"
    python run_experiment.py emnist FedEM --n_learners 3 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run pFedMoE
    echo "Run pFedMoE"
    python run_experiment.py emnist pFedMoE --n_learners 2 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha

    # run pFedMM
    echo "Run pFedMM"
    python run_experiment.py emnist pFedMM --n_learners 3 --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.03 \
    --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1 --alpha $alpha
done


# n_learners=(2 4 5 6 7)

# for n in "${n_learners[@]}"; do
#     # run pFedMM
#     echo "Run pFedMM with ${n} learners"
#     python run_experiment.py emnist pFedMM --n_learners $n --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.03 \
#     --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1
# done

# n_learners=(2 4 5 6 7)
# for n in "${n_learners[@]}"; do
#     # run FedEM
#     echo "Run FedEM with ${n} learners"
#     python run_experiment.py emnist FedEM --n_learners $n --n_tasks 100 --n_rounds 100 --bz 128 --lr 0.03 \
#     --lr_scheduler multi_step --log_freq 1 --device cuda --optimizer sgd --seed 1234 --verbose 1
# done