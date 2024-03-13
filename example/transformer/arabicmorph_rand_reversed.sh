#!/usr/bin/bash

python src/train_al.py \
    --training_method "rand" \
    --database "arabicmorphbaselinereversed" \
    --prefix "rand_rev_am1" \
    --start_cycle 0 \
    --train_batch_size 400 \
    --test_batch_size 6 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 250 \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/1_tune_13171.tsv \
    data/arabicmorph/active/1_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/1_train_1000.tsv
