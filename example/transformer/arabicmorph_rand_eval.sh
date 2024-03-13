#!/usr/bin/bash

python src/train_rand_eval.py \
    --training_method "rand" \
    --database "arabicmorphbaseline" \
    --prefix "rand_am0" \
    --start_cycle 6 \
    --train_batch_size 400 \
    --test_batch_size 6 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 1000 \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/0_tune_13171.tsv \
    data/arabicmorph/active/0_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/0_train_1000.tsv
