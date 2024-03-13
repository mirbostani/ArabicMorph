#!/bin/bash

python src/train_al.py \
    --prefix "am0" \
    --start_cycle 0 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 250 \
    --sampling_method "max_entropy" \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/0_tune_13171.tsv \
    data/arabicmorph/active/0_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/0_train_1000.tsv