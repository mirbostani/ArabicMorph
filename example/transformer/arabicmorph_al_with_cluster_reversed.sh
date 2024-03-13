#!/bin/bash

python src/train_al.py \
    --training_method "al" \
    --database "arabicmorphbaselinereversed" \
    --prefix "al_8c_rev_am1" \
    --start_cycle 0\
    --train_batch_size 400 \
    --test_batch_size 6 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 250 \
    --num_clusters 8 \
    --num_pre_clustering_samples 1000 \
    --clustering_method "kmeans-chars2vec" \
    --kmeans_init_method "k-means++" \
    --sampling_method "max_entropy" \
    --chars2vec_emb_dim 50 \
    --pca_n_components 3 \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/1_tune_13171.tsv \
    data/arabicmorph/active/1_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/1_train_1000.tsv