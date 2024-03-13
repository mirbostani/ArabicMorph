#!/bin/bash

python src/train_al.py \
    --prefix "a_8c_c2v_am1" \
    --start_cycle 0 \
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