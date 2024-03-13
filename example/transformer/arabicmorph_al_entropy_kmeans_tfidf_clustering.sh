#!/bin/bash
# # export MKL_SERVICE_FORCE_INTEL=1 # NumPy using OpenMP
# export MKL_THREADING_LAYER=GNU # PyTorch

python src/train_al.py \
    --prefix "a_8c_1500_tfidf_am1" \
    --start_cycle 0 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 250 \
    --num_clusters 8 \
    --num_pre_clustering_samples 1500 \
    --clustering_method "kmeans-tfidf" \
    --kmeans_init_method "k-means++" \
    --tfidf_ngram_range 3 \
    --sampling_method "max_entropy" \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/1_tune_13171.tsv \
    data/arabicmorph/active/1_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/1_train_1000.tsv