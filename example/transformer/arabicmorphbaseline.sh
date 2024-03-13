#!/bin/bash

# $ bash arabicmorphbaseline.sh train
# $ bash arabicmorphbaseline.sh test

mode=$1
task=arabicmorphbaseline-5000-0 # modify based on the training file
database=arabicmorphbaseline
arch=transformer

lr=0.001
scheduler=warmupinvsqr
max_steps=20000
warmup=4000
beta2=0.98       # 0.999
label_smooth=0.1 # 0.0
total_eval=50
# bs=400 # 256
bs=1

# transformer
layers=4
hs=1024
embed_dim=256
nb_heads=4
dropout=${2:-0.3}

data_dir="data/arabicmorph/Egy_Data/ECAL/baseline_data"
ckpt_dir="checkpoints/transformer"

train_mode_train_file="0_train_5000.tsv"
train_mode_dev_file="0_tune_5000.tsv"
train_mode_test_file="test.tsv" 
train_mode_model_file_prefix="model"

test_mode_train_file="$train_mode_train_file"
test_mode_dev_file="$train_mode_dev_file"
test_mode_test_file="dev_oov.tsv" # test.tsv | test_oov.tsv | dev.tsv | dev_oov.tsv
test_mode_prediction_file_prefix="model-eval"
test_mode_model="model.nll_0.7451.acc_93.0.dist_0.132.epoch_1518"

train() {
    time python src/train.py \
        --dataset $database \
        --train $data_dir/$train_mode_train_file \
        --dev $data_dir/$train_mode_dev_file \
        --test $data_dir/$train_mode_test_file \
        --model $ckpt_dir/$arch/$task/$train_mode_model_file_prefix \
        --embed_dim $embed_dim \
        --src_hs $hs \
        --trg_hs $hs \
        --dropout $dropout \
        --nb_heads $nb_heads \
        --label_smooth $label_smooth \
        --total_eval $total_eval \
        --src_layer $layers \
        --trg_layer $layers \
        --max_norm 1 \
        --lr $lr \
        --shuffle \
        --arch $arch \
        --gpuid 0 \
        --estop 1e-8 \
        --bs $bs \
        --max_steps $max_steps \
        --scheduler $scheduler \
        --warmup_steps $warmup \
        --cleanup_anyway \
        --beta2 $beta2 \
        --bestacc
}

test() {
    time python src/test.py \
        --dataset $database \
        --train $data_dir/$test_mode_train_file \
        --dev $data_dir/$test_mode_dev_file \
        --test $data_dir/$test_mode_test_file \
        --model $ckpt_dir/$arch/$task/$test_mode_prediction_file_prefix \
        --load $ckpt_dir/$arch/$task/$test_mode_model \
        --embed_dim $embed_dim \
        --src_hs $hs \
        --trg_hs $hs \
        --dropout $dropout \
        --nb_heads $nb_heads \
        --label_smooth $label_smooth \
        --total_eval $total_eval \
        --src_layer $layers \
        --trg_layer $layers \
        --max_norm 1 \
        --lr $lr \
        --shuffle \
        --arch $arch \
        --gpuid 0 \
        --estop 1e-8 \
        --bs $bs \
        --max_steps $max_steps \
        --scheduler $scheduler \
        --warmup_steps $warmup \
        --cleanup_anyway \
        --beta2 $beta2 \
        --bestacc
}

if [ $1 == "train" ]
then
    train
else
    test
fi