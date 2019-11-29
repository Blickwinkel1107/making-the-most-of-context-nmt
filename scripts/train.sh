#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME=
CONFIG="./configs/d2d_iwslt15_zh2en.yaml"
SAVETO="./save/"

mkdir -p $SAVETO
cp $CONFIG $SAVETO/configs.yaml

python -m src.bin.train \
    --model_name ${MODEL_NAME} \
    --reload \
    --config_path ${CONFIG} \
    --log_path "/home/zzheng/experiments/njunmt/docmt/tblogs/${MODEL_NAME}" \
    --saveto ${SAVETO} \
    --use_gpu
    # --pretrain_path "/home/zzheng/experiments/njunmt/docmt/s2s.best.bleu11.6loss75" \
