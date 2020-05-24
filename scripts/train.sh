#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

MODEL_NAME="mdl"

CONFIG="./configs/news_docEnc+docDec.yaml"

ROOT="."
LOG_PATH="${ROOT}/log/"
SAVETO="${ROOT}/save/"
VALID_PATH="${ROOT}/valid/"

mkdir -p $SAVETO
cp $CONFIG $SAVETO/configs.yaml

python3 -m src.bin.train \
    --model_name ${MODEL_NAME} \
    --config_path ${CONFIG} \
    --log_path ${LOG_PATH} \
    --saveto ${SAVETO} \
	--valid_path ${VALID_PATH} \
    --use_gpu
    #--reload \
