#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

export MODEL_NAME="D2D"

SOURCE_PATH="/home/zzheng/data/mt/IWSLT15.zh-en/docs_data/dev.zh.doc.bpe.20"
SAVEDIR="./results"
SAVETO="$SAVEDIR/trans.txt"
mkdir -p $SAVEDIR

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path ${SOURCE_PATH} \
    --model_path "./save/$MODEL_NAME.best.final" \
    --config_path "./save/configs.yaml" \
    --batch_size 1 \
    --beam_size 5 \
    --saveto $SAVETO \
    --use_gpu

sacrebleu -lc -tok none $REFERENCE_PATH < $SAVETO