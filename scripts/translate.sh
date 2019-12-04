#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="save/"
SOURCE_PATH="/home/zzheng/data/mt/IWSLT15.zh-en/docs_data/tst.zh.doc.bpe.20"
REFERENCE_PATH="/home/zzheng/data/mt/IWSLT15.zh-en/sents_data/tst.zh.norm.tok"
SAVEDIR="./results"
SAVETO="$SAVEDIR/trans.txt"
mkdir -p $SAVEDIR

python -m src.bin.translate \
    --model_name $MODEL_NAME \
    --source_path ${SOURCE_PATH} \
    --model_path $MODEL_NAME \
    --config_path "./save/configs.yaml" \
    --batch_size 1 \
    --beam_size 5 \
    --saveto $SAVETO \
    --use_gpu

sacrebleu -lc -tok none $REFERENCE_PATH < $SAVETO