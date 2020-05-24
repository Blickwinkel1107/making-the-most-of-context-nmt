#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

MODEL_NAME="mdl"
MODEL_PATH="./L67.30B21.20.para"
SOURCE_PATH="/home/user_data55/yuex/News/docs_data/tst.en.doc.bpe.20"
REFERENCE_PATH="/home/user_data55/yuex/News/sents_data/tst.de.norm.tok"
SAVEDIR="./results"
SAVETO="$SAVEDIR/trans_a0.5b5.txt"
mkdir -p $SAVEDIR

python3 -m src.bin.translate \
    --model_name ${MODEL_NAME} \
    --source_path ${SOURCE_PATH} \
    --model_path ${MODEL_PATH} \
    --config_path "./save/configs.yaml" \
    --batch_size 1 \
    --beam_size 5 \
	--alpha 0.5 \
    --saveto $SAVETO \
    --use_gpu

sacrebleu -w 2 -lc -tok none $REFERENCE_PATH < $SAVETO
