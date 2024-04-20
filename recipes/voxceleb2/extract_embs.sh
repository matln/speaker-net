#!/bin/bash

### Start
. ../../path.sh

# model="ecapa-tdnn-c512"
# model="ecapa-tdnn-c512_lm"
# model="ecapa-tdnn-c1024"
# model="ecapa-tdnn-c1024_lm"
# model="resnet34-32"
# model="resnet34-32_lm"
model="resnet34-64"
# model="resnet34-64_lm"
# model="repvgg-B1"
# model="repvgg-B1_lm"

# time=$(cat tmp/train.timestamp)
time=2023-12-13_22:07:05
extract_epochs=""

export speakernet=exp/${model}/${time}/backup/speakernet

python3 ${speakernet}/pipelines/extract_embeddings.py \
	--model-dir="$model" \
	--gpu-id="0 1" \
	--train-time-string="$time" \
	--extract-epochs="$extract_epochs" \
    --extract_data="voxsrc2022_val" \
    --lower-epoch=15
