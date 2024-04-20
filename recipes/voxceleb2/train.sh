#!/bin/bash

### Start
. ../../path.sh

# model="ecapa-tdnn-c512"
# hparam_file="ecapa-tdnn-c512_sgd.yaml"
# model="ecapa-tdnn-c1024"
# hparam_file="ecapa-tdnn-c1024_sgd.yaml"
# model="resnet101"
# hparam_file="resnet101.yaml"
# model="resnet34-32"
# hparam_file="resnet34-32.yaml"
model="resnet34-64"
hparam_file="resnet34-64.yaml"
# model="repvgg-B1"
# hparam_file="repvgg-B1.yaml"

resume_training="false"
debug="false"

# Specify the time to resume training
time=

[[ $time != "" ]] && resume_training="true" && export speakernet=exp/${model}/${time}/backup/speakernet
: =${time:=$(date "+%Y-%m-%d_%H:%M:%S")}
mkdir -p tmp
# echo $time | tee tmp/train.timestamp
echo $time > tmp/train1.timestamp

# multi-gpu:
#         --gpu-id="0 1" \
${speakernet}/pipelines/launcher.sh --pdb "false" \
    ${speakernet}/pipelines/train.py --hparams-file=hparams/${hparam_file} \
        --model-dir="$model" \
        --gpu-id="0" \
        --debug="$debug" \
        --train-time-string="$time" \
        --resume-training=$resume_training
