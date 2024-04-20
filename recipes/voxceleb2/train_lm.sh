#!/bin/bash

### Start
. ../../path.sh

# model="resnet34-32_lm"
# hparam_file="resnet34-32_lm.yaml"
model="resnet34-64_lm"
hparam_file="resnet34-64_lm.yaml"
# model="ecapa-tdnn-c512_lm"
# hparam_file="ecapa-tdnn-c512_sgd_lm.yaml"
# model="ecapa-tdnn-c1024_lm"
# hparam_file="ecapa-tdnn-c1024_sgd_lm.yaml"
# model="repvgg-B1_lm"
# hparam_file="repvgg-B1_lm.yaml"

resume_training="false"
debug="false"

# Specify the time to resume training
time=

[[ $time != "" ]] && resume_training="true" && export speakernet=exp/${model}/${time}/backup/speakernet
: =${time:=$(date "+%Y-%m-%d_%H:%M:%S")}
mkdir -p tmp
# echo $time | tee tmp/train.timestamp
echo $time > tmp/train0.timestamp

${speakernet}/pipelines/launcher.sh --pdb "false" \
    ${speakernet}/pipelines/train.py --hparams-file=hparams/${hparam_file} \
        --model-dir="$model" \
        --gpu-id="0 1" \
        --debug="$debug" \
        --train-time-string="$time" \
        --resume-training=$resume_training
