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
# time=2023-01-01_01:18:12
# time=2023-02-21_12:20:38
# time=2023-02-21_14:52:52
# time=2023-02-22_22:46:37
time=2023-12-13_22:07:05
extracted_epochs=""

trials="voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean"
# trials="voxceleb1-O"
# trials="voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean voxsrc2021-val voxsrc2022-dev"

exp_dir="exp/${model}/${time}" && [ ! -d $exp_dir ] && echo "$exp_dir doesn't exist" && exit 1

submean="false"    # "true", "false"
score_norm_method=""    # "asnorm", ""
score_calibration="false"    # "true", "false"

python3 ${speakernet}/pipelines/score.py \
    --exp-dir=$exp_dir \
    --epochs="$extracted_epochs" \
    --trials="$trials" \
    --evalset="voxsrc2022_val" \
    --submean="$submean" \
    --submean-set="voxceleb2_dev" \
    --score-norm-method="$score_norm_method" \
    --cohort-set="voxceleb2_dev" \
    --average-cohort="true" \
    --top-n=300 \
    --score-calibration="$score_calibration" \
    --cali-dev-set="calibration" \
    --cali-dev-trials="cali_trials" \
    --quality-measures="duration imposter" \
    --return-thresh="false" \
    --ptarget=0.01 \
    --force="true"
