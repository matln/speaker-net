#!/bin/bash
# Copyright 2022 Jianchen Li

set -e

stage=1
endstage=1

suffix=
limit_dur=60

# Get chunk egs
duration=2.0
min_duration=2.0
sample_type="sequential"  # sequential | speaker_balance | random_segment
chunk_num=-1
scale=1.5
overlap=0.1
frame_overlap=0.015
drop_last_duration=0.2
val_split_type="--total-spk"
val_num_spks=1024
val_chunk_num=2
val_sample_type="sequential" # With sampleSplit type [--total-spk] and sample type [every_utt], we will get enough spkers as more
                              # as possible and finally we get val_num_utts * val_chunk_num = 1024 * 2 = 2048 val chunks.
val_split_from_trainset="true"
seed=1024
amp_th=0.0005      # 50 / (1 << 15)
samplerate=16000

expected_files="utt2spk,spk2utt,wav.scp"

. parse_options.sh

if [[ $# != 2 && $# != 3 ]]; then
  echo "[exit] Num of parameters is not equal to 2 or 3"
  echo "usage:$0 <data-dir> <egs-dir>"
  exit 1
fi

# Key params
train_data=$1
egsdir=$2
val_data=$3

if [[ $stage -le 0 && 0 -le $endstage ]]; then
    echo "$0: stage 0"

    if [[ $suffix != "" ]]; then
        \rm -rf ${train_data}${suffix}
        cp -r ${train_data} ${train_data}${suffix}
        train_data=${train_data}${suffix}
    fi

    # Remove speakers with too few training samples
    ${speakernet}/pipelines/modules/remove_short_utt.sh --limit-dur $limit_dur \
        ${train_data} $(echo "$min_duration + $frame_overlap" | bc) || exit 1
fi

if [[ $stage -le 1 && 1 -le $endstage ]]; then
    echo "$0: stage 1"
    [ "$egsdir" == "" ] && echo "The egsdir is not specified." && exit 1

	# val: validation
	python3 ${speakernet}/pipelines/modules/get_chunk_csv.py \
		--duration=$duration \
		--min-duration=$min_duration \
		--sample-type=$sample_type \
		--chunk-num=$chunk_num \
		--scale=$scale \
		--overlap=$overlap \
		--frame-overlap=$frame_overlap \
		--drop-last-duration=$drop_last_duration \
		--val-split-type=$val_split_type \
		--val-num-spks=$val_num_spks \
		--val-chunk-num=$val_chunk_num \
		--val-sample-type=$val_sample_type \
        --val-dir="${val_data}" \
        --val-split-from-trainset=$val_split_from_trainset \
		--seed=$seed \
		--amp-th=$amp_th \
		--samplerate=$samplerate \
        --expected-files="$expected_files" \
		${train_data} ${egsdir} || exit 1
fi

exit 0
