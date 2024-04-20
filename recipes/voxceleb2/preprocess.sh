#!/bin/bash

### Start 
. ../../path.sh

# --------------------------------------------------------------------------------------------- #

# ==> Make sure the audio datasets (voxceleb1, voxceleb2, RIRS and Musan) have been downloaded by yourself.
voxceleb1_path=/home/lijianchen/pdata/VoxCeleb1_v2
voxceleb2_path=/home/lijianchen/pdata/VoxCeleb2
voxsrc2022_path=/home/lijianchen/pdata/voxsrc/voxsrc2022/Track1_2/VoxSRC2022_dev

stage=0
endstage=1

# --------------------------------- dataset preparation ------------------------------- #
if [[ $stage -le 0 && 0 -le $endstage ]]; then
	# Prepare the data/voxceleb1_train, data/voxceleb1_test and data/voxceleb2_train.
	local/make_voxceleb1_v2.pl $voxceleb1_path dev data/voxceleb1_dev
	local/make_voxceleb1_v2.pl $voxceleb1_path test data/voxceleb1_test
	local/make_voxceleb2.pl $voxceleb2_path dev data/voxceleb2_dev
	# python local/make_voxconverse.py --wav-dir=$voxsrc2022_path --out-dir=data

	# Combine testset voxceleb1 = voxceleb1_dev + voxceleb1_test
	combine_data.sh data/voxceleb1 data/voxceleb1_dev data/voxceleb1_test
	# combine_data.sh data/voxsrc2022_val data/voxceleb1 data/voxconverse

	# Get trials
	# ==> Make sure the original trials is in data/voxceleb1.
	local/get_trials.sh --dir data/voxceleb1
	# local/get_trials.sh --dir data/voxsrc2022_val \
    #     --tasks "voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean voxsrc2021-val voxsrc2022-dev"

    python3 tools/make_utt2dur.py --data-dir=data/voxceleb2_dev

    # Used for training score calibration model
    python3 local/make_calib_trials.py \
        --data-dir=data/voxceleb2_dev \
        --out-dir=data/calibration

    cat data/calibration/trials/calibration | awk '{print $1}' > utts.lst
    cat data/calibration/trials/calibration | awk '{print $2}' >> utts.lst

    filter_data_dir.sh --check false data/voxceleb2_dev utts.lst data/calibration

    # Duration-based quality measures
    mkdir -p data/calibration/quality_measures
    # <enroll> <test> <low_quality> <high_quality>
    awk 'NR==FNR{
    utt2dur[$1]=$2
    }NR>FNR{
    if(utt2dur[$1]<utt2dur[$2]){
        print $1, $2, utt2dur[$1], utt2dur[$2]
    }else{
    print $1, $2, utt2dur[$2], utt2dur[$1]}
    }' data/calibration/utt2dur data/calibration/trials/cali_trials > data/calibration/quality_measures/duration

    python3 tools/make_utt2dur.py --data-dir=data/voxceleb1
    mkdir -p data/voxceleb1/quality_measures
    for trials in voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean; do
        awk 'NR==FNR{
        utt2dur[$1]=$2
        }NR>FNR{
        if(utt2dur[$1]<utt2dur[$2]){
            print $1, $2, utt2dur[$1], utt2dur[$2]
        }else{
        print $1, $2, utt2dur[$2], utt2dur[$1]}
        }' data/voxceleb1/utt2dur data/voxceleb1/trials/$trials > data/voxceleb1/quality_measures/${trials}_dur
    done
    
    # python3 tools/make_utt2dur.py --data-dir=data/voxsrc2022_val
    # mkdir -p data/voxsrc2022_val/quality_measures
    # for trials in voxceleb1-O voxceleb1-O-clean voxceleb1-E voxceleb1-E-clean voxceleb1-H voxceleb1-H-clean voxsrc2021-val voxsrc2022-dev; do
    #     awk 'NR==FNR{
    #     utt2dur[$1]=$2
    #     }NR>FNR{
    #     if(utt2dur[$1]<utt2dur[$2]){
    #         print $1, $2, utt2dur[$1], utt2dur[$2]
    #     }else{
    #     print $1, $2, utt2dur[$2], utt2dur[$1]}
    #     }' data/voxsrc2022_val/utt2dur data/voxsrc2022_val/trials/$trials > data/voxsrc2022_val/quality_measures/${trials}_dur
    # done

    python3 tools/make_utt2dur.py --data-dir=data/voxceleb1_test


    # # Get the language labels of voxceleb2_dev
    # wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data_workshop_2021/lang_vox2_final.csv 
    # cat lang_vox2_final.csv | sed 's/\//-/g' | sed 's/,/ /g' | \
    #     awk 'NR==FNR{
    #         utt=substr($1, 1, length($1)-4)
    #         utts[utt]=$2
    #     }
    #     NR>FNR{
    #         if($1 in utts){
    #             print $1, utts[$1] 
    #         }
    #         else{
    #             print $1, "other"
    #         }
    #     }' - data/voxceleb2_dev/utt2spk > data/voxceleb2_dev/utt2lang
    # \rm lang_vox2_final.csv
fi

# ------------------------------ preprocess to generate chunks ------------------------- #
if [[ $stage -le 1 && 1 -le $endstage ]]; then
    traindata=data/voxceleb2_dev
    seed=1024
    egsdir=exp/egs/waveform_2s

    ${speakernet}/pipelines/prepare_train_csv.sh \
        --seed ${seed} \
        --val-num-spks 2048 \
        --amp_th 0.0005 \
        ${traindata} ${egsdir}

    egsdir=exp/egs/waveform_6s

    ${speakernet}/pipelines/prepare_train_csv.sh \
        --seed ${seed} \
        --duration 6.0 \
        --min-duration 2.0 \
        --drop-last-duration 0.6 \
        --val-num-spks 2048 \
        --amp-th 0.0005 \
        ${traindata} ${egsdir}

    # # Get chunk2lang
    # cat ${egsdir}/train.egs.csv | sed 's/,/ /g' | \
    #     awk 'NR==FNR{
    #         utt2lang[$1]=$2
    #     }
    #     NR>FNR&&FNR>1{
    #         split($1, a, "-")
    #         sub_str=a[length(a)-2]"-"a[length(a)-1]"-"a[length(a)]
    #         utt=$1
    #         sub(sub_str, a[length(a)-2]"-"a[length(a)-1], utt)
    #         print $1, utt2lang[utt]
    #     }' data/voxceleb2_dev/utt2lang - > ${egsdir}/chunk2lang
fi

# ------------------------------ Prepare augmentation csv file ------------------------- #
if [[ $stage -le 2 && 2 -le $endstage ]]; then
    python3 ${speakernet}/pipelines/prepare_aug_csv.py \
        --openrir-folder=/data/corpus/rirs_noises \
        --musan-folder=/data/corpus/MUSAN \
        --save-folder=/home/lijianchen/pdata/noise_2.015s \
        --max-noise-len=2.015 \
        --overlap=0 \
        --force-clear=true \
        exp/aug_csv

    python3 ${speakernet}/pipelines/prepare_aug_csv.py \
        --openrir-folder=/data/corpus/rirs_noises \
        --musan-folder=/data/corpus/MUSAN \
        --save-folder=/home/lijianchen/pdata/noise_6.015s \
        --max-noise-len=6.015 \
        --overlap=0 \
        --force-clear=true \
        exp/aug_csv_6.015
fi
