# -*- coding:utf-8 -*-

import os
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument("--wav-dir", type=str, default="", help="")
parser.add_argument("--out-dir", type=str, default="./data", help="")

args = parser.parse_args()

data_dir = f"{args.out_dir}/voxconverse"
os.makedirs(data_dir, exist_ok=True)

f1 = open(f"{data_dir}/wav.scp", "w")
f2 = open(f"{data_dir}/utt2spk", "w")

for wav in os.listdir(args.wav_dir):
    utt = f"VoxSRC2022_dev-{wav[:-4]}"
    full_path = f"{args.wav_dir}/{wav}"
    f1.write(f"{utt} {full_path}\n")
    f2.write(f"{utt} {utt}\n")
f1.close()
f2.close()

if os.system(f"utt2spk_to_spk2utt.pl {data_dir}/utt2spk >{data_dir}/spk2utt") != 0:
    print(f"Error creating spk2utt file in directory {data_dir}")
os.system(f"env LC_COLLATE=C fix_data_dir.sh {data_dir}")
if os.system(f"env LC_COLLATE=C validate_data_dir.sh --no-text --no-feats {data_dir}") != 0:
    print(f"Error validating directory {data_dir}")
