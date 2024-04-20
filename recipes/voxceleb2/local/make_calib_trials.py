import os
import sys
import random
import argparse
import warnings
from rich.progress import Progress
from itertools import combinations, product

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

from speakernet.utils.utils import read_scp, read_trials
from speakernet.utils.rich_utils import track, progress_columns

parser = argparse.ArgumentParser(description="")
parser.add_argument("--data-dir", type=str, default="",
                    help="The directory that should contain wav.scp utt2spk spk2utt utt2dur")
parser.add_argument("--out-dir", type=str, default="", help="")
args = parser.parse_args()

random.seed(1024)

utt2dur = read_scp(f"{args.data_dir}/utt2dur", value_type="float")
spk2utt = read_scp(f"{args.data_dir}/spk2utt", multi_value=True)
spk2utt_short = {}
spk2utt_long = {}
for spk, utts in spk2utt.items():
    for utt in utts:
        if utt2dur[utt] > 2 and utt2dur[utt] < 6:
            spk2utt_short.setdefault(spk, []).append(utt)
        elif utt2dur[utt] >= 6:
            spk2utt_long.setdefault(spk, []).append(utt)

# num enroll speakers
num_spks = 5000
num_target_per_spk = 1
# num of nontarget speaker pairs
num_pairs = 5000
num_nontarget_per_pairs = 1

os.makedirs(f"{args.out_dir}/trials", exist_ok=True)
for (enroll_type, test_type) in [("short", "short"), ("short", "long"), ("long", "long")]:
    with open(f"{args.out_dir}/trials/{enroll_type}-{test_type}", "w") as fw:
        with Progress(*progress_columns) as progress:
            task = progress.add_task(f"working for {enroll_type}-{test_type} ...", total=num_spks)
            # target trials
            spks = list(spk2utt.keys())
            random.shuffle(spks)
            used_spks = 0
            for spk in spks:
                utts = eval(f"spk2utt_{enroll_type}[spk]")
                if enroll_type == test_type:
                    candidate_targets = list(combinations(utts, 2))
                else:
                    _utts = eval(f"spk2utt_{test_type}[spk]")
                    candidate_targets = list(product(utts, _utts))
                if len(candidate_targets) < num_target_per_spk:
                    continue
                targets = random.sample(candidate_targets, num_target_per_spk)
                for target in targets:
                    fw.write(f"{target[0]} {target[1]} target\n")
                used_spks += 1
                progress.update(task, advance=1)
                if used_spks >= num_spks:
                    break

        # nontarget trials
        with Progress(*progress_columns) as progress:
            task = progress.add_task(f"working for {enroll_type}-{test_type} ...", total=num_pairs)

            candidate_spk_pairs = list(combinations(spks, 2))
            random.shuffle(candidate_spk_pairs)
            used_pairs = 0
            for (enroll_spk, test_spk) in candidate_spk_pairs:
                enroll_utts = eval(f"spk2utt_{enroll_type}[enroll_spk]")
                test_utts = eval(f"spk2utt_{test_type}[test_spk]")
                if (
                    len(enroll_utts) < num_nontarget_per_pairs
                    or len(test_utts) < num_nontarget_per_pairs
                ):
                    continue
                enroll_utts = random.sample(enroll_utts, num_nontarget_per_pairs)
                test_utts = random.sample(test_utts, num_nontarget_per_pairs)
                for counter in range(num_nontarget_per_pairs):
                    fw.write(f"{enroll_utts[counter]} {test_utts[counter]} nontarget\n")
                used_pairs += 1
                progress.update(task, advance=1)
                if used_pairs >= num_pairs:
                    break


# cat data/calibration/trials/short-short | awk -F ' ' 'NR<5001{split($1, a, "-"); split($2, b, "-"); if(a[1]!=b[1]){print "error"}}'
# cat data/calibration/trials/short-short | awk -F ' ' 'NR>5000{split($1, a, "-"); split($2, b, "-"); if(a[1]==b[1]){print "error"}}'


# Combine trials
with open(f"{args.out_dir}/trials/cali_trials", "w") as fw:
    for (enroll_type, test_type) in [("short", "short"), ("short", "long"), ("long", "long")]:
        with open(f"{args.out_dir}/trials/{enroll_type}-{test_type}", "r") as fr:
            lines = fr.readlines()
            for line in lines:
                fw.write(line)
