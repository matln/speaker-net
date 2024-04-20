import os
import sys
import torchaudio
from rich.progress import track

sys.path.insert(0, "/data/lijianchen/workspace/sre/speechbrain_official")
from speechbrain.pretrained import EncoderClassifier

language_id = EncoderClassifier.from_hparams(source="TalTechNLP/voxlingua107-epaca-tdnn", savedir="tmp")

with open("data/voxceleb2_dev/wav.scp", "r") as fr, open("data/voxceleb2_dev/utt2lang", "w") as fw:
    lines = fr.readlines()
    for line in track(lines):
        utt, wav = line.split()
        signal, _ = torchaudio.load(wav)
        prediction =  language_id.classify_batch(signal)
        lang = prediction[3][0]
        fw.write(f"{utt} {lang}\n")
