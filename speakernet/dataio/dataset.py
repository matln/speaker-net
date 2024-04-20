# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import yaml
import math
import torch
import random
import logging
import torchaudio
import numpy as np
from rich import print
from copy import deepcopy
from typing import Union, Optional
from torch.utils.data import Dataset

import sys

sys.path.insert(0, "./")

import speakernet.utils.utils as utils
from speakernet.utils.rich_utils import track
from speakernet.preprocessing.augmentation import SpeechAug

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


# Relation: features -> chunk-egs-mapping-file -> chunk-egs -> bunch(dataloader+bunch) => trainer


class WaveDataset(Dataset):
    """ """

    def __init__(
        self,
        egs_csv: str,
        duration: float,
        samplerate: int = 16000,
        frame_overlap: float = 0.015,
        random_segment: bool = False,
        replacements: Union[dict, str] = {},
        delimiter: str = ",",
        repl_field: str = "wav_path",
        val: bool = False,
        num_targets: int = 0,
        aug: bool = False,
        aug_conf: str = "",
    ):
        """
        @egs_csv:
            utt_id:str  wav_path:str duration:float  start_position:int  end_position:int  spk_label:int

        Other option
        """
        self.duration = duration
        self.samplerate = samplerate
        self.frame_overlap_size = int(frame_overlap * samplerate)
        self.chunk_size = int(self.duration * samplerate) + self.frame_overlap_size
        self.random_segment = random_segment
        self.num_targets = num_targets

        assert egs_csv != "" and egs_csv is not None
        self.data = utils.load_data_csv(
            egs_csv, replacements, repl_field=repl_field, delimiter=delimiter
        )
        self.data_ids = list(self.data.keys())

        if (
            "start_position" in self.data[self.data_ids[0]].keys()
            and "end_position" in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = True
        elif (
            "start_position" not in self.data[self.data_ids[0]].keys()
            and "end_position" not in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = False
        else:
            raise TypeError(
                "Expected both start-position and end-position are exist in {}.".format(egs_csv)
            )

        self.worker_state = {}
        self.val = val

        # Augmentation.
        if aug:
            # self.aug = AugmentOnline(aug, aug_params)
            with open(aug_conf, "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
            self.aug = SpeechAug(**speech_aug_conf)
        else:
            self.aug = None

    def _load_chunk(self, data_point: dict, speed_factor: Optional[float] = None):
        wav_size = int(data_point["duration"] * self.samplerate)
        if wav_size < self.chunk_size:
            # logger.warning(f"wav_size {wav_size} < self.chunk_size {self.chunk_size}")
            pass

        if self.chunk_position:
            wav_path = data_point["wav_path"].split(" ")
            if self.random_segment:
                if len(wav_path) == 1:
                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size
                    sig, fs = utils.load_wavs(data_point["wav_path"], str(start), str(stop))
                else:
                    every_size = [int(x) for x in data_point["wav_size"].split(" ")]
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)

                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size

                    for i, item in enumerate(cumsum_size):
                        if start - item < 0:
                            start_position = "{}_{}".format(i - 1, start - cumsum_size[i - 1])
                            break
                    for i, item in enumerate(cumsum_size):
                        if stop - item <= 0:
                            end_position = "{}_{}".format(i - 1, stop - cumsum_size[i - 1])
                            break

                    sig, fs = utils.load_wavs(data_point["wav_path"], start_position, end_position)
            else:
                sig, fs = utils.load_wavs(
                    data_point["wav_path"], data_point["start_position"], data_point["end_position"]
                )
        else:
            # Custom valset
            assert self.aug is None
            sig, fs = utils.load_wavs(data_point["wav_path"])

        if self.aug is not None:
            sig, label_multiplier = self.aug(sig, speed_factor)
        else:
            label_multiplier = 0

        # 1. self.chunk_position is True (self.aug may be None), but wav_size < self.chunk_size
        # 2. sig.size(1) == self.chunk_size, but SpeedPerturb or TempoPerturb in self.aug change the sig length
        if self.chunk_position:
            sig = self._pad_or_truncate(sig)

        return sig, label_multiplier

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        chunk, label_multiplier = self._load_chunk(data_point)

        label = int(data_point["spk_label"])
        label = label + label_multiplier * self.num_targets

        if not self.val:
            # state will be omitted when num_workers=0, i.e., SingleProcessDataLoader
            state = self._get_random_state()
            return chunk.squeeze(0), label, state
        else:
            return chunk.squeeze(0), label

    def _pad_or_truncate(self, sig: torch.Tensor) -> torch.Tensor:
        if sig.size(1) > self.chunk_size:
            start = random.randint(0, sig.size(1) - self.chunk_size)
            sig = sig[:, start : start + self.chunk_size]
        else:
            pad_warp_num = self.chunk_size // sig.size(1)
            pad_size = self.chunk_size % sig.size(1)
            cat_list = [sig for _ in range(pad_warp_num)]
            if pad_size != 0:
                pad_start = random.randint(0, sig.size(1) - pad_size)
                pad_chunk = sig[:, pad_start : pad_start + pad_size]
                cat_list.append(pad_chunk)
            sig = torch.cat(cat_list, dim=1)
        return sig

    def _get_random_state(self):
        if torch.utils.data.get_worker_info() is not None:
            worker_id = torch.utils.data.get_worker_info().id
            np_state = np.random.get_state()
            random_state = random.getstate()
            torch_state = torch.get_rng_state()
            worker_state = {
                "worker_id": worker_id,
                "np_state": np_state,
                "random_state": random_state,
                "torch_state": torch_state,
            }
            if self.aug is not None:
                worker_state["aug_state"] = self.aug.get_aug_state()
            return worker_state
        else:
            # SingleProcessing
            if self.aug is not None:
                self.worker_state["aug_state"] = self.aug.get_aug_state()
            return {}

    def __len__(self):
        return len(self.data_ids)


class PairedWaveDataset(WaveDataset):
    """
    Utterances within a speaker are organized into pairs, and then mini-batches are generated based on the pairs.
    """

    def __init__(
        self,
        egs_csv: str,
        chunk2lang: str,
        duration: float,
        samplerate: int = 16000,
        frame_overlap: float = 0.015,
        random_segment: bool = False,
        replacements: Union[dict, str] = {},
        delimiter: str = ",",
        repl_field: str = "wav_path",
        num_targets: int = 0,
        p: float = 0.5,
        aug: bool = False,
        aug_conf: str = "",
    ):
        """
        egs_csv (str): Path to the .csv file of samples. csv format:
            utt_id wav_path duration start_position end_position spk_label
        chunk2lang (str):
        duration (float): Length of speech segment, e.g., 2s
        samplerate (int): Sample rate of audio.
        frame_overlap (float): Frame shift.
        random_segment (bool): Whether to randomly select fixed-length chunks from an utterance.
        """
        self.duration = duration
        self.samplerate = samplerate
        self.frame_overlap_size = int(frame_overlap * samplerate)
        self.chunk_size = int(self.duration * samplerate) + self.frame_overlap_size
        self.random_segment = random_segment
        self.num_targets = num_targets
        self.p = p

        assert egs_csv != "" and egs_csv is not None
        self.data = utils.load_data_csv(
            egs_csv, replacements, repl_field=repl_field, delimiter=delimiter
        )
        self.data_ids = list(self.data.keys())

        self.chunk2lang = utils.read_dict(chunk2lang)
        self.spk2chunks = {}
        self.spk2langs2chunks = {}
        for key, value in self.data.items():
            spk = value["spk_label"]
            self.spk2chunks.setdefault(spk, []).append(key)
            self.spk2langs2chunks.setdefault(spk, {}).setdefault(self.chunk2lang[key], []).append(
                key
            )
        for spk in self.spk2chunks:
            assert len(self.spk2chunks[spk]) > 1

        assert (
            "start_position" in self.data[self.data_ids[0]].keys()
            and "end_position" in self.data[self.data_ids[0]].keys()
        )
        self.chunk_position = True

        self.worker_state = {}

        # Augmentation.
        if aug:
            # self.aug = AugmentOnline(aug, aug_params)
            with open(aug_conf, "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
            self.aug = SpeechAug(**speech_aug_conf)
            self.label_multipler2factor = {0: 1.0, 1: 0.9, 2: 1.1}
        else:
            self.aug = None

    def __getitem__(self, index):
        data_id1 = self.data_ids[index]
        data_point1 = self.data[data_id1]
        label = data_point1["spk_label"]
        lang = self.chunk2lang[data_id1]
        lang2chunks = self.spk2langs2chunks[label]
        same_lang_chunks = lang2chunks[lang]
        diff_lang_chunks = [
            value for key, values in lang2chunks.items() if key != lang for value in values
        ]

        if (
            random.random() > self.p and same_lang_chunks != [data_id1]
            or len(diff_lang_chunks) == 0
        ):
            weights = [1 / (len(same_lang_chunks) - 1)] * len(same_lang_chunks)
            weights[same_lang_chunks.index(data_id1)] = 0
            data_id2 = random.choices(same_lang_chunks, weights=weights, k=1)[0]
            assert data_id1 != data_id2
        else:
            data_id2 = random.choice(diff_lang_chunks)

        data_point2 = self.data[data_id2]

        sig1, label_multiplier = self._load_chunk(data_point1)
        speed_factor = self.label_multipler2factor[label_multiplier]
        sig2, label_multiplier2 = self._load_chunk(data_point2, speed_factor)

        label = int(label) + label_multiplier * self.num_targets
        sig = torch.cat((sig1, sig2), dim=0)
        assert sig.size(0) == 2

        # state will be omitted when num_workers=0, i.e., SingleProcessDataLoader
        state = self._get_random_state()
        return sig, torch.LongTensor([label] * 2), state


class SpeakerAwareWaveDataset(WaveDataset):
    """
    When performing speed perturbation, it is inconvenient to obtain all spk labels
    for the SpeakerAwareSampler in advance. Thus we split the speed perturbation and
    other augmentation methods, and perform speed perturbation according to the
    label multiplier derived by index.
    """

    def __init__(
        self,
        egs_csv: str,
        duration: float,
        samplerate: int = 16000,
        frame_overlap: float = 0.015,
        random_segment: bool = False,
        replacements: Union[dict, str] = {},
        delimiter: str = ",",
        repl_field: str = "wav_path",
        val: bool = False,
        num_targets: int = 0,
        aug: bool = False,
        aug_conf: str = "",
    ):
        """
        @egs_csv:
            utt_id: str  wav_path: str duration: float  start_position:int  end_position:int  spk_label:int

        Other option
        """
        self.duration = duration
        self.samplerate = samplerate
        self.frame_overlap_size = int(frame_overlap * samplerate)
        self.chunk_size = int(self.duration * samplerate) + self.frame_overlap_size
        self.random_segment = random_segment
        self.num_targets = num_targets

        assert egs_csv != "" and egs_csv is not None
        self.data = utils.load_data_csv(
            egs_csv, replacements, repl_field=repl_field, delimiter=delimiter
        )
        self.data_ids = list(self.data.keys())

        if (
            "start_position" in self.data[self.data_ids[0]].keys()
            and "end_position" in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = True
        elif (
            "start_position" not in self.data[self.data_ids[0]].keys()
            and "end_position" not in self.data[self.data_ids[0]].keys()
        ):
            self.chunk_position = False
        else:
            raise TypeError(
                "Expected both start-position and end-position are exist in {}.".format(egs_csv)
            )

        self.worker_state = {}
        self.val = val
        self.data_lens = len(self.data_ids)

        # Augmentation.
        # It is assumed that speed perturbation is performed prior to other augmentation methods
        # and the mode is chain.
        if aug:
            with open(aug_conf, "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                if (
                    speech_aug_conf["mod"] == "chain"
                    and speech_aug_conf["aug_classes"][0]["aug_type"] == "Speed"
                ):
                    _ = speech_aug_conf["aug_classes"].pop(0)
                    speed_up_conf = [
                        {
                            "aug_name": "speed_up",
                            "aug_type": "Speed",
                            "perturb_prob": 1.0,
                            "sample_rate": 16000,
                            "speeds": [1.1],
                        }
                    ]
                    speed_down_conf = [
                        {
                            "aug_name": "speed_down",
                            "aug_type": "Speed",
                            "perturb_prob": 1.0,
                            "sample_rate": 16000,
                            "speeds": [0.9],
                        }
                    ]
                    self.speed_up = SpeechAug(aug_classes=speed_up_conf, mod="chain")
                    self.speed_down = SpeechAug(aug_classes=speed_down_conf, mod="chain")
                else:
                    self.speed_up = None
                    self.speed_down = None
            self.aug = SpeechAug(**speech_aug_conf)
        else:
            self.aug = None
            self.speed_up = None
            self.speed_down = None

    def __getitem__(self, index):
        label_multiplier = index // self.data_lens
        index = index % self.data_lens

        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        wav_size = int(data_point["duration"] * self.samplerate)
        if wav_size < self.chunk_size:
            # logger.warning(f"wav_size {wav_size} < self.chunk_size {self.chunk_size}")
            pass

        if self.chunk_position:
            wav_path = data_point["wav_path"].split(" ")
            if self.random_segment:
                if len(wav_path) == 1:
                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size
                    sig, fs = utils.load_wavs(data_point["wav_path"], str(start), str(stop))
                else:
                    every_size = [int(x) for x in data_point["wav_size"].split(" ")]
                    cumsum_size = list(np.cumsum(every_size))
                    cumsum_size.insert(0, 0)

                    start = random.randint(0, wav_size - self.chunk_size)
                    stop = start + self.chunk_size

                    for i, item in enumerate(cumsum_size):
                        if start - item < 0:
                            start_position = "{}_{}".format(i - 1, start - cumsum_size[i - 1])
                            break
                    for i, item in enumerate(cumsum_size):
                        if stop - item <= 0:
                            end_position = "{}_{}".format(i - 1, stop - cumsum_size[i - 1])
                            break

                    sig, fs = utils.load_wavs(data_point["wav_path"], start_position, end_position)
            else:
                sig, fs = utils.load_wavs(
                    data_point["wav_path"], data_point["start_position"], data_point["end_position"]
                )
        else:
            # Custom valset
            assert self.aug is None
            sig, fs = utils.load_wavs(data_point["wav_path"])

        label = int(data_point["spk_label"])

        if label_multiplier == 1:
            sig, multiplier = self.speed_down(sig)
        elif label_multiplier == 2:
            sig, multiplier = self.speed_up(sig)

        if self.aug is not None:
            sig, multiplier = self.aug(sig)
            assert multiplier == 0

        label = label + label_multiplier * self.num_targets

        # 1. self.chunk_position is True (self.aug may be None), but wav_size < self.chunk_size
        # 2. sig.size(1) == self.chunk_size, but SpeedPerturb or TempoPerturb in self.aug change the sig length
        if self.chunk_position:
            if sig.size(1) > self.chunk_size:
                start = random.randint(0, sig.size(1) - self.chunk_size)
                sig = sig[:, start : start + self.chunk_size]
            else:
                pad_warp_num = self.chunk_size // sig.size(1)
                pad_size = self.chunk_size % sig.size(1)
                cat_list = [sig for _ in range(pad_warp_num)]
                if pad_size != 0:
                    pad_start = random.randint(0, sig.size(1) - pad_size)
                    pad_chunk = sig[:, pad_start : pad_start + pad_size]
                    cat_list.append(pad_chunk)
                sig = torch.cat(cat_list, dim=1)

        if not self.val:
            # state will be omitted when num_workers=0, i.e., SingleProcessDataLoader
            state = self._get_random_state()
            return sig.squeeze(0), label, state
        else:
            return sig.squeeze(0), label

    def __len__(self):
        if self.speed_up is not None:
            return self.data_lens * 3
        else:
            return self.data_lens


if __name__ == "__main__":
    from speakernet.dataio.collate import default_collate
    from speakernet.dataio.sampler import SpeakerAwareSampler
    from speakernet.dataio.dataloader import SaveableDataLoader, worker_init_fn

    trainset = SpeakerAwareWaveDataset(
        "/data/lijianchen/workspace/sre/CDMA+/exp/egs/unlabeled_waveform_2s/train.egs.csv",
        2,
        samplerate=16000,
        num_targets=1806,
        aug=True,
        aug_conf="/data/lijianchen/workspace/sre/CDMA+/hparams/speech_aug_chain.yaml",
    )

    spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]
    spk_labels.extend(
        [int(item["spk_label"]) + trainset.num_targets for item in trainset.data.values()]
    )
    spk_labels.extend(
        [int(item["spk_label"]) + 2 * trainset.num_targets for item in trainset.data.values()]
    )

    generator = torch.Generator()
    train_sampler = SpeakerAwareSampler(
        spk_labels,
        num_samples_cls=4,
        generator=generator,
    )

    train_loader = SaveableDataLoader(
        trainset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        collate_fn=default_collate,
        generator=generator,
    )

    for i, batch in enumerate(train_loader):
        data, targets, state = batch
        print(targets)
        if i == 100:
            break
