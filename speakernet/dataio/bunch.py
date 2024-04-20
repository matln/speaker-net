# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""

import os
import yaml
import torch
import logging
from rich import print
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import List, Optional, Union

import speakernet.utils.utils as utils
from speakernet.dataio.collate import default_collate
from speakernet.dataio.sampler import SpeakerAwareSampler
from speakernet.dataio.dataloader import SaveableDataLoader, worker_init_fn
from speakernet.dataio.dataset import (
    WaveDataset,
    PairedWaveDataset,
    SpeakerAwareWaveDataset
)

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)


class DataBunch:
    """DataBunch:(trainset,[val]).
    """

    def __init__(
        self,
        trainset: Union[WaveDataset, PairedWaveDataset, SpeakerAwareWaveDataset],
        val: Optional[WaveDataset] = None,
        batch_size=512,
        val_batch_size=512,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        prefetch_factor=2,
        drop_last=True,
        speaker_aware_sampling=False,
        paired_sampling=False,
        num_samples_cls=8,
        seed=1024,
    ):

        num_gpu = 1
        self.generator = torch.Generator()

        if utils.use_ddp():
            # The num_replicas/world_size and rank will be set automatically with DDP.
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                trainset, shuffle=shuffle, drop_last=drop_last, seed=seed
            )
            shuffle = False
            num_gpu = dist.get_world_size()
        elif speaker_aware_sampling:
            if trainset.speed_up is not None:
                spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]
                spk_labels.extend(
                    [
                        int(item["spk_label"]) + trainset.num_targets
                        for item in trainset.data.values()
                    ]
                )
                spk_labels.extend(
                    [
                        int(item["spk_label"]) + 2 * trainset.num_targets
                        for item in trainset.data.values()
                    ]
                )
            else:
                spk_labels = [int(item["spk_label"]) for item in trainset.data.values()]

            train_sampler = SpeakerAwareSampler(
                spk_labels, num_samples_cls=num_samples_cls, generator=self.generator,
            )
            shuffle = False
        else:
            train_sampler = None

        self.train_loader = SaveableDataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=drop_last,
            sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            collate_fn=default_collate,
            generator=self.generator,
        )

        self.num_batch_train = len(self.train_loader)
        self.batch_size = batch_size
        self.train_sampler = train_sampler
        self.seed = seed
        if utils.use_ddp():
            self.seed += int(os.environ['RANK'])

        if self.num_batch_train <= 0:
            raise ValueError(
                "Expected num_batch of trainset > 0. There are your egs info: num_gpu={}, num_samples/gpu={}, "
                "batch-size={}, drop_last={}.\nNote: If batch-size > num_samples/gpu and drop_last is true, then it "
                "will get 0 batch.".format(num_gpu, len(trainset) / num_gpu, batch_size, drop_last)
            )

        if val is not None:
            val_batch_size = min(val_batch_size, len(val))  # To save GPU memory

            if len(val) <= 0:
                raise ValueError("Expected num_samples of val > 0.")

            self.val_loader = DataLoader(
                val,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=pin_memory,
                drop_last=False,
            )

            self.num_batch_val = len(self.val_loader)
        else:
            self.val_loader = None
            self.num_batch_val = 0

    @classmethod
    def get_bunch_from_csv(
        cls,
        trainset_csv: str,
        val_csv: str = None,
        egs_params: dict = {},
        data_loader_params_dict: dict = {},
    ):
        if data_loader_params_dict["speaker_aware_sampling"]:
            trainset = SpeakerAwareWaveDataset(trainset_csv, **egs_params)
        elif data_loader_params_dict["paired_sampling"]:
            trainset = PairedWaveDataset(trainset_csv, **egs_params)
        else:
            trainset = WaveDataset(trainset_csv, **egs_params)

        # For multi-GPU training.
        if not utils.is_primary_process():
            val = None
        else:
            if val_csv != "" and val_csv is not None:
                egs_params["aug"] = False
                egs_params["aug_conf"] = ""
                egs_params.pop("chunk2lang", None)
                egs_params.pop("p", None)
                val = WaveDataset(val_csv, val=True, **egs_params)
            else:
                val = None
        return cls(trainset, val, **data_loader_params_dict)

    @classmethod
    def get_bunch_from_egsdir(self, egs_params: dict = {}, data_loader_params_dict: dict = {}):
        train_csv_name = None
        val_csv_name = None

        egs_dir = egs_params.pop("egs_dir")

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "val_csv_name" in egs_params.keys():
            val_csv_name = egs_params.pop("val_csv_name")

        num_targets, train_csv, val_csv = self.get_info_from_egsdir(
            egs_dir, train_csv_name=train_csv_name, val_csv_name=val_csv_name
        )

        # target_num_multiplier = 1

        # If speed perturbation was used before, set target_num_multiplier to 3
        # in egs_params when performing large-margin fine-tuning.
        target_num_multiplier = egs_params.pop("target_num_multiplier", 1)
        if egs_params["aug"]:

            def get_targets_multiplier(aug_classes):
                # target_num_multiplier = 1
                _target_num_multiplier = target_num_multiplier
                for aug_class in aug_classes:
                    if "aug_classes" in aug_class:
                        _target_num_multiplier = get_targets_multiplier(aug_class["aug_classes"])
                    else:
                        if aug_class["aug_type"] == "Speed":
                            return 3
                return _target_num_multiplier

            with open(egs_params["aug_conf"], "r") as fin:
                speech_aug_conf = yaml.load(fin, Loader=yaml.FullLoader)
                target_num_multiplier = get_targets_multiplier(speech_aug_conf["aug_classes"])
        info = {"num_targets": num_targets * target_num_multiplier}

        egs_params["num_targets"] = num_targets
        bunch = self.get_bunch_from_csv(train_csv, val_csv, egs_params, data_loader_params_dict)
        return bunch, info

    @classmethod
    def get_info_from_egsdir(self, egs_dir, train_csv_name=None, val_csv_name=None):
        if os.path.exists(egs_dir + "/info"):
            num_targets = int(utils.read_file_to_list(egs_dir + "/info/num_targets")[0])

            train_csv_name = train_csv_name if train_csv_name is not None else "train.egs.csv"
            val_csv_name = val_csv_name if val_csv_name is not None else "validation.egs.csv"

            train_csv = egs_dir + "/" + train_csv_name
            val_csv = egs_dir + "/" + val_csv_name

            if not os.path.exists(val_csv):
                val_csv = None

            return num_targets, train_csv, val_csv
        else:
            raise ValueError("Expected dir {0} to exist.".format(egs_dir + "/info"))


if __name__ == "__main__":
    data_loader_params_dict = {"batch_size": 5, "shuffle": False, "num_workers": 0}
    bunch, info = DataBunch.get_bunch_from_egsdir(
        "../../../../voxceleb/exp/egs/mfcc_23_pitch-voxceleb1_train_aug-speaker_balance",
        data_loader_params_dict=data_loader_params_dict,
    )
    for i, (data, label) in enumerate(bunch.train_loader):
        print("-----------")
        if i > 2:
            break
        # print(data.size())
        # print(label.size())
