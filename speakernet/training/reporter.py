# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""

import os
import sys
import time
import shutil
import logging
import traceback
import progressbar
import pandas as pd
# from rich import print

from typing import Optional
from multiprocessing import Process, Queue
from progressbar.widgets import (
    FormatWidgetMixin,
    VariableMixin,
    WidgetBase
)

import speakernet.utils.utils as utils

# Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PBVariable(FormatWidgetMixin, VariableMixin, WidgetBase):
    """
    Displays a custom variable.
    The original progressbar.Variable does not print the integer 0
    """

    def __init__(self, name, format="{name}: {formatted_value}", width=6, precision=3, **kwargs):
        """Creates a Variable associated with the given name."""
        self.format = format
        self.width = width
        self.precision = precision
        VariableMixin.__init__(self, name=name)
        WidgetBase.__init__(self, **kwargs)

    def __call__(self, progress, data):
        value = data["variables"][self.name]
        context = data.copy()
        context["value"] = value
        context["name"] = self.name
        context["width"] = self.width
        context["precision"] = self.precision

        try:
            # Make sure to try and cast the value first, otherwise the
            # formatting will generate warnings/errors on newer Python releases
            value = float(value)
            fmt = "{value:{width}.{precision}}"
            context["formatted_value"] = fmt.format(**context)
        except (TypeError, ValueError):
            if value:
                context["formatted_value"] = "{value:{width}}".format(**context)
            else:
                context["formatted_value"] = "0"

        return self.format.format(**context)


class Reporter():
    def __init__(self, trainer):
        default_params = {
            "report_times_every_epoch": None,
            "report_interval_iters": 100,
            "record_file": "train.csv",
            "use_tensorboard": False
        }
        self.trainer = trainer
        default_params = utils.assign_params_dict(default_params, self.trainer.params)

        if default_params["report_times_every_epoch"] is not None:
            self.report_interval_iters = max(1, self.trainer.training_point[2] // default_params["report_times_every_epoch"])
        else:
            self.report_interval_iters = default_params["report_interval_iters"]

        if not self.trainer.params["debug"] and default_params["use_tensorboard"]:
            # from tensorboardX import SummaryWriter
            from torch.utils.tensorboard import SummaryWriter
            model_name = os.path.basename(self.trainer.params["model_dir"])
            # time_string = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            # time_string = self.trainer.params["time_string"]
            # self.board_writer = SummaryWriter("{}/log/{}-{}-tensorboard".format(self.trainer.params["model_dir"], model_name, time_string))
            # self.board_writer = SummaryWriter("{}/log/{}-{}-tensorboard".format(
            #     self.trainer.params["model_dir"], time_string, model_name))
            self.board_writer = SummaryWriter("{}/log/tensorboard".format(self.trainer.params["model_dir"]))
        else:
            self.board_writer = None

        self.epochs = self.trainer.epochs

        self.optimizer = self.trainer.optimizer

        # For optimizer wrapper such as lookahead.
        # "None" is the default value
        if getattr(self.optimizer, "optimizer", None) is not None:
            self.optimizer = self.optimizer.optimizer

        self.device = "{0}".format(utils.get_device(self.trainer.elements["encoder"]))

        self.record_value = []

        self.start_write_log = False
        if (
            not self.trainer.params["debug"]
            and default_params["record_file"] != ""
            and default_params["record_file"] is not None
        ):
            self.record_file = "{0}/log/{1}".format(self.trainer.params["model_dir"], default_params["record_file"])

            # The case to resume training
            if self.trainer.checkpointer.find_checkpoint() is not None:
                # train.csv using append mode
                self.start_write_log = True
        else:
            self.record_file = None

        # A format to show progress
        # Do not use progressbar.Bar(marker="\x1b[32m█\x1b[39m") and progressbar.SimpleProgress(format='%(value_s)s/%(max_value_s)s') to avoid too long string.
        widgets = [progressbar.Percentage(format='%(percentage)3.2f%%'), " | ",
                   PBVariable('current_epoch', format='Epoch: {formatted_value}', width=0, precision=0), "/{0}, ".format(self.epochs),
                   progressbar.Variable('current_iter', format='Iter: {formatted_value}', width=0, precision=0), "/{0}".format(self.trainer.training_point[2]),
                   " (", progressbar.Timer(format='ELA: %(elapsed)s'), ", ", progressbar.AdaptiveETA(), ")"]

        # total num of iter
        max_value = self.epochs * self.trainer.training_point[2]

        self.bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets, redirect_stdout=True)

        # Use multi-process for update.
        self.queue = Queue()
        self.process = Process(target=self._update, daemon=True)
        self.process.start()

    def is_report(self, training_point):
        return (training_point[1] % self.report_interval_iters == 0 or
                training_point[1] + 1 == training_point[2])

    def record(self, info_dict, training_point):
        if self.record_file is not None:
            self.record_value.append(info_dict)

            dataframe = pd.DataFrame(self.record_value)
            if self.start_write_log:
                dataframe.to_csv(self.record_file, mode='a', header=False, index=False)
            else:
                # with open(self.record_file, "w") as f:
                #     f.truncate()
                dataframe.to_csv(self.record_file, header=True, index=False)
                self.start_write_log = True
            self.record_value.clear()
        if self.is_report(training_point):
            print("Device={0}, {1}".format(self.device, utils.dict_to_params_str(info_dict, auto=False, sep=", ")))

    def _update(self):
        # Do not use any var which will be updated by main process, such as self.trainer.training_point.
        while True:
            try:
                res = self.queue.get()
                if res is None:
                    self.bar.finish()
                    break

                snapshot, training_point, current_lr = res
                current_epoch, current_iter, num_batchs_train = training_point
                updated_iters = current_epoch * num_batchs_train + current_iter + 1
                self.bar.update(updated_iters, current_epoch=current_epoch, current_iter=current_iter + 1)

                if snapshot is not None:
                    real_snapshot = snapshot.pop("real")
                    if self.board_writer is not None:
                        self.board_writer.add_scalars("scalar_base", {"epoch": float(current_epoch),
                                                                      "lr": current_lr}, updated_iters)

                        loss_dict = {}
                        acc_dict = {}
                        for key in real_snapshot.keys():
                            if "loss" in key:
                                loss_dict[key] = real_snapshot[key]
                            elif "acc" in key:
                                acc_dict[key] = real_snapshot[key]
                            else:
                                self.board_writer.add_scalar(key, real_snapshot[key], updated_iters)

                        self.board_writer.add_scalars("scalar_acc", acc_dict, updated_iters)
                        self.board_writer.add_scalars("scalar_loss", loss_dict, updated_iters)

                    info_dict = {"epoch": current_epoch, "iter": current_iter + 1,
                                 "position": updated_iters, "lr": "{0:.8f}".format(current_lr)}
                    info_dict.update(snapshot)
                    self.record(info_dict, training_point)
            except BaseException as e:
                self.bar.finish()
                if not isinstance(e, KeyboardInterrupt):
                    traceback.print_exc()
                sys.exit(1)

    def update(self, snapshot: Optional[dict] = None):
        # One update calling and one using of self.trainer.training_point and current_lr.
        # training_point is updated on line 265 in trainer.py
        current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        # snapshot is None when resuming training
        self.queue.put((snapshot, self.trainer.training_point, current_lr))

    def finish(self):
        self.queue.put(None)
        # Wait process completed.
        self.process.join()
