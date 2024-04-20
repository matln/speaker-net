# -*- coding:utf-8 -*-
"""
Copyright 2019 Snowdar
          2021 Jianchen Li
"""

import sys
import math
import torch
import warnings
import numpy as np
from torch.optim.lr_scheduler import (
    _LRScheduler,
    ReduceLROnPlateau,
)

# sys.path.insert(0, "./")

import speakernet.utils.utils as utils
from speakernet.training.optimizer import Lookahead
from speakernet.training.checkpointer import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)


# Wrapper ✿
@register_checkpoint_hooks
class LRSchedulerWrapper:
    def __init__(self, optimizer, params: dict = {}):
        # Suggested weight_decay: 1e-4 for l2 regularization (sgd, adam) and
        #                         1e-1 for decouped weight decay (sgdw, adamw, radam, ralamb, adamod etc.)
        default_params = {
            "name": "MultiStepLR",
            "StepLR.step_size": 1,
            "StepLR.gamma": 0.9,
            "MultiStepLR.milestones": [10, 20],
            "MultiStepLR.gamma": 0.1,
            "IterMultiStepLR.milestones": [10000, 20000],
            "IterMultiStepLR.gamma": 0.1,
            "cyclic.max_lr": 1e-3,
            "cyclic.base_lr": 1e-8,
            "cyclic.step_size_up": 2e4,
            "cyclic.step_size_down": None,
            "cyclic.mode": "triangular2",
            "cyclic.gamma": 1.0,
            "cyclic.scale_fn": None,
            "cyclic.scale_mode": "cycle",
            "cyclic.cycle_momentum": False,
            "cyclic.base_momentum": 0.8,
            "cyclic.max_momentum": 0.9,
            "1cycle.learn_rate": 0.001,
            "1cycle.total_steps": None,
            "1cycle.epochs": None,
            "1cycle.steps_per_epoch": None,
            "1cycle.pct_start": 0.3,
            "1cycle.anneal_strategy": "linear",
            "1cycle.cycle_momentum": False,
            "1cycle.base_momentum": 0.85,
            "1cycle.max_momentum": 0.95,
            "1cycle.div_factor": 25.0,
            "1cycle.final_div_factor": 10000.0,
            "reduceP.metric": "val_acc",
            "reduceP.check_interval": 0,
            "reduceP.factor": 0.5,
            "reduceP.patience": 10,
            "reduceP.threshold": 0.0001,
            "reduceP.cooldown": 0,
            "reduceP.min_lr": 0.0,
            "ExponentialDecay.final_lr": 0.00005,
            "ExponentialDecay.num_iters_per_epoch": None,
            "ExponentialDecay.num_epochs": None,
            "warmup_epoch": "0.0",
        }

        used_params = utils.assign_params_dict(
            default_params, params, force_check=False, support_unknow=True
        )
        split_params = utils.split_params(used_params)

        if isinstance(optimizer, Lookahead):
            base_optimizer = optimizer.optimizer
        else:
            base_optimizer = optimizer

        self.name = split_params["public"]["name"]
        warmup_epoch = split_params["public"]["warmup_epoch"].split('.')
        if len(warmup_epoch) == 2:
            self.warmup_epoch = int(warmup_epoch[0])
            self.warmup_iter = int(warmup_epoch[1])
        elif len(warmup_epoch) == 1:
            self.warmup_epoch = int(warmup_epoch[0])
            self.warmup_iter = 0
        else:
            raise ValueError

        if self.name == "cyclic":
            base_lr = split_params["cyclic"].pop("base_lr")
            max_lr = split_params["cyclic"].pop("max_lr")
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                base_optimizer, base_lr, max_lr, **split_params["cyclic"]
            )
        elif self.name == "1cycle":
            max_lr = split_params["1cycle"].pop("learn_rate")
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                base_optimizer, max_lr, **split_params["1cycle"]
            )
        elif self.name == "stepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                base_optimizer, **split_params["stepLR"]
            )
        elif self.name == "MultiStepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                base_optimizer, **split_params["MultiStepLR"]
            )
        elif self.name == "IterMultiStepLR":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                base_optimizer, **split_params["IterMultiStepLR"]
            )
        elif self.name == "reduceP":
            self.check_interval = split_params["reduceP"].pop("check_interval")
            self.metric = split_params["reduceP"].pop("metric")
            self.min_lr = split_params["reduceP"]["min_lr"]
            if self.metric == "val_acc":
                mode = "max"
            elif self.metric == "val_loss":
                mode = "min"
            else:
                raise ValueError(
                    "Do not support {} metric for ReduceLROnPlateau strategy.".format(
                        self.metric
                    )
                )
            self.lr_scheduler = ReduceLROnPlateau(
                base_optimizer, mode=mode, **split_params["reduceP"]
            )
            self.init = False
        elif self.name == "ExponentialDecay":
            final_lr = split_params["ExponentialDecay"].pop("final_lr")
            num_iters_per_epoch = split_params["ExponentialDecay"].pop(
                "num_iters_per_epoch"
            )
            num_epochs = split_params["ExponentialDecay"].pop("num_epochs")
            assert num_iters_per_epoch is not None
            assert num_epochs is not None
            self.lr_scheduler = ExponentialDecay(
                base_optimizer,
                num_epochs,
                num_iters_per_epoch,
                final_lr,
                **split_params["ExponentialDecay"]
            )
        else:
            raise ValueError("Do not support {0} lr_scheduler now.".format(self.name))

    def is_reduce_point(self, training_point):
        if self.name == "reduceP":
            # It will check the point with a global num_iter value.
            updated_iter = training_point[0] * training_point[2] + training_point[1] + 1
            warmup_iter = int(self.warmup_epoch * training_point[2] + self.warmup_iter)
            return (
                self.check_interval > 0
                and (updated_iter - warmup_iter) % self.check_interval == 0
            ) or (
                self.check_interval <= 0
                and (
                    updated_iter > warmup_iter
                    and training_point[1] + 1 == training_point[2]
                )
            )
        else:
            return False

    def step(self, training_point=None, val_metric=None):
        updated_iter = training_point[0] * training_point[2] + training_point[1] + 1
        warmup_iter = int(self.warmup_epoch * training_point[2] + self.warmup_iter)

        if self.name == "cyclic":
            self.lr_scheduler.step()
        elif self.name == "1cycle":
            self.lr_scheduler.step(updated_iter)
        elif self.name == "stepLR":
            self.lr_scheduler.step(training_point[0])
        elif self.name == "MultiStepLR":
            self.lr_scheduler.step(training_point[0])
        elif self.name == "IterMultiStepLR":
            self.lr_scheduler.step(updated_iter)
        elif self.name == "reduceP":
            if (
                (self.warmup_epoch > 0 or self.warmup_iter > 0)
                and updated_iter <= warmup_iter
            ):
                for group, base_lr in zip(
                    self.lr_scheduler.optimizer.param_groups, self.lr_scheduler.base_lrs
                ):
                    group["lr"] = base_lr
            else:
                # Sample a point in which the metrics of val are computed and adjust learning rate at this point.
                if self.is_reduce_point(training_point):
                    metric = (
                        val_metric[0]
                        if self.metric == "val_loss"
                        else val_metric[1]
                    )
                    self.lr_scheduler.step(metric)
        elif self.name == "ExponentialDecay":
            self.lr_scheduler.step(updated_iter)

        if (
            (self.warmup_epoch > 0 or self.warmup_iter > 0)
            and updated_iter <= warmup_iter
        ):
            for group in self.lr_scheduler.optimizer.param_groups:
                omega = self.get_warmup_factor(updated_iter, warmup_iter)
                group["lr"] *= omega

    def get_warmup_factor(self, updated_iter, warmup_iter):
        return min(1.0, updated_iter / warmup_iter)

    @mark_as_saver
    def _save(self, path):
        if hasattr(self.lr_scheduler, "state_dict") and hasattr(
            self.lr_scheduler, "load_state_dict"
        ):
            torch.save(self.lr_scheduler.state_dict(), path)
        else:
            raise ValueError(
                "state_dict() or load_state_dict() method is not defined in"
                "lr_scheduler!"
            )

    @mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del end_of_epoch
        del device
        if hasattr(self.lr_scheduler, "state_dict") and hasattr(
            self.lr_scheduler, "load_state_dict"
        ):
            self.lr_scheduler.load_state_dict(torch.load(path))
        else:
            raise ValueError(
                "state_dict() or load_state_dict() method is not defined in"
                "lr_scheduler!"
            )


# Learn rate scheduler ✿
class ReduceLROnPlateau(ReduceLROnPlateau):
    def __init__(self, *args, **kargs):
        super(ReduceLROnPlateau, self).__init__(*args, **kargs)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]


class ExponentialDecay(_LRScheduler):
    """

    """

    def __init__(
        self,
        optimizer,
        num_epochs,
        num_iters_per_epoch,
        final_lr,
        last_epoch=-1,
        verbose=False,
    ):
        self.max_iter = num_epochs * num_iters_per_epoch
        self.final_lr = final_lr
        super(ExponentialDecay, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        gamma_lst = [
            math.exp((1 / self.max_iter) * math.log(self.final_lr / base_lr))
            for base_lr in self.base_lrs
        ]
        return [
            group["lr"] * gamma_lst[idx]
            for idx, group in enumerate(self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        current_iter = self.last_epoch
        current_lrs = []
        for base_lr in self.base_lrs:
            current_lr = base_lr * math.exp(
                (current_iter / self.max_iter) * math.log(self.final_lr / base_lr)
            )
            current_lrs.append(current_lr)
        return current_lrs


def show_lr_curve(scheduler):
    import matplotlib.pyplot as plt

    lr_list = []
    # warmup_iter = 1.26759 * 20191
    warmup_iter = 1 * 20191
    for current_iter in range(0, scheduler.max_iter):
        warmup_factor = min(1.0, current_iter / warmup_iter)

        lr_list.append(scheduler._get_closed_form_lr()[0] * warmup_factor)
        if current_iter % 20191 == 0:
            print(f"{current_iter / 20191}: {scheduler._get_closed_form_lr()[0] * warmup_factor}")
        scheduler.last_epoch += 1
        # scheduler.step()
        # lr_list.append(scheduler._last_lr[0])

    data_index = list(range(1, len(lr_list) + 1))

    plt.plot(data_index, lr_list, "-o", markersize=1)
    plt.legend(loc="best")
    plt.xlabel("Iteration")
    plt.ylabel("LR")

    # plt.show()
    plt.savefig("lr.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import inspect

    model = torch.nn.Linear(in_features=1, out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

    num_epochs = 32
    epoch_iter = 20191
    final_lr = 0.00001
    scheduler = ExponentialDecay(optimizer, num_epochs, epoch_iter, final_lr)
    show_lr_curve(scheduler)
