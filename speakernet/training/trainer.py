# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""

import os
import sys
import math
import time
import torch
import random
import logging
import traceback
import progressbar
import numpy as np
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import speakernet.utils.utils as utils
from speakernet.training.reporter import Reporter
from speakernet.training.lr_scheduler import LRSchedulerWrapper
from speakernet.training.checkpointer import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

# Wrap stderr before logger init.
progressbar.streams.wrap_stderr()

# Logger
logger = logging.getLogger("fit_progressbar")
root_logger = logging.getLogger(__name__)

RECENT_CKPT_FLAG = "recent-ckpt"
STEP_CKPT_FLAG = "end-of-step"  # Save every multiple steps
LR_SCHEGULER_CYCLE_CKPT_FLAG = "end-of-lr-scheduler-cycle"
END_OF_EPOCH_CKPT_FLAG = "end-of-epoch"  # Do not modify this one!


# Trainer ✿
@register_checkpoint_hooks
class BaseTrainer:
    """
    This is the structure of Package:

    Package(
        Elements{
            data:Bunch
            encoder:TopVirtualNnet
            },

        Params{
            model_dir:str
            exist_encoder:str
            epochs:int
            ...
            }
        )


    trainer:
        self.elements
        self.params
        self.training_point(current_epoch, current_iter, data.num_batch_train)
    """

    def __init__(self, package):
        default_elements = {
            "data": None,
            "encoder": None,
            "optimizer": None,
            "lr_scheduler": None,
            "checkpointer": None,
        }
        default_params = {
            "model_dir": "",
            "encoder_blueprint": "",
            "exist_encoder": "",
            "gpu_id": 0,
            "max_change": 10.0,
            "compute_accuracy": True,
            "compute_val_accuracy": True,
            "compute_one_batch_val": False,
            "epochs": 21,
            "ckpt_interval_minutes": 15,
            "use_tensorboard": True,
            "mixed_prec": False,
            "mixed_prec_loss": True,
            "debug": False,
            "saved_step": 10000000,
        }

        elements, params = package
        self.elements = utils.assign_params_dict(default_elements, elements, support_unknow=True)
        self.params = utils.assign_params_dict(default_params, params, support_unknow=True)

        assert self.elements["data"] is not None
        assert self.elements["encoder"] is not None
        assert self.elements["optimizer"] is not None
        assert self.elements["lr_scheduler"] is not None
        assert self.elements["checkpointer"] is not None

        assert self.params["model_dir"] != ""
        assert self.params["encoder_blueprint"] != ""

        for k, v in self.elements.items():
            setattr(self, k, v)

        for k, v in self.params.items():
            setattr(self, k, v)

        self.encoder_forward = self.encoder
        self.training_point = (0, 0, self.data.num_batch_train)

        if self.mixed_prec:
            self.scaler = GradScaler()
            self.checkpointer.add_recoverable("mixed_prec", self.scaler)

        # Prepare iterating variables
        self.resume_iter = 0
        self.resume_epoch = 0

        # Add this class to the checkpointer for intra-epoch checkpoints
        self.checkpointer.add_recoverable("trainer", self)

    def init_training(self):
        if utils.is_primary_process():
            root_logger.info(f"Training will run for {self.epochs} epochs.")

        # TODO
        # List parameter count for the user
        if utils.is_primary_process():
            total_params = sum(
                p.numel()
                for name, p in self.encoder.named_parameters()
                if p.requires_grad and "loss" not in name
            )
            if total_params > 0:
                clsname = self.__class__.__name__
                fmt_num = utils.format_order_of_magnitude(total_params)
                root_logger.info(f"{fmt_num} trainable parameters in {clsname}")

        # Original encoder is built in libs.nnet.framework.TopVirtualNnet, and it is not available after
        # wrapped by DistributedDataParallel.
        if isinstance(self.encoder, torch.nn.parallel.DistributedDataParallel):
            self.encoder = self.encoder.module

        # if self.data.val_loader is not None:
        #     val_metric, val_results_dict = self.compute_valation(self.data.val_loader)
        #     print(utils.dict_to_params_str(val_results_dict, auto=False, sep=", "))

        if self.checkpointer.find_checkpoint() is None:
            if utils.is_primary_process() and self.debug is False:
                encoder_creation = self.encoder.get_model_creation()
                utils.write_nnet_config(
                    self.encoder_blueprint,
                    encoder_creation,
                    f"{self.model_dir}/config/nnet.config",
                )
            if self.exist_encoder != "":
                if not os.path.exists(self.exist_encoder):
                    raise FileExistsError
                if utils.is_primary_process():
                    root_logger.info(
                        f"Use {self.exist_encoder} as the initial encoder to start transform-training."
                    )
                self.encoder.load_transform_state_dict(
                    torch.load(self.exist_encoder, map_location="cpu")
                )

        else:
            # ckpt_predicate = lambda c: c.meta["unixtime"] == 1642424800.6493108
            ckpt_predicate = None
            if utils.use_ddp():
                ckpt_predicate = lambda c: c.meta.get(RECENT_CKPT_FLAG, False)

            # Load latest checkpoint to resume training if interrupted
            if utils.is_primary_process():
                # logger 会改变随机种子状态
                # root_logger.info(f"Resume training from {chosen_ckpt.path.name}.")
                root_logger.info(f"Resume training...")

            device = torch.device(f"cuda:{self.gpu_id}")
            chosen_ckpt = self.checkpointer.recover_if_possible(
                ckpt_predicate=ckpt_predicate, device=device
            )

    def save_ckpt(
        self,
        val_loss=None,
        val_acc=None,
        num_to_keep=40,
        recent_num_to_keep=1,
        keep_recent=False,
        recent_ckpt=False,
        end_of_lr_scheduler_cycle=False,
        end_of_epoch=False,
    ):
        """Save the checkpoints.
        meta and min_keys (max_keys), or, meta and ckpt_predicate and importance_keys must be
        specified at the same time.

        Arguments
        ---------
        recent_ckpt : bool
            Periodically save the checkpoint to recover from interruptions.
        keep_recent: bool
            if True, the importance_keys contains "ckpt_recency"
        """
        if end_of_epoch:
            epoch = self.training_point[0] + 1
        else:
            # Saves a intra-epoch CKPT.
            epoch = f"{self.training_point[0]}.{self.training_point[1] + 1}"

        meta = {"epoch": epoch}
        min_keys = []
        max_keys = []
        importance_keys = []
        ckpt_predicate = None
        verbosity = logging.INFO

        # importance_keys and ckpt_predicate are defined in checkpoints.py:find_checkpoints()
        if val_loss is not None and val_acc is not None:
            meta.update({"ValLoss": val_loss, "ValAcc": val_acc})
            min_keys = ["ValLoss"]
            max_keys = ["ValAcc"]
        elif val_loss is not None:
            meta.update({"ValLoss": val_loss})
            min_keys = ["ValLoss"]
            max_keys = []
        elif val_acc is not None:
            meta.update({"ValAcc": val_acc})
            min_keys = []
            max_keys = ["ValAcc"]
        else:
            # importance_keys is [ckpt_recency]
            keep_recent = True
            if recent_ckpt:
                ckpt_predicate = lambda c: RECENT_CKPT_FLAG in c.meta
                epoch = epoch + "_rec"
                meta = {"epoch": epoch, RECENT_CKPT_FLAG: True}
                verbosity = logging.DEBUG
            elif end_of_lr_scheduler_cycle:
                ckpt_predicate = lambda c: LR_SCHEGULER_CYCLE_CKPT_FLAG in c.meta
                meta.update({LR_SCHEGULER_CYCLE_CKPT_FLAG: True})
            elif end_of_epoch:
                # In addition to the default keys "epoch", "unixtime", and "end-of-epoch",
                # meta does not contain other keys.
                ckpt_predicate = lambda c: len(c.meta) == 3
                # "end-of-epoch" is included in meta by default
            else:
                # Save every multiple steps
                ckpt_predicate = lambda c: STEP_CKPT_FLAG in c.meta
                meta = {STEP_CKPT_FLAG: True}

        self.checkpointer.save_and_keep_only(
            meta=meta,
            end_of_epoch=end_of_epoch,
            epoch=epoch,
            num_to_keep=num_to_keep,
            recent_num_to_keep=recent_num_to_keep,
            keep_recent=keep_recent,
            importance_keys=importance_keys,
            min_keys=min_keys,
            max_keys=max_keys,
            ckpt_predicate=ckpt_predicate,
            verbosity=verbosity,
        )

    def compute_forward(self, batch):
        raise NotImplementedError

    def fit_one_batch(self, *batch):
        """A normal training core without fetching data from iterator."""
        self.encoder_forward.train()

        self.optimizer.zero_grad()

        if self.mixed_prec:
            with autocast():
                loss, results_dict = self.compute_forward(*batch)
            self.scaler.scale(loss).backward()
        else:
            loss, results_dict = self.compute_forward(*batch)
            loss.backward()

        if self.max_change > 0:
            if self.mixed_prec:
                # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
                self.scaler.unscale_(self.optimizer)

                # clip_grad_norm需要所有参数的总模长来计算归一化参数，所以若网络是分开定义的，梯度归一化时要传入
                # 所有参数。例如：
                # grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                #                                            list(self.classifier.parameters()), self.max_change)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.max_change
                )
                if not torch.isfinite(grad_norm):
                    if utils.is_primary_process():
                        logger.warning(
                            f"NaN grad in epoch/iter: {self.current_epoch}/{self.current_iter + 1}"
                        )

                # It skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # clip_grad_norm 需要所有参数的总模长来计算归一化参数，所以若网络是分开定义的，梯度归一化时要传入
                # 所有参数。例如：
                # grad_norm = torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) +
                #                                            list(self.classifier.parameters()), self.max_change)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.max_change
                )

                if math.isnan(grad_norm):
                    raise RuntimeError(
                        "There is Nan problem in iter/epoch: {0}/{1} (nan batch and params are saved in {2})".format(
                            self.training_point[1] + 1,
                            self.training_point[0],
                            "{0}/nan.*".format(self.model_dir),
                        )
                    )
                else:
                    self.optimizer.step()
        else:
            if self.mixed_prec:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        return results_dict

    def compute_validation(self, data_loader):
        """A normal evaluation core."""
        train_status = self.encoder.training  # Record status.
        self.encoder.eval()

        loss = 0.0
        accuracy = 0.0 if self.compute_val_accuracy else None

        num_samples = 0
        with torch.no_grad():
            for this_data in data_loader:
                inputs, targets = this_data
                inputs = self.encoder.get_feats(inputs)
                loss += self.encoder(inputs, targets).item() * len(targets)
                num_samples += len(targets)

                if self.compute_val_accuracy:
                    # This will occupy extra GPU memory.
                    accuracy += self.encoder.get_accuracy(targets) * len(targets)

                if self.compute_one_batch_val:
                    break

            avg_loss = loss / num_samples
            avg_accuracy = accuracy / num_samples if self.compute_val_accuracy else None

        if train_status:
            self.encoder.train()

        # For ReduceLROnPlateau and checkpointer. (loss, accuracy)
        val_metric = (avg_loss, avg_accuracy)

        return val_metric, {"val_loss": avg_loss, "val_acc": avg_accuracy * 100}

    def step_lr_scheduler(self, lr_scheduler_params):
        if (
            self.data.val_loader is not None
            and not self.val_computed
            and self.lr_scheduler.name == "reduceP"
            and self.lr_scheduler.is_reduce_point(self.training_point)
        ):
            assert self.data.val_loader is not None
            val_metric, val_results_dict = self.compute_validation(self.data.val_loader)
            lr_scheduler_params["val_metric"] = val_metric
            self.val_computed = True
        elif self.val_computed:
            val_metric = lr_scheduler_params["val_metric"]

        # It is not convenient to wrap lr_scheduler (doing).
        if isinstance(self.lr_scheduler, LRSchedulerWrapper):
            self.lr_scheduler.step(**lr_scheduler_params)
            if utils.is_primary_process() and not self.debug:
                current_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                if self.lr_scheduler.name == "reduceP":
                    if (
                        current_lr < self.last_lr
                        and self.training_point[0] >= self.lr_scheduler.warmup_epoch
                    ):
                        self.last_lr = current_lr
                        self.save_ckpt(val_loss=val_metric[0], val_acc=val_metric[1])
                    elif (
                        current_lr <= self.lr_scheduler.min_lr
                        and self.lr_scheduler.is_reduce_point(self.training_point)
                        and self.training_point[0] >= self.lr_scheduler.warmup_epoch
                    ):
                        self.save_ckpt(val_loss=val_metric[0], val_acc=val_metric[1])
                elif self.lr_scheduler.name == "cyclic":
                    cyclic_size = self.lr_scheduler.lr_scheduler.total_size
                    total_iter = (
                        self.training_point[0] * self.training_point[2] + self.training_point[1] + 1
                    )
                    if total_iter % cyclic_size == 0:
                        if self.data.val_loader is not None:
                            if not self.val_computed:
                                (
                                    val_metric,
                                    val_results_dict,
                                ) = self.compute_validation(self.data.val_loader)
                                self.val_computed = True
                            self.save_ckpt(val_loss=val_metric[0], val_acc=val_metric[1])
                        else:
                            self.save_ckpt(rencent_num_to_keep=10, end_of_lr_scheduler_cycle=True)
        else:
            # For some pytorch lr_schedulers, but it is not available for all.
            self.lr_scheduler.step(self.current_epoch)

    def fit(self):
        """Main function to start a training process."""
        try:
            self.init_training()

            if utils.is_primary_process():
                self.reporter = Reporter(self)

            # For lookahead.
            if getattr(self.optimizer, "optimizer", None) is not None:
                self.optimizer = self.optimizer.optimizer
            self.last_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

            for self.current_epoch in range(self.resume_epoch, self.epochs):
                if isinstance(
                    self.data.train_loader.sampler,
                    torch.utils.data.distributed.DistributedSampler,
                ) or hasattr(self.data.train_loader.sampler, "set_epoch"):
                    self.data.train_loader.sampler.set_epoch(self.current_epoch)

                self.data.generator.manual_seed(self.data.seed + self.current_epoch)

                assert self.data.train_loader.persistent_workers is False

                worker_state = {}

                last_ckpt_time = time.time()

                for self.current_iter, batch in enumerate(self.data.train_loader, self.resume_iter):
                    if self.data.train_loader.num_workers > 0:
                        # The loader return the batch in the order of worker_id (cyclic)
                        _worker_state = batch[-1]

                        # A batch of data is from the same worker
                        worker_state[_worker_state["worker_id"]] = _worker_state
                        self.data.train_loader.dataset.worker_state = worker_state
                    batch = batch[:-1]

                    self.training_point = (
                        self.current_epoch,
                        self.current_iter,
                        self.data.num_batch_train,
                    )  # It is important for reporter.

                    if self.encoder.use_step:
                        self.encoder.step(*self.training_point)

                    results_dict = self.fit_one_batch(batch)

                    if self.encoder.margin_loss:
                        results_dict[
                            "margin_percent"
                        ] = f"{1 / (1 + self.encoder.loss.lambda_factor):.2f}"

                    self.encoder.backward_step(*self.training_point)

                    # For multi-GPU training. Remember that it is not convenient to wrap lr_scheduler
                    # for there are many strategies with different details. Here, only ReduceLROnPlateau
                    # and some simple schedulers whose step() parameter is 'epoch' only are supported.
                    lr_scheduler_params = {"training_point": self.training_point}

                    self.val_computed = False
                    if utils.is_primary_process():
                        if self.data.val_loader and self.reporter.is_report(self.training_point):
                            if not self.val_computed:
                                (
                                    val_metric,
                                    val_results_dict,
                                ) = self.compute_validation(self.data.val_loader)
                                lr_scheduler_params["val_metric"] = val_metric
                                self.val_computed = True

                            # real_snapshot is set for tensorboard to avoid workspace problem
                            real_snapshot = dict(**results_dict, **val_results_dict)
                            snapshot = {
                                k: (
                                    "{0:.6f}".format(v)
                                    if "loss" in k
                                    else "{0:.2f}".format(v)
                                    if "acc" in k
                                    else v
                                )
                                for (k, v) in real_snapshot.items()
                            }
                            snapshot["real"] = {
                                k: v for (k, v) in real_snapshot.items() if type(v) == float
                            }

                        else:
                            real_snapshot = results_dict
                            snapshot = {
                                k: (
                                    "{0:.6f}".format(v)
                                    if "loss" in k
                                    else "{0:.2f}".format(v)
                                    if "acc" in k
                                    else v
                                )
                                for (k, v) in real_snapshot.items()
                            }
                            snapshot["real"] = {
                                k: v for (k, v) in real_snapshot.items() if type(v) == float
                            }

                    if self.lr_scheduler is not None:
                        self.step_lr_scheduler(lr_scheduler_params)

                    if utils.is_primary_process():
                        self.reporter.update(snapshot)

                        ## Save the end-of-step checkpoint.
                        if (self.current_iter + 1) % self.saved_step == 0 and not self.debug:
                            # self.save_ckpt 后面不要有随机化的操作
                            self.save_ckpt(recent_num_to_keep=100000)

                    ## Save the recent checkpoints on all devices
                    if utils.use_ddp():
                        shared_timestamp = torch.Tensor([time.time() - last_ckpt_time]).cuda()
                        dist.broadcast(shared_timestamp, src=0)
                        timestamp = shared_timestamp[0].item()
                    else:
                        timestamp = time.time() - last_ckpt_time

                    if (
                        not self.debug
                        and self.ckpt_interval_minutes > 0
                        and timestamp >= self.ckpt_interval_minutes * 60.0
                    ):
                        self.save_ckpt(recent_num_to_keep=1, recent_ckpt=True)
                        last_ckpt_time = time.time()

                # End of epoch
                self.resume_iter = 0

                # hasattr(self, "val_computed"): "val_computed" is not present when resuming
                # training. Skip saving checkpoint at the end of the epoch when resuming training.
                if utils.is_primary_process() and not self.debug and hasattr(self, "val_computed"):
                    if self.data.val_loader is not None:
                        assert self.val_computed is True
                        self.save_ckpt(
                            val_loss=val_metric[0],
                            val_acc=val_metric[1],
                            end_of_epoch=True,
                        )
                    else:
                        self.save_ckpt(recent_num_to_keep=1, end_of_epoch=True)

                # Initialize worker_state. Otherwise recover random state in worker_init_fn() will be re-executed.
                self.data.train_loader.dataset.worker_state = {}

            if utils.is_primary_process():
                self.reporter.finish()
        except BaseException as e:
            if utils.use_ddp():
                utils.cleanup_ddp()
            if not isinstance(e, KeyboardInterrupt):
                traceback.print_exc()
            sys.exit(1)

    @mark_as_saver
    def _save(self, path):
        save_dict = {
            "iter": self.current_iter + 1,
            "epoch": self.current_epoch,
            "np_state": np.random.get_state(),
            "random_state": random.getstate(),
            "torch_state": torch.get_rng_state(),
        }

        torch.save(save_dict, path)

    @mark_as_loader
    def _recover(self, path, end_of_epoch, device):
        del device
        save_dict = torch.load(path)
        if end_of_epoch:
            self.resume_iter = 0
            self.resume_epoch = save_dict["epoch"] + 1
        else:
            self.resume_iter = save_dict["iter"]
            self.resume_epoch = save_dict["epoch"]

        np.random.set_state(save_dict["np_state"])
        random.setstate(save_dict["random_state"])
        torch.set_rng_state(save_dict["torch_state"])


class Trainer(BaseTrainer):
    def compute_forward(self, batch):
        inputs, targets = batch
        if len(inputs.size()) > 2 and inputs.size(1) == 2:
            # For PairedWaveDataset
            inputs = inputs.reshape(inputs.size(0) * inputs.size(1), inputs.size(2))
            targets = targets.reshape(targets.size(0) * targets.size(1))
        inputs = self.encoder.get_feats(inputs)
        loss = self.encoder_forward(inputs, targets)

        accuracy = self.encoder.get_accuracy(targets) if self.compute_accuracy else None

        return loss, {"train_loss": loss.detach().item(), "train_acc": accuracy * 100}
