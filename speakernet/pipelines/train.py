# -*- coding:utf-8 -*-
"""
Copyright 2022 Jianchen Li
"""

import os
import sys
import torch
import logging
import warnings
import argparse
from hyperpyyaml import load_hyperpyyaml

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

import speakernet.utils.utils as utils
from speakernet.utils.logging_utils import init_logger
from speakernet.training.optimizer import get_optimizer
from speakernet.training.checkpointer import Checkpointer
from speakernet.utils.kaldi_common import StrToBoolAction
from speakernet.training.lr_scheduler import LRSchedulerWrapper

warnings.filterwarnings("ignore")
logger = init_logger()

# Parser: add this parser to run launcher with some frequent options (really for conveninece).
parser = argparse.ArgumentParser(
    description="""Train xvector framework with pytorch.""",
    formatter_class=argparse.RawTextHelpFormatter,
    conflict_handler="resolve",
)

parser.add_argument(
    "--hparams-file",
    type=str,
    help="A yaml-formatted file using the extended YAML syntax. ",
)
parser.add_argument(
    "--compile",
    type=str,
    action=StrToBoolAction,
    default=False,
    choices=["true", "false"],
    help="Use torch.compile or not.",
)
parser.add_argument(
    "--compile-mode",
    type=str,
    default="reduce-overhead",
    choices=["default", "reduce-overhead", "max-autotune"],
    help="mode of the torch.compile, which is introduced in pytorch2",
)
parser.add_argument(
    "--use-gpu",
    type=str,
    action=StrToBoolAction,
    default=True,
    choices=["true", "false"],
    help="Use GPU or not.",
)
parser.add_argument(
    "--gpu-id",
    type=str,
    default="",
    help="If NULL, then it will be auto-specified.\n"
    "set --gpu-id=1,2,3 to use multi-gpu to extract xvector.\n"
    "Doesn't support multi-gpu training",
)
parser.add_argument(
    "--multi-gpu-solution",
    type=str,
    default="ddp",
    choices=["ddp", "dp"],
    help="if number of gpu_id > 1, this option will be valid to init a multi-gpu solution.",
)
parser.add_argument(
    "--debug",
    type=str,
    action=StrToBoolAction,
    default=False,
    choices=["true", "false"],
    help="",
)
parser.add_argument(
    "--resume-training",
    type=str,
    action=StrToBoolAction,
    default=False,
    choices=["true", "false"],
    help="",
)
parser.add_argument("--train-time-string", type=str, default=" ")
parser.add_argument(
    "--model-dir", type=str, default="", help="extract xvector dir name"
)

# Accept extra args to override yaml
args, overrides = parser.parse_known_args()
overrides_yaml = utils.convert_to_yaml(overrides)

# >>> Init environment
# It is used for multi-gpu training.
gpu_id = utils.parse_gpu_id_option(args.gpu_id)
utils.init_multi_gpu_training(gpu_id, args.multi_gpu_solution)
rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0

time_string = args.train_time_string
if utils.is_primary_process():
    logger.info(f"Timestamp: {time_string}")
model_dir = "exp/{}/{}".format(args.model_dir, time_string)
description = ""

# Load hyperparameters file with command-line overrides
if args.resume_training:
    # Recover training
    with open(f"{model_dir}/config/hyperparams.yaml") as fin:
        hparams = load_hyperpyyaml(fin, overrides_yaml)
else:
    with open(args.hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides_yaml)

    if utils.is_primary_process() and not args.debug:
        logger.info(
            "Please provide a brief description of the experiment and press ENTER to end:"
        )
        description = utils.input_with_timeout(timeout=60)

    # Create experiment directory
    utils.create_model_dir(model_dir, args.debug)

    # Save all code files.
    utils.backup_code(args, hparams, model_dir, overrides, description)

# >>> Set seed
seed = hparams["seed"]
if utils.use_ddp():
    seed += int(os.environ['RANK'])
utils.set_seed(seed, deterministic=True if "deterministic" not in hparams else hparams["deterministic"])

# ---------------------------------------- START ------------------------------------------ #
if utils.is_primary_process():
    logger.info("Loading the dataset to a bunch.")
# The dict [info] contains feat_dim and num_targets.
bunch = utils.import_module(hparams["bunch"])
bunch, info = bunch.DataBunch.get_bunch_from_egsdir(hparams["dataset_params"], hparams["loader_params"])

if utils.is_primary_process():
    logger.info("Loading the encoder.")
feat_params = hparams["feat_params"]
if hparams["features"] == "kaldi_fbank":
    feat_params["device"] = f"cuda:{gpu_id[rank]}"
encoder = utils.import_module(hparams["encoder"])
encoder = encoder.Encoder(
    info["num_targets"],
    **hparams["encoder_params"],
    features=hparams["features"],
    feat_params=feat_params,
)

if torch.__version__ >= "2.0.0" and args.compile:
    encoder = torch.compile(encoder, mode=args.compile_mode)

encoder = utils.convert_synchronized_batchnorm(encoder)

# Select device to GPU
# Order of distributedDataParallel(dataparallel) and optimizer has no effect
# https://medium.com/analytics-vidhya/distributed-training-in-pytorch-part-1-distributed-data-parallel-ae5c645e74cb
# https://discuss.pytorch.org/t/order-of-dataparallel-and-optimizer/114063
encoder = utils.select_model_device(
    encoder, args.use_gpu, gpu_id=gpu_id, benchmark=hparams["benchmark"]
)

if utils.is_primary_process():
    logger.info("Define the optimizer.")
# If ddp used, it will auto-scale learning rate by multiplying number of processes.
hparams["optimizer_params"]["learn_rate"] = utils.auto_scale_lr(hparams["optimizer_params"]["learn_rate"])
optimizer = get_optimizer(
    filter(lambda p: p.requires_grad, encoder.parameters()), hparams["optimizer_params"]
)

if utils.is_primary_process():
    logger.info("Define the lr_scheduler.")
if hparams["lr_scheduler_params"]["name"] == "ExponentialDecay":
    hparams["lr_scheduler_params"][
        "ExponentialDecay.num_iters_per_epoch"
    ] = bunch.num_batch_train
lr_scheduler = LRSchedulerWrapper(optimizer, hparams["lr_scheduler_params"])

if utils.is_primary_process():
    logger.info("Define the checkpointer.")
recoverables = {
    "encoder": encoder,
    "optimizer": optimizer,
    "lr_scheduler": lr_scheduler,
    "dataloader": bunch.train_loader,
}
suffix = ""
if utils.use_ddp():
    suffix = f"/rank{rank}"
checkpointer = Checkpointer(
    checkpoints_dir=f"{model_dir}/checkpoints{suffix}",
    recoverables=recoverables,
    debug=args.debug,
)

if utils.is_primary_process():
    logger.info("Initing the trainer.")
# Package(Elements:dict, Params:dict}. It is a key parameter's package to trainer and model_dir/config/.
package = (
    {
        "data": bunch,
        "encoder": encoder,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "checkpointer": checkpointer,
    },
    {
        "model_dir": model_dir,
        "encoder_blueprint": f"{model_dir}/backup/{hparams['encoder']}",
        "gpu_id": gpu_id[rank],
        "debug": args.debug,
        "record_file": "train.csv",
        "time_string": time_string,
        **hparams["trainer_params"]
    },
)

trainer = utils.import_module(hparams["trainer"])
trainer = trainer.Trainer(package)
trainer.fit()

if utils.use_ddp():
    torch.distributed.destroy_process_group()
