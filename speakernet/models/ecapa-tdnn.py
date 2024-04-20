""" Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))
# sys.path.insert(0, os.path.dirname("/data/lijianchen/workspace/sre/speaker-net/speakernet"))

import speakernet.utils.utils as utils
from speakernet.preprocessing.features import Fbank, KaldiFbank
from speakernet.models.modules import (
    Conv1dBnReluLayer,
    SpeakerNet,
    MarginSoftmaxLoss,
    SoftmaxLoss,
    for_extract_embedding,
    SE_Res2Block,
    ChannelContextStatisticsPooling
)


class Encoder(SpeakerNet):
    def init(
        self,
        num_targets: int,
        channels: int = 512,
        emb_dim: int = 192,
        dropout: float = 0.0,
        training: bool = True,
        extracted_embedding: str = "near",
        features: str = "fbank",
        feat_params: dict = {},
        tdnn_layer_params: dict = {},
        layer5_params: dict = {},
        fc1: bool = False,
        fc1_params: dict = {},
        fc2_params: dict = {},
        margin_loss: bool = True,
        margin_loss_params: dict = {},
        label_smoothing: float = 0.0,
        pooling: str = "channel_context",
        pooling_params: dict = {},
        use_step: bool = True,
        step_params: dict = {},
    ):
        default_feat_params = {
            "num_bins": 80,
            "dither": 0.0,
            "cmvn": True,
            "norm_var": False
        }

        default_tdnn_layer_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        default_layer5_params = {"nonlinearity": "relu", "bn": False}

        default_fc2_params = {"nonlinearity": "", "bn": True}

        default_pooling_params = {
            "num_head": 1,
            "hidden_size": 128,
            "share": False,
            "affine_layers": 2,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True,
            "nonlinearity": "tanh",
            "global_context_att": True,
        }

        default_margin_loss_params = {
            "method": "am",
            "m": 0.2,
            "feature_normalize": True,
            "s": 30,
        }

        default_step_params = {
            "T": None,
            "m": False,
            "lambda_0": 0,
            "lambda_b": 1000,
            "alpha": 5,
            "gamma": 1e-4,
            "s": False,
            "s_tuple": (30, 12),
            "s_list": None,
            "t": False,
            "t_tuple": (0.5, 1.2),
            "p": False,
            "p_tuple": (0.5, 0.1),
        }

        feat_params = utils.assign_params_dict(default_feat_params, feat_params)
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        layer5_params = utils.assign_params_dict(default_layer5_params, layer5_params)
        layer5_params = utils.assign_params_dict(default_tdnn_layer_params, layer5_params)
        fc1_params = utils.assign_params_dict(default_tdnn_layer_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc2_params, fc2_params)
        fc2_params = utils.assign_params_dict(default_tdnn_layer_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params
        )
        step_params = utils.assign_params_dict(default_step_params, step_params)

        self.use_step = use_step
        self.step_params = step_params
        self.extracted_embedding = extracted_embedding  # For extract.
        self.features = features
        self.margin_loss = margin_loss
        self.inputs_dim = feat_params["num_bins"]
        if feat_params.get("use_energy", False):
            self.inputs_dim += 1

        # Acoustic features
        if features == "fbank":
            self.extract_fbank = Fbank(**feat_params, training=training)
        elif features == "kaldi_fbank":
            self.extract_fbank = KaldiFbank(**feat_params)

        # Encoder
        self.layer1 = Conv1dBnReluLayer(
            self.inputs_dim, channels, [-2, -1, 0, 1, 2], **tdnn_layer_params
        )
        # channels, kernel_size, stride, padding, dilation, scale
        self.layer2 = SE_Res2Block(channels, [-2, 0, 2], 8, tdnn_layer_params)
        self.layer3 = SE_Res2Block(channels, [-3, 0, 3], 8, tdnn_layer_params)
        self.layer4 = SE_Res2Block(channels, [-4, 0, 4], 8, tdnn_layer_params)

        cat_channels = channels * 3
        self.layer5 = Conv1dBnReluLayer(cat_channels, cat_channels, [0], **layer5_params)

        # Pooling layer
        if pooling == "channel_context":
            self.pooling = ChannelContextStatisticsPooling(cat_channels, **pooling_params)
            self.bn_pool = nn.BatchNorm1d(cat_channels * 2)
            self.fc1 = (
                Conv1dBnReluLayer(cat_channels * 2, emb_dim, **fc1_params) if fc1 else None
            )
        else:
            raise ValueError
            # self.pooling = StatisticsPooling(cat_channels, stddev=True)

        # Embedding layer
        if fc1:
            fc2_in_dim = emb_dim
        else:
            fc2_in_dim = cat_channels * 2
        self.fc2 = Conv1dBnReluLayer(fc2_in_dim, emb_dim, **fc2_params)
        self.dropout = torch.nn.Dropout2d(p=dropout) if dropout > 0 else None

        # Loss
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(emb_dim, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets, label_smoothing=label_smoothing)

    @utils.for_device_free
    def get_feats(self, wavs: torch.Tensor) -> torch.Tensor:
        return self.extract_fbank(wavs)

    @utils.for_device_free
    def forward(self, inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        inputs: [batch, features-dim, frames-lens]
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        out = self.bn_pool(self.pooling(out))
        out = self.auto(self.fc1, out)
        out = self.fc2(out)
        out = self.auto(self.dropout, out)

        out = self.loss(out.float(), targets)

        return out

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """
        out1 = self.layer1(inputs)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        # 在 channel 维连接
        out = torch.cat([out2, out3, out4], dim=1)
        out = self.layer5(out)
        out = self.bn_pool(self.pooling(out))

        if self.extracted_embedding == "far":
            assert self.fc1 is not None
            xvector = self.fc1.affine(out)
        elif self.extracted_embedding == "near_affine":
            out = self.auto(self.fc1, out)
            xvector = self.fc2.affine(out)
        elif self.extracted_embedding == "near":
            out = self.auto(self.fc1, out)
            xvector = self.fc2(out)

        return xvector


if __name__ == "__main__":
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(128, 80, 200)
    model = Encoder(inputs_dim=80, num_targets=5994, channels=512, emb_dim=192)
    # out = model(x)
    print(model)
    # print(out.shape)    # should be [2, 192]

    import numpy as np

    print(np.sum([p.numel() for p in model.parameters()]).item())
