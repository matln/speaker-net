"""
Copyright 2022 Jianchen Li
"""

import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

import speakernet.utils.utils as utils
from speakernet.preprocessing.features import Fbank
from speakernet.models.modules import (
    Conv1dBnReluLayer,
    SpeakerNet,
    StatisticsPooling,
    SoftmaxLoss,
    for_extract_embedding,
)


class Encoder(SpeakerNet):
    def init(
        self,
        inputs_dim: int,
        num_targets: int,
        training: bool = True,
        extracted_embedding: str = "near",
        features: str = "fbank",
        feat_params: dict = {},
        tdnn_layer_params: dict = {},
        label_smoothing: float = 0.0,
    ):

        default_feat_params = {
            "sample_rate": 16000,
            "n_fft": 512,
            "window_size": 0.025,
            "window_stride": 0.01,
            "n_mels": 80,
            "preemph": None,
            "log": True,
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

        feat_params = utils.assign_params_dict(default_feat_params, feat_params)
        tdnn_layer_params = utils.assign_params_dict(default_tdnn_layer_params, tdnn_layer_params)

        # Var
        self.extracted_embedding = extracted_embedding
        self.features = features
        self.inputs_dim = feat_params["n_mels"]

        # Acoustic features
        if features == "fbank":
            self.extract_fbank = Fbank(**feat_params, training=training)

        # Encoder
        self.tdnn1 = Conv1dBnReluLayer(inputs_dim, 512, [-2, -1, 0, 1, 2], **tdnn_layer_params)
        self.tdnn2 = Conv1dBnReluLayer(512, 512, [-2, 0, 2], **tdnn_layer_params)
        self.tdnn3 = Conv1dBnReluLayer(512, 512, [-3, 0, 3], **tdnn_layer_params)
        self.tdnn4 = Conv1dBnReluLayer(512, 512, **tdnn_layer_params)
        self.tdnn5 = Conv1dBnReluLayer(512, 1500, **tdnn_layer_params)

        # Pooling layer
        self.stats = StatisticsPooling(1500, stddev=True)

        # Embedding layer
        self.tdnn6 = Conv1dBnReluLayer(self.stats.get_output_dim(), 512, **tdnn_layer_params)
        self.tdnn7 = Conv1dBnReluLayer(512, 512, **tdnn_layer_params)

        # Loss
        # No need when extracting embedding.
        if training:
            self.loss = SoftmaxLoss(512, num_targets, label_smoothing=label_smoothing)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = [
                "tdnn1",
                "tdnn2",
                "tdnn3",
                "tdnn4",
                "tdnn5",
                "stats",
                "tdnn6",
                "tdnn7",
            ]

    @utils.for_device_free
    def get_feats(self, wavs: torch.Tensor) -> torch.Tensor:
        return self.extract_fbank(wavs)

    @utils.for_device_free
    def forward(self, inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor (a batch), including [samples-index,  frames-dim-index, frames-index]
        """
        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)
        x = self.tdnn6(x)
        x = self.tdnn7(x)

        loss = self.loss(x.float(), targets)

        return loss

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs
        x = self.tdnn1(x)
        x = self.tdnn2(x)
        x = self.tdnn3(x)
        x = self.tdnn4(x)
        x = self.tdnn5(x)
        x = self.stats(x)

        if self.extracted_embedding == "far":
            xvector = self.tdnn6.affine(x)
        elif self.extracted_embedding == "near":
            x = self.tdnn6(x)
            xvector = self.tdnn7.affine(x)

        return xvector


if __name__ == "__main__":
    pass
