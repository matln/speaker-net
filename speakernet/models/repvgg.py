"""
RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
Github source: https://github.com/DingXiaoH/RepVGG
Licensed under The MIT License [see LICENSE for details]
"""
import os
import sys
import copy
import math
import torch
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.getenv("speakernet")))

import speakernet.utils.utils as utils
from speakernet.preprocessing.features import Fbank, KaldiFbank
from speakernet.models.modules import (
    AttentiveStatisticsPooling,
    LDEPooling,
    MarginSoftmaxLoss,
    MultiHeadAttentionPooling,
    MultiResolutionMultiHeadAttentionPooling,
    Conv1dBnReluLayer,
    RepVGG,
    SoftmaxLoss,
    StatisticsPooling,
    SpeakerNet,
    for_extract_embedding,
)


class Encoder(SpeakerNet):
    """RepVGG models"""

    def init(
        self,
        num_targets,
        training=True,
        deploy=False,
        extracted_embedding="near",
        features: str = "fbank",
        feat_params: dict = {},
        repvgg_params={},
        pooling="statistics",
        pooling_params={},
        fc1=False,
        fc1_params={},
        fc2_params={},
        margin_loss=False,
        margin_loss_params={},
        label_smoothing=0.0,
        use_step=False,
        step_params={},
        transfer_from="softmax_loss",
        emb_dim: Optional[int] = None,
    ):
        default_feat_params = {
            "num_bins": 80,
            "dither": 0.0,
            "cmvn": True,
            "norm_var": False
        }

        optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
        g2_map = {layer: 2 for layer in optional_groupwise_layers}
        g4_map = {layer: 4 for layer in optional_groupwise_layers}

        repvgg_A0_params = {
            "num_blocks": [2, 4, 14, 1],
            "planes": [48, 96, 192, 1280],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_A1_params = {
            "num_blocks": [2, 4, 14, 1],
            "planes": [64, 128, 256, 1280],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_A2_params = {
            "num_blocks": [2, 4, 14, 1],
            "planes": [96, 192, 384, 1408],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B0_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [64, 128, 256, 1280],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B1_custom_params = {
            "num_blocks": [3, 4, 23, 3],
            "planes": [64, 128, 256, 512],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B1_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [128, 256, 512, 2048],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B1g2_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [128, 256, 512, 2048],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g2_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B1g4_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [128, 256, 512, 2048],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g4_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B2_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [160, 320, 640, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B2g2_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [160, 320, 640, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g2_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B2g4_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [160, 320, 640, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g4_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B3_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [192, 384, 768, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B3g2_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [192, 384, 768, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g2_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_B3g4_params = {
            "num_blocks": [4, 6, 16, 1],
            "planes": [192, 384, 768, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": g4_map,
            "deploy": deploy,
            "use_checkpoint": False,
        }

        repvgg_D2se_params = {
            "num_blocks": [8, 14, 24, 1],
            "planes": [160, 320, 640, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_se": True,
            "use_checkpoint": False,
        }

        repvggplus_params = {
            "num_blocks": [8, 14, 24, 1],
            "planes": [160, 320, 640, 2560],
            "strides": [1, 1, 2, 2, 2],
            "norm_layer_params": {"momentum": 0.1, "affine": True},
            "override_groups_map": None,
            "deploy": deploy,
            "use_post_se": True,
            "use_checkpoint": False,
        }

        default_pooling_params = {
            "num_head": 1,
            "hidden_size": 64,
            "share": True,
            "affine_layers": 1,
            "context": [0],
            "stddev": True,
            "temperature": False,
            "fixed": True,
        }

        default_fc_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.5, "affine": True, "track_running_stats": True},
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

        repvgg_params_dict = {
            "RepVGG-A0": repvgg_A0_params,
            "RepVGG-A1": repvgg_A1_params,
            "RepVGG-A2": repvgg_A2_params,
            "RepVGG-B0": repvgg_B0_params,
            "RepVGG-B1": repvgg_B1_params,
            "RepVGG-B1-custom": repvgg_B1_custom_params,
            "RepVGG-B1g2": repvgg_B1g2_params,
            "RepVGG-B1g4": repvgg_B1g4_params,
            "RepVGG-B2": repvgg_B2_params,
            "RepVGG-B2g2": repvgg_B2g2_params,
            "RepVGG-B2g4": repvgg_B2g4_params,
            "RepVGG-B3": repvgg_B3_params,
            "RepVGG-B3g2": repvgg_B3g2_params,
            "RepVGG-B3g4": repvgg_B3g4_params,
            "RepVGG-D2se": repvgg_D2se_params,
            "RepVGGPlus": repvggplus_params
        }

        assert "name" in repvgg_params
        repvgg_name = repvgg_params.pop("name")
        default_repvgg_params = repvgg_params_dict[repvgg_name]
        repvgg_params = utils.assign_params_dict(default_repvgg_params, repvgg_params)
        feat_params = utils.assign_params_dict(default_feat_params, feat_params)
        pooling_params = utils.assign_params_dict(default_pooling_params, pooling_params)
        fc1_params = utils.assign_params_dict(default_fc_params, fc1_params)
        fc2_params = utils.assign_params_dict(default_fc_params, fc2_params)
        margin_loss_params = utils.assign_params_dict(
            default_margin_loss_params, margin_loss_params
        )
        step_params = utils.assign_params_dict(default_step_params, step_params)

        # Var.
        self.extracted_embedding = extracted_embedding  # only near here.
        self.use_step = use_step
        self.step_params = step_params
        self.margin_loss = margin_loss
        self.features = features
        self.inputs_dim = feat_params["num_bins"]
        if feat_params.get("use_energy", False):
            self.inputs_dim += 1

        assert emb_dim is not None

        # Acoustic features
        if features == "fbank":
            self.extract_fbank = Fbank(**feat_params, training=training)
        elif features == "kaldi_fbank":
            self.extract_fbank = KaldiFbank(**feat_params)

        # Encoder
        self.repvgg = RepVGG(**repvgg_params)

        # 将 channel 和 feat_dim 乘起来，feat_dim = ceil(inputs_dim / 16)
        repvgg_out_dim = (
            math.ceil(self.inputs_dim / np.prod(self.repvgg.strides)) * self.repvgg.planes[-1]
        )

        # Pooling layer
        stddev = pooling_params.pop("stddev")
        if pooling == "lde":
            self.stats = LDEPooling(repvgg_out_dim, c_num=pooling_params["num_head"])
        elif pooling == "attentive":
            self.stats = AttentiveStatisticsPooling(
                repvgg_out_dim,
                hidden_size=pooling_params["hidden_size"],
                context=pooling_params["context"],
                stddev=stddev,
            )
        elif pooling == "multi-head":
            self.stats = MultiHeadAttentionPooling(
                repvgg_out_dim, stddev=stddev, **pooling_params
            )
        elif pooling == "multi-resolution":
            self.stats = MultiResolutionMultiHeadAttentionPooling(
                repvgg_out_dim, **pooling_params
            )
        else:
            self.stats = StatisticsPooling(repvgg_out_dim, stddev=stddev)

        # Embedding layer
        self.fc1 = (
            Conv1dBnReluLayer(self.stats.get_output_dim(), emb_dim, **fc1_params)
            if fc1
            else None
        )

        if fc1:
            fc2_in_dim = emb_dim
        else:
            fc2_in_dim = self.stats.get_output_dim()

        self.fc2 = Conv1dBnReluLayer(fc2_in_dim, emb_dim, **fc2_params)

        # Loss
        # No need when extracting embedding.
        if training:
            if margin_loss:
                self.loss = MarginSoftmaxLoss(emb_dim, num_targets, **margin_loss_params)
            else:
                self.loss = SoftmaxLoss(emb_dim, num_targets, label_smoothing=label_smoothing)

            # An example to using transform-learning without initializing loss.affine parameters
            self.transform_keys = ["repvgg", "stats", "fc1", "fc2", "loss"]

            if margin_loss and transfer_from == "softmax_loss":
                # For softmax_loss to am_softmax_loss
                self.rename_transform_keys = {"loss.affine.weight": "loss.weight"}

    @utils.for_device_free
    def get_feats(self, wavs: torch.Tensor) -> torch.Tensor:
        return self.extract_fbank(wavs)

    @utils.for_device_free
    def forward(self, inputs: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor (a batch), including [batch, frames-dim-index, frames-index]
        """
        # [batch, frames-dim-index, frames-index] -> [batch, 1, frames-dim-index, frames-index]
        x = inputs.unsqueeze(1)
        x = self.repvgg(x)
        # [batch, channel, frames-dim-index, frames-index] -> [batch, channel*frames-dim-index, frames-index]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.stats(x)
        x = self.auto(self.fc1, x)
        x = self.fc2(x)

        x = self.loss(x.float(), targets)

        return x

    @for_extract_embedding(maxChunk=10000)
    def extract_embedding(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: a 3-dimensional tensor with batch-dim = 1 or normal features matrix
        return: an 1-dimensional vector after processed by decorator
        """

        x = inputs.unsqueeze(1)
        x = self.repvgg(x)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.stats(x)

        with torch.cuda.amp.autocast(enabled=False):
            if self.extracted_embedding == "far":
                assert self.fc1 is not None
                xvector = self.fc1.affine(x)
            elif self.extracted_embedding == "near_affine":
                x = self.auto(self.fc1, x)
                xvector = self.fc2.affine(x)
            elif self.extracted_embedding == "near":
                x = self.auto(self.fc1, x)
                xvector = self.fc2(x)
            else:
                raise TypeError(
                    "Expected far or near position, but got {}".format(self.extracted_embedding)
                )

        return xvector


    # Use this for converting a RepVGG model or a bigger model with RepVGG as its component
    # Use like this
    # model = create_RepVGG_A0(deploy=False)
    # train model or load weights
    # repvgg_model_convert(model, save_path='repvgg_deploy.pth')
    # If you want to preserve the original model, call with do_copy=True

    # ====================== for using RepVGG as the backbone of a bigger model, e.g., PSPNet, the pseudo code will be like
    # train_backbone = create_RepVGG_B2(deploy=False)
    # train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
    # train_pspnet = build_pspnet(backbone=train_backbone)
    # segmentation_train(train_pspnet)
    # deploy_pspnet = repvgg_model_convert(train_pspnet)
    # segmentation_test(deploy_pspnet)
    # =====================   example_pspnet.py shows an example

    def convert_model(self, model: torch.nn.Module, save_path: Optional[str] = None, do_copy: bool = True):
        if do_copy:
            model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        if save_path is not None:
            torch.save(model.state_dict(), save_path)
        return model


if __name__ == "__main__":
    pass
