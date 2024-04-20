"""
RepVGG: Making VGG-style ConvNets Great Again (https://openaccess.thecvf.com/content/CVPR2021/papers/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.pdf)
Github source: https://github.com/DingXiaoH/RepVGG
Licensed under The MIT License [see LICENSE for details]
"""

import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from speakernet.models.modules.components import SEBlock2d

logger = logging.getLogger(__name__)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, norm_layer_params={}):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels, **norm_layer_params))
    return result


class RepVGGBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm_layer_params={},
        padding_mode="zeros",
        deploy=False,
        use_se=False,
        use_post_se=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1
        # Cannot be True at the same time
        assert (use_se and use_post_se) is not True

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            # Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock2d(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        # For RepVGGplus
        if use_post_se:
            self.post_se = SEBlock2d(out_channels, internal_neurons=out_channels // 4)
        else:
            self.post_se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels, **norm_layer_params)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                norm_layer_params=norm_layer_params
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                norm_layer_params=norm_layer_params
            )
            # logger.info(f"RepVGG Block, identity = {self.rbr_identity}")

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.post_se(self.nonlinearity(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.post_se(self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    # Optional. This may improve the accuracy and facilitates quantization in some cases.
    # 1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    # 2.  Use like this.
    #     loss = criterion(....)
    #     for every RepVGGBlock blk:
    #         loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #     optimizer.zero_grad()
    #     loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (
            (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt()))
            .reshape(-1, 1, 1, 1)
            .detach()
        )
        t1 = (
            (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt()))
            .reshape(-1, 1, 1, 1)
            .detach()
        )

        # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        l2_loss_circle = (K3**2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()
        # The equivalent resultant central point of 3x3 kernel.
        eq_kernel = (K3[:, :, 1:2, 1:2] * t3 + K1 * t1)
        # Normalize for an L2 coefficient comparable to regular L2.
        l2_loss_eq_kernel = (eq_kernel**2 / (t3**2 + t1**2)).sum()
        return l2_loss_eq_kernel + l2_loss_circle

    # This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    # You can get the equivalent kernel and bias at any time and do whatever you want,
    # for example, apply some penalties or constraints during training, just like you do to the other models.
    # May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class RepVGG(nn.Module):
    def __init__(
        self,
        num_blocks,
        planes=None,
        strides=[1, 1, 2, 2, 2],
        norm_layer_params={},
        override_groups_map=None,
        deploy=False,
        use_se=False,
        use_post_se=False,
        use_checkpoint=False,
    ):
        super(RepVGG, self).__init__()
        assert len(planes) == 4
        assert len(strides) == 5
        self.strides = strides
        self.planes = planes
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.norm_layer_params = norm_layer_params
        self.use_se = use_se
        self.use_post_se = use_post_se
        self.use_checkpoint = use_checkpoint

        self.in_planes = min(64, planes[0])
        self.stage0 = RepVGGBlock(
            in_channels=1,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=self.strides[0],
            padding=1,
            norm_layer_params=self.norm_layer_params,
            deploy=self.deploy,
            use_se=self.use_se,
            use_post_se=self.use_post_se,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(planes[0], num_blocks[0], stride=self.strides[1])
        self.stage2 = self._make_stage(planes[1], num_blocks[1], stride=self.strides[2])
        self.stage3 = self._make_stage(planes[2], num_blocks[2], stride=self.strides[3])
        self.stage4 = self._make_stage(planes[3], num_blocks[3], stride=self.strides[4])

        if "affine" in self.norm_layer_params.keys():
            norm_layer_affine = self.norm_layer_params["affine"]
        else:
            norm_layer_affine = True  # The default is True

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # torch.nn.init.normal_(m.weight, 0., 0.01)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif (
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)) and norm_layer_affine
            ):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    norm_layer_params=self.norm_layer_params,
                    groups=cur_groups,
                    deploy=self.deploy,
                    use_se=self.use_se,
                    use_post_se=self.use_post_se,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.ModuleList(blocks)

    def forward(self, x):
        out = self.stage0(x)
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            for block in stage:
                if self.use_checkpoint:
                    out = checkpoint.checkpoint(block, out)
                else:
                    out = block(out)
        return out
