"""
Copyright 2019 Snowdar
          2022 Jianchen Li
"""

import numpy as np

import torch
import torch.nn.functional as F

from .activation import Nonlinearity

from speakernet.utils.utils import assign_params_dict


### There are some basic custom components/layers. ###


# ReluBatchNormLayer
class _BaseActivationBatchNorm(torch.nn.Module):
    """[Affine +] Relu + BatchNorm1d.
    Affine could be inserted by a child class.
    """

    def __init__(self):
        super(_BaseActivationBatchNorm, self).__init__()
        self.affine = None
        self.activation = None
        self.batchnorm = None

    def add_relu_bn(self, output_dim=None, options: dict = {}):
        default_params = {
            "bn-relu": False,
            "nonlinearity": 'relu',
            "nonlinearity_params": {"inplace": True, "negative_slope": 0.01},
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        default_params = assign_params_dict(default_params, options)
        self.bn_relu = default_params["bn-relu"]

        # This 'if else' is used to keep a corrected order when printing model.
        # torch.sequential is not used for I do not want too many layer wrappers and just keep structure as tdnn1.affine
        # rather than tdnn1.layers.affine or tdnn1.layers[0] etc..
        if not default_params["bn-relu"]:
            # ReLU-BN
            # For speaker recognition, relu-bn seems better than bn-relu. And w/o affine (scale and shift) of bn is
            # also better than w/ affine.
            # Assume the activation function has no parameters
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
        else:
            # BN-ReLU
            if default_params["bn"]:
                self.batchnorm = torch.nn.BatchNorm1d(output_dim, **default_params["bn_params"])
            self.activation = Nonlinearity(default_params["nonlinearity"], **default_params["nonlinearity_params"])

    def _bn_relu_forward(self, x):
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.activation is not None:
            # Assume the activation function has no parameters
            x = self.activation(x)
        return x

    def _relu_bn_forward(self, x):
        if self.activation is not None:
            # Assume the activation function has no parameters
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        x = self.affine(inputs)
        if not self.bn_relu:
            outputs = self._relu_bn_forward(x)
        else:
            outputs = self._bn_relu_forward(x)
        return outputs


class Conv1dBnReluLayer(_BaseActivationBatchNorm):
    """ Conv1d-BN-Relu
    """

    def __init__(self, input_dim, output_dim, context=[0], **options):
        super(Conv1dBnReluLayer, self).__init__()

        affine_options = {
            "bias": True,
            "groups": 1,
        }

        affine_options = assign_params_dict(affine_options, options)

        # Only keep the order: affine -> layers.insert -> add_relu_bn,
        # the structure order will be:
        # (fc2): Conv1dBnReluLayer(
        #        (affine): Conv1d()
        #        (activation): ReLU()
        #        (batchnorm): BatchNorm1d()
        dilation = context[1] - context[0] if len(context) > 1 else 1
        for i in range(1, len(context) - 1):
            assert dilation == context[i + 1] - context[i]

        left_context = context[0] if context[0] < 0 else 0
        right_context = context[-1] if context[-1] > 0 else 0
        receptive_field_size = right_context - left_context + 1
        padding = receptive_field_size // 2

        self.affine = torch.nn.Conv1d(input_dim, output_dim, kernel_size=len(context),
                                      stride=1, padding=padding, dilation=dilation,
                                      groups=affine_options["groups"],
                                      bias=affine_options["bias"])

        self.add_relu_bn(output_dim, options=options)

        # Implement forward function extrally if needed when forward-graph is changed.


class AdaptivePCMN(torch.nn.Module):
    """ Using adaptive parametric Cepstral Mean Normalization to replace traditional CMN.
        It is implemented according to [Ozlem Kalinli, etc. "Parametric Cepstral Mean Normalization 
        for Robust Automatic Speech Recognition", icassp, 2019.]
    """

    def __init__(self, input_dim, left_context=-10, right_context=10, pad=True):
        super(AdaptivePCMN, self).__init__()

        assert left_context < 0 and right_context > 0

        self.left_context = left_context
        self.right_context = right_context
        self.tot_context = self.right_context - self.left_context + 1

        kernel_size = (self.tot_context,)

        self.input_dim = input_dim
        # Just pad head and end rather than zeros using replicate pad mode
        # or set pad false with enough context egs.
        self.pad = pad
        self.pad_mode = "replicate"

        self.groups = input_dim
        output_dim = input_dim

        # The output_dim is equal to input_dim and keep every dims independent by using groups conv.
        self.beta_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.alpha_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.mu_n_0_w = torch.nn.Parameter(torch.randn(
            output_dim, input_dim//self.groups, *kernel_size))
        self.bias = torch.nn.Parameter(torch.randn(output_dim))

        # init weight and bias. It is important
        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.beta_w, 0., 0.01)
        torch.nn.init.normal_(self.alpha_w, 0., 0.01)
        torch.nn.init.normal_(self.mu_n_0_w, 0., 0.01)
        torch.nn.init.constant_(self.bias, 0.)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        assert inputs.shape[2] >= self.tot_context

        if self.pad:
            pad_input = F.pad(inputs, (-self.left_context,
                                       self.right_context), mode=self.pad_mode)
        else:
            pad_input = inputs
            inputs = inputs[:, :, -self.left_context:-self.right_context]

        # outputs beta + 1 instead of beta to avoid potentially zeroing out the inputs cepstral features.
        self.beta = F.conv1d(pad_input, self.beta_w,
                             bias=self.bias, groups=self.groups) + 1
        self.alpha = F.conv1d(pad_input, self.alpha_w,
                              bias=self.bias, groups=self.groups)
        self.mu_n_0 = F.conv1d(pad_input, self.mu_n_0_w,
                               bias=self.bias, groups=self.groups)

        outputs = self.beta * inputs - self.alpha * self.mu_n_0

        return outputs


class SEBlock1d(torch.nn.Module):
    """ A SE Block layer which can learn to use global information to selectively emphasise informative 
    features and suppress less useful ones.
    This is a pytorch implementation of SE Block based on the paper:
    Squeeze-and-Excitation Networks
    by JFChou xmuspeech 2019-07-13
       Snowdar xmuspeech 2020-04-28 [Check and update]
       lijianchen 2020-11-18
    """

    def __init__(self, input_dim, ratio=16, inplace=True):
        '''
        @ratio: a reduction ratio which allows us to vary the capacity and computational cost of the SE blocks 
        in the network.
        '''
        super(SEBlock1d, self).__init__()

        self.input_dim = input_dim

        self.fc_1 = torch.nn.Linear(input_dim, input_dim // ratio, bias=False)
        self.fc_2 = torch.nn.Linear(input_dim // ratio, input_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.fc_1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.xavier_normal_(self.fc_2.weight, gain=1.0)
        self.relu = torch.nn.ReLU(inplace=inplace)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        x = inputs.mean(dim=2, keepdim=True)
        x = self.relu(self.fc_1(x))
        scale = self.sigmoid(self.fc_2(x))

        return inputs * scale


class Mixup(torch.nn.Module):
    """Implement a mixup component to augment data and increase the generalization of model training.
    Reference:
        [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. n.d. Mixup: BEYOND EMPIRICAL RISK MINIMIZATION.
        [2] Zhu, Yingke, Tom Ko, and Brian Mak. 2019. “Mixup Learning Strategies for Text-Independent Speaker Verification.”

    Github: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
    """

    def __init__(self, alpha=1.0):
        super(Mixup, self).__init__()

        self.alpha = alpha

    def forward(self, inputs):
        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        self.lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0. else 1.
        # Shuffle the original index to generate the pairs, such as
        # Origin:           1 2 3 4 5
        # After Shuffling:  3 4 1 5 2
        # Then the pairs are (1, 3), (2, 4), (3, 1), (4, 5), (5,2).
        self.index = torch.randperm(batch_size, device=inputs.device)

        mixed_data = self.lam * inputs + (1 - self.lam) * inputs[self.index, :]

        return mixed_data


class Res2Conv1dReluBn(torch.nn.Module):
    """
    Res2Conv1d + BatchNorm1d + ReLU

    in_channels == out_channels == channels
    """

    def __init__(self, channels, context=[0], bias=True, scale=4, tdnn_params={}):
        super().__init__()
        default_tdnn_params = {
            "nonlinearity": "relu",
            "nonlinearity_params": {"inplace": True},
            "bn-relu": False,
            "bn": True,
            "bn_params": {"momentum": 0.1, "affine": True, "track_running_stats": True},
        }

        tdnn_params = assign_params_dict(default_tdnn_params, tdnn_params)

        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.blocks = torch.nn.ModuleList(
            [
                Conv1dBnReluLayer(self.width, self.width, context, **tdnn_params, bias=bias)
                for i in range(self.nums)
            ]
        )

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.blocks[i](sp)
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class SEBlock2d(torch.nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock2d, self).__init__()
        self.down = torch.nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = torch.nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class SE_Connect(torch.nn.Module):
    def __init__(self, channels, s=4):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        # assert channels // s == 128
        self.linear1 = torch.nn.Linear(channels, channels // s)
        self.linear2 = torch.nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SE_Res2Block(torch.nn.Module):
    """ SE-Res2Block.
    """

    def __init__(self, channels, context, scale, tdnn_layer_params={}):
        super().__init__()
        self.se_res2block = torch.nn.Sequential(
            # Order: conv -> relu -> bn
            Conv1dBnReluLayer(channels, channels, context=[0], **tdnn_layer_params),
            Res2Conv1dReluBn(channels, context, scale=scale, tdnn_params=tdnn_layer_params,),
            # Order: conv -> relu -> bn
            Conv1dBnReluLayer(channels, channels, context=[0], **tdnn_layer_params),
            # SEBlock(channels, ratio=4),
            SE_Connect(channels),
        )

    def forward(self, x):
        return x + self.se_res2block(x)
