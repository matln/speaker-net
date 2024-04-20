# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""

import torch

from .components import Conv1dBnReluLayer

# Pooling ✿


class StatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""

    def __init__(self, input_dim, stddev=True, unbiased=False, eps=1.0e-10):
        super(StatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev:
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean) ** 2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim

    def extra_repr(self):
        return "{input_dim}, {output_dim}, stddev={stddev}, unbiased={unbiased}, eps={eps}".format(
            **self.__dict__
        )

    @classmethod
    def thop_count(self, m, x, y):
        pass
        # To do
        # x = x[0]

        # kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        # bias_ops = 1 if m.bias is not None else 0

        # # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        # total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        # m.total_ops += torch.DoubleTensor([int(total_ops)])


class FreeStatisticsPooling(torch.nn.Module):
    """ An usual mean [+ stddev] poolling layer"""

    def __init__(self, stddev=True, unbiased=False, eps=1.0e-10):
        super(FreeStatisticsPooling, self).__init__()

        self.stddev = stddev

        self.eps = eps
        # Used for unbiased estimate of stddev
        self.unbiased = unbiased

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """

        inputs = inputs.reshape(inputs.shape[0], -1, inputs.shape[len(inputs.shape) - 1])

        # Get the num of frames
        counts = inputs.shape[2]

        mean = inputs.sum(dim=2, keepdim=True) / counts

        if self.stddev:
            if self.unbiased and counts > 1:
                counts = counts - 1

            # The sqrt (as follows) is deprecated because it results in Nan problem.
            # std = torch.unsqueeze(torch.sqrt(torch.sum((inputs - mean)**2, dim=2) / counts), dim=2)
            # There is a eps to solve this problem.
            # Another method: Var is equal to std in "cat" way, actually. So, just use Var directly.

            var = torch.sum((inputs - mean) ** 2, dim=2, keepdim=True) / counts
            std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean


class LDEPooling(torch.nn.Module):
    """A novel learnable dictionary encoding layer.
    Reference: Weicheng Cai, etc., "A NOVEL LEARNABLE DICTIONARY ENCODING LAYER FOR END-TO-END
               LANGUAGE IDENTIFICATION", icassp, 2018
    """

    def __init__(self, input_dim, c_num=64, eps=1.0e-10):
        super(LDEPooling, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim * c_num
        self.eps = eps

        self.mu = torch.nn.Parameter(torch.randn(input_dim, c_num))
        self.s = torch.nn.Parameter(torch.ones(c_num))

        self.softmax_for_w = torch.nn.Softmax(dim=3)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        r = inputs.transpose(1, 2).unsqueeze(3) - self.mu
        # Make sure beta=self.s**2+self.eps > 0
        w = self.softmax_for_w(-(self.s ** 2 + self.eps) * torch.sum(r ** 2, dim=2, keepdim=True))
        e = torch.mean(w * r, dim=1)

        return e.reshape(-1, self.output_dim, 1)

    def get_output_dim(self):
        return self.output_dim


# Attention-based
class AttentionAlphaComponent(torch.nn.Module):
    """Compute the alpha with attention module.
            alpha = softmax(v'·f(w·x + b) + k) or softmax(v'·x + k)
    where f is relu here and bias could be lost.
    Support:
            1. Single or Multi-head attention
            2. One affine or two affine
            3. Share weight (last affine = vector) or un-shared weight (last affine = matrix)
            4. Self-attention or time context attention (supported by context parameter of TdnnAffine)
            5. Different temperatures for different heads.
    """

    def __init__(
        self,
        input_dim,
        num_head=1,
        split_input=True,
        share=True,
        affine_layers=2,
        hidden_size=64,
        context=[0],
        bias=True,
        temperature=False,
        nonlinearity="relu",
        fixed=True,
        global_context_att=False,
    ):
        super(AttentionAlphaComponent, self).__init__()
        assert num_head >= 1
        # Multi-head case.
        if num_head > 1:
            if split_input:
                # Make sure fatures/planes with input_dim dims could be splited to num_head parts.
                assert input_dim % num_head == 0
            if temperature:
                if fixed:
                    t_list = []
                    for i in range(num_head):
                        t_list.append([[max(1, (i // 2) * 5)]])
                    # shape [1, num_head, 1, 1]
                    self.register_buffer("t", torch.tensor([t_list]))
                else:
                    # Different heads have different temperature.
                    # Use 1 + self.t**2 in forward to make sure temperature >= 1.
                    self.t = torch.nn.Parameter(torch.zeros(1, num_head, 1, 1))

        self.input_dim = input_dim
        self.num_head = num_head
        self.split_input = split_input
        self.share = share
        self.temperature = temperature
        self.fixed = fixed

        if share:
            # weight: [input_dim, 1] or [input_dim, hidden_size] -> [hidden_size, 1]
            final_dim = 1
        elif not global_context_att:
            if split_input:
                # weight: [input_dim, input_dim // num_head] or [input_dim, hidden_size] -> [hidden_size, input_dim // num_head]
                final_dim = input_dim // num_head
            else:
                # weight: [input_dim, input_dim] or [input_dim, hidden_size] -> [hidden_size, input_dim]
                final_dim = input_dim
        else:
            if split_input:
                final_dim = input_dim // num_head // 3
            else:
                final_dim = input_dim // 3

        first_groups = 1
        last_groups = 1

        if affine_layers == 1:
            last_affine_input_dim = input_dim
            # (x, 1) for global case and (x, h) for split case.
            if num_head > 1 and split_input:
                last_groups = num_head
            self.relu_affine = False
        elif affine_layers == 2:
            last_affine_input_dim = hidden_size * num_head
            if num_head > 1:
                # (1, h) for global case and (h, h) for split case.
                last_groups = num_head
                if split_input:
                    first_groups = num_head
            # Add a relu-affine with affine_layers=2.
            self.relu_affine = True
            self.first_affine = Conv1dBnReluLayer(
                input_dim,
                last_affine_input_dim,
                context=context,
                bias=bias,
                groups=first_groups,
                nonlinearity=nonlinearity,
                bn=False,
            )
        else:
            raise ValueError("Expected 1 or 2 affine layers, but got {}.", format(affine_layers))

        self.last_affine = Conv1dBnReluLayer(
            last_affine_input_dim,
            final_dim * num_head,
            context=context,
            bias=bias,
            groups=last_groups,
            nonlinearity="",
            bn=False,
        )
        # Dim=2 means to apply softmax in different frames-index (batch is a 3-dim tensor in this case).
        self.softmax = torch.nn.Softmax(dim=2)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        if self.temperature:
            batch_size = inputs.shape[0]
            chunk_size = inputs.shape[2]

        x = inputs
        if self.relu_affine:
            # x = self.relu(self.first_affine(x))
            x = self.first_affine(x)
        if self.num_head > 1 and self.temperature:
            if self.fixed:
                t = self.t
            else:
                t = 1 + self.t ** 2
            x = self.last_affine(x).reshape(batch_size, self.num_head, -1, chunk_size) / t
            return self.softmax(x.reshape(batch_size, -1, chunk_size))
        else:
            return self.softmax(self.last_affine(x))


class AttentiveStatisticsPooling(torch.nn.Module):
    """ An attentive statistics pooling.
    Reference: Okabe, Koji, Takafumi Koshinaka, and Koichi Shinoda. 2018. "Attentive Statistics Pooling
               for Deep Speaker Embedding." ArXiv Preprint ArXiv:1803.10963.
    """

    def __init__(
        self,
        input_dim,
        affine_layers=2,
        hidden_size=64,
        context=[0],
        stddev=True,
        stddev_attention=True,
        eps=1.0e-10,
    ):
        super(AttentiveStatisticsPooling, self).__init__()

        self.stddev = stddev
        self.input_dim = input_dim

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim

        self.eps = eps
        self.stddev_attention = stddev_attention

        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=1,
            share=True,
            affine_layers=affine_layers,
            hidden_size=hidden_size,
            context=context,
        )

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        alpha = self.attention(inputs)

        # Weight avarage
        mean = torch.sum(alpha * inputs, dim=2, keepdim=True)

        if self.stddev:
            if self.stddev_attention:
                var = torch.sum(alpha * inputs ** 2, dim=2, keepdim=True) - mean ** 2
                std = torch.sqrt(var.clamp(min=self.eps))
            else:
                var = torch.mean((inputs - mean) ** 2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=self.eps))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim


class MultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Safari, Pooyan, and Javier Hernando. 2019. “Self Multi-Head Attention for Speaker
               Recognition.” ArXiv Preprint ArXiv:1906.09890.
    Note, in this paper, affine_layers is default to 1, and final_dim is 1 which means the weights are shared.
    """

    def __init__(
        self,
        input_dim,
        stddev=True,
        stddev_attention=True,
        num_head=4,
        share=True,
        affine_layers=1,
        **options
    ):
        super(MultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        self.num_head = num_head

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if not options["split_input"]:
                raise ValueError(
                    "split_input==False is not valid for this MultiHeadAttentionPooling."
                )
            options.pop("split_input")

        # In this pooling, the special point is that inputs will be splited.
        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=num_head,
            split_input=True,
            share=share,
            affine_layers=affine_layers,
            bias=False,
            **options
        )

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2]  # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, splited-features, frames]
        # for another case.
        # inputs: [batch, head, splited-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * inputs.reshape(
            batch_size, self.num_head, -1, chunk_size
        )

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev:
            if self.stddev_attention:
                after_mul_2 = (
                    alpha.reshape(batch_size, self.num_head, -1, chunk_size)
                    * inputs.reshape(batch_size, self.num_head, -1, chunk_size) ** 2
                )
                var = (
                    torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)
                    - mean ** 2
                )
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean) ** 2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim


class GlobalMultiHeadAttentionPooling(torch.nn.Module):
    """Implement global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    It is not equivalent to multi-head attention pooling even when
               input_dim of global multi-head = 1/num_head * input_dim of multi-head.
    """

    def __init__(
        self,
        input_dim,
        stddev=True,
        stddev_attention=True,
        num_head=4,
        share=True,
        affine_layers=2,
        **options
    ):
        super(GlobalMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError(
                    "split_input==True is not valid for GlobalMultiHeadAttentionPooling."
                )
            options.pop("split_input")
        if "temperature" in options.keys():
            if options["temperature"]:
                raise ValueError(
                    "temperature==True is not valid for GlobalMultiHeadAttentionPooling."
                )
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used.
        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=num_head,
            split_input=False,
            share=share,
            temperature=False,
            affine_layers=affine_layers,
            bias=True,
            **options
        )

    def forward(self, inputs):
        """
        inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2]  # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * inputs.reshape(
            batch_size, 1, -1, chunk_size
        )

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev:
            if self.stddev_attention:
                after_mul_2 = (
                    alpha.reshape(batch_size, self.num_head, -1, chunk_size)
                    * inputs.reshape(batch_size, 1, -1, chunk_size) ** 2
                )
                var = (
                    torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)
                    - mean ** 2
                )
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean) ** 2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head


class MultiResolutionMultiHeadAttentionPooling(torch.nn.Module):
    """Implement multi-resolution global multi-head attention pooling based on AttentionAlphaComponent.
    Reference: Zhiming Wang, Kaisheng Yao, Xiaolong Li, Shuo Fang. "MULTI-RESOLUTION MULTI-HEAD
               ATTENTION IN DEEP SPEAKER EMBEDDING." ICASSP, 2020.
    """

    def __init__(
        self,
        input_dim,
        stddev=True,
        stddev_attention=True,
        num_head=4,
        share=True,
        affine_layers=2,
        **options
    ):
        super(MultiResolutionMultiHeadAttentionPooling, self).__init__()

        self.input_dim = input_dim
        self.num_head = num_head
        self.stddev = stddev
        self.stddev_attention = stddev_attention

        if self.stddev:
            self.output_dim = 2 * input_dim
        else:
            self.output_dim = input_dim

        if "split_input" in options.keys():
            if options["split_input"]:
                raise ValueError(
                    "split_input==True is not valid for MultiResolutionMultiHeadAttentionPooling."
                )
            options.pop("split_input")
        if "temperature" in options.keys():
            if not options["temperature"]:
                raise ValueError(
                    "temperature==False is not valid for MultiResolutionMultiHeadAttentionPooling."
                )
            options.pop("temperature")

        # In this pooling, the special point is that all (global) features of inputs will be used and
        # the temperature will be added.
        self.attention = AttentionAlphaComponent(
            input_dim,
            num_head=num_head,
            split_input=False,
            temperature=True,
            share=share,
            affine_layers=affine_layers,
            bias=True,
            **options
        )

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2]  # a.k.a total frames

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(inputs)

        # In sharing weight case, the shape of alpha is [batch, head, 1, frames] and [batch, head, all-features, frames]
        # for another case.
        # inputs: [batch, 1, all-features, frames]
        after_mul = alpha.reshape(batch_size, self.num_head, -1, chunk_size) * inputs.reshape(
            batch_size, 1, -1, chunk_size
        )

        # After multi-multipling alpha and inputs for multi-head case, the mean could be got by reshaping back.
        mean = torch.sum(after_mul.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)

        if self.stddev:
            if self.stddev_attention:
                after_mul_2 = (
                    alpha.reshape(batch_size, self.num_head, -1, chunk_size)
                    * inputs.reshape(batch_size, 1, -1, chunk_size) ** 2
                )
                var = (
                    torch.sum(after_mul_2.reshape(batch_size, -1, chunk_size), dim=2, keepdim=True)
                    - mean ** 2
                )
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean) ** 2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        return self.output_dim * self.num_head


class ChannelContextStatisticsPooling(torch.nn.Module):
    """ Attentive statistics pooling: Channel- and context-dependent
        statistics pooling, first used in ECAPA_TDNN.
    """

    def __init__(
        self,
        input_dim,
        stddev=True,
        stddev_attention=True,
        num_head=1,
        share=False,
        affine_layers=2,
        hidden_size=128,
        nonlinearity="tanh",
        global_context_att=True,
        **options
    ):
        super(ChannelContextStatisticsPooling, self).__init__()
        self.input_dim = input_dim
        self.global_context_att = global_context_att
        self.stddev = stddev
        self.stddev_attention = stddev_attention
        assert num_head == 1

        if global_context_att:
            _input_dim = input_dim * 3
        else:
            _input_dim = input_dim

        # ReLU may be hard to converge.
        self.attention = AttentionAlphaComponent(
            _input_dim,
            num_head=num_head,
            split_input=False,
            share=False,
            hidden_size=hidden_size,
            affine_layers=affine_layers,
            nonlinearity=nonlinearity,
            bias=True,
            global_context_att=global_context_att,
            **options
        )

    def forward(self, inputs):
        """
        inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        batch_size = inputs.shape[0]
        chunk_size = inputs.shape[2]  # a.k.a total frames

        if self.global_context_att:
            context_mean = torch.mean(inputs, dim=-1, keepdim=True).expand_as(inputs)
            context_std = torch.sqrt(torch.var(inputs, dim=-1, keepdim=True) + 1e-10).expand_as(inputs)
            _inputs = torch.cat((inputs, context_mean, context_std), dim=1)
        else:
            _inputs = inputs

        # alpha: [batch, weight, frames]
        # When using the conv1d to implement the multi-multiple of multi-head, we can get
        # the weight distribution of multi-head: [h11, h12, h13, h21, h22, h23, ..., hn1, hn2, ...]
        # So, just reshape it to split different heads.
        alpha = self.attention(_inputs)

        mean = torch.sum(alpha * inputs, dim=2, keepdim=True)

        if self.stddev:
            if self.stddev_attention:
                var = (
                    torch.sum(alpha * (inputs ** 2), dim=2, keepdim=True)
                    - mean ** 2
                )
                std = torch.sqrt(var.clamp(min=1.0e-10))
            else:
                var = torch.mean((inputs - mean) ** 2, dim=2, keepdim=True)
                std = torch.sqrt(var.clamp(min=1.0e-10))
            return torch.cat((mean, std), dim=1)
        else:
            return mean

    def get_output_dim(self):
        if self.stddev:
            return 2 * self.input_dim
        else:
            return self.input_dim
