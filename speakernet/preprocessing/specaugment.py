# -*- coding:utf-8 -*-
"""
Copyright 2020 Snowdar
          2022 Jianchen Li
"""


import torch
import numpy as np
import torch.nn.functional as F

import speakernet.utils.utils as utils


class SpecAugment(torch.nn.Module):
    """Implement specaugment for acoustics features' augmentation but without time wraping.
    It is different to egs.augmentation.SpecAugment for all egs have a same dropout method in one batch here.

    Reference: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). 
               Specaugment: A simple data augmentation method for automatic speech recognition. arXiv 
               preprint arXiv:1904.08779.

    Likes in Compute Vision:
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks 
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation. 
               arXiv preprint arXiv:1708.04896.
    """
    def __init__(self, frequency=0.2, frame=0.2, rows=1, cols=1, random_rows=False, random_cols=False):
        super(SpecAugment, self).__init__()

        assert 0. <= frequency < 1.
        assert 0. <= frame < 1. # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows # Mask rows times for frequency.
        self.cols = cols # Mask cols times for frame.

        self.random_rows = random_rows
        self.random_cols = random_cols

        self.init = False

    def __call__(self, inputs):
        """
        @inputs: a 3-dimensional tensor, including [batch, frenquency, time]
        """
        assert len(inputs.shape) == 3

        if not self.training: return inputs

        if self.p_f > 0. or self.p_t > 0.:
            if not self.init:
                input_size = (inputs.shape[1], inputs.shape[2])
                if self.p_f > 0.:
                    self.num_f = input_size[0] # Total channels.
                    self.F = int(self.num_f * self.p_f) # Max channels to drop.
                if self.p_t > 0.:
                    self.num_t = input_size[1] # Total frames. It requires all egs with the same frames.
                    self.T = int(self.num_t * self.p_t) # Max frames to drop.
                self.init = True

            if self.p_f > 0.:
                if self.random_rows:
                    multi = np.random.randint(1, self.rows+1)
                else:
                    multi = self.rows

                for i in range(multi):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)
                    inverted_factor = self.num_f / (self.num_f - f)
                    inputs[f_0:f_0+f,:].fill_(0.)
                    inputs.mul_(inverted_factor)

            if self.p_t > 0.:
                if self.random_cols:
                    multi = np.random.randint(1, self.cols+1)
                else:
                    multi = self.cols

                for i in range(multi):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)
                    inputs[:,t_0:t_0+t].fill_(0.)

        return inputs
