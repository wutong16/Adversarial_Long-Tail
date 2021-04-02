from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
import functools
from torch.nn import Module


class NoNorm(nn.BatchNorm2d):
    """
    This is just placeholder, used when no norm is intended to use
    """
    def forward(self, x):
        return x


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_tracked_mean=True, use_tracked_var=True):
        nn.BatchNorm2d.__init__(self, num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                track_running_stats=track_running_stats)
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_var = use_tracked_var

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        sigma2 = y.var(dim=1)

        if self.training is not True:
            if self.use_tracked_mean:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_var:
                y = y / (self.running_var.view(-1, 1)**.5 + self.eps)
            else:
                y = y / (sigma2.view(-1, 1)**.5 + self.eps)

        elif self.training is True:
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                    self.running_var = (1-self.momentum)*self.running_var + self.momentum*sigma2
            y = y - mu.view(-1, 1)
            y = y / (sigma2.view(-1, 1)**.5 + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)


class RobustNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, use_tracked_mean=True, use_tracked_range=True, power=0.2):
        nn.BatchNorm2d.__init__(self, num_features=num_features, eps=eps, momentum=momentum, affine=affine,
                                track_running_stats=track_running_stats)
        self.power = power
        self.use_tracked_mean = use_tracked_mean
        self.use_tracked_range = use_tracked_range

    def forward(self, x):
        self._check_input_dim(x)
        y = x.transpose(0, 1)
        return_shape = y.shape
        y = y.contiguous().view(x.size(1), -1)
        mu = y.mean(dim=1)
        min = y.min(dim=1)[0]
        max = y.max(dim=1)[0]
        range = torch.sub(max, min)

        # during validation, whether tracked stat to be used or not
        if self.training is not True:
            if self.use_tracked_mean is True:
                y = y - self.running_mean.view(-1, 1)
            else:
                y = y - mu.view(-1, 1)
            if self.use_tracked_range is True:
                y = y / (self.running_var.view(-1, 1)**self.power + self.eps)
            else:
                y = y / (range.view(-1, 1)**self.power + self.eps)

        # during training tracking will be always be used
        elif self.training is True:
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
                self.running_var = (1-self.momentum)*self.running_var + self.momentum*range
            y = y - mu.view(-1, 1)
            y = y / (range.view(-1, 1)**self.power + self.eps)

        y = self.weight.view(-1, 1) * y + self.bias.view(-1, 1)
        return y.view(return_shape).transpose(0, 1)





def select_norm(norm_type, norm_power=0.2):
    if norm_type == 'NoBN':
        normlayer = functools.partial(NoNorm)
    # BatchNorm variants
    elif norm_type == 'BN':
        normlayer = functools.partial(BatchNorm, affine=True, track_running_stats=True, use_tracked_mean=True,
                                      use_tracked_var=True)
    elif norm_type == 'BNFT':
        normlayer = functools.partial(BatchNormFT, affine=True, track_running_stats=True)
    elif norm_type == 'BNwoA':
        normlayer = functools.partial(nn.BatchNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'GNR':
        normlayer = functools.partial(GroupNormR, affine=False)
    elif norm_type == 'BNwoT':
        normlayer = functools.partial(BatchNorm, affine=True, track_running_stats=True, use_tracked_mean=False,
                                      use_tracked_var=False)
    elif norm_type == 'BNM':
        normlayer = functools.partial(BatchNorm, affine=True, track_running_stats=True, use_tracked_mean=True,
                                      use_tracked_var=False)

    # RobustNorm variatns
    elif norm_type == 'RN':
        normlayer = functools.partial(RobustNorm, affine=True, power=norm_power, track_running_stats=True,
                                      use_tracked_mean=False, use_tracked_range=False)


    elif norm_type == 'RNT':
        normlayer = functools.partial(RobustNorm, affine=True, power=norm_power, track_running_stats=True,
                                      use_tracked_mean=True, use_tracked_range=True)
    elif norm_type == 'RNM':
        normlayer = functools.partial(RobustNorm, affine=True, power=norm_power, track_running_stats=True,
                                      use_tracked_mean=True, use_tracked_range=False)
    else:
        Exception("Norm is not selected")

    return normlayer


if __name__ == '__main__':
    a = select_norm("RNT")
    a(2)
    print(a)
