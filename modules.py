"""
Custom layers that implement weight quantization
Equation 1 in the paper.
"""

import math
import operator
from functools import reduce

import torch
import torch.nn.functional as F
from torch.nn import Module

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def prod(x):
    return reduce(operator.mul, x, 1)

class QConv2d(Module):
    """
    Quantized 2d convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        self.kernel_size = cast_tuple(kernel_size, length=2)
        scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
        self.weight = torch.FloatTensor(out_channels, in_channels, *self.kernel_size).uniform_(-scale, scale)
        self.e = torch.full((out_channels, 1, 1, 1), -8.)
        self.b = torch.full((out_channels, 1, 1, 1), 2.) # 2 bits per weight

    def qbits(self):
        return F.relu(self.b).sum() * prod(self.weight.shape[1:])
    
    def qweight(self):
        """
        Quantizing weights via differentiable min max rounding (Equation 1 in paper).
        Independent of input tensor, we are basically adding a regularization factor based on the 
        amount of parameters in the model.
        """
        out = torch.minimum(
            torch.maximum(
                2**-self.e * self.weight,
                -2**(F.relu(self.b)-1)
            ),
            2**(F.relu(self.b)-1) - 1
        )

    def forward(self, x):
        qw = self.qweight() # quantized weight is independent of input tensor
        w = (qw.round() - qw).detach() + qw  # straight through estimator
        return F.conv2d(x, weight=2**self.e*w)

