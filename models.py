"""
Example Models that use weight quantization layers to reduce model size while training.
"""

import math
import operator
from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn

from modules import QConv2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = QConv2d(1, 32, 5)
        self.conv2 = QConv2d(32, 32, 5)
        self.conv3 = QConv2d(32, 64, 3)
        self.conv4 = QConv2d(64, 64, 3)

        self.bnorm1 = nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.bnorm2 = nn.BatchNorm2d(64, affine=False, track_running_stats=False)

        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))

        self.final_conv = QConv2d(576, 10, 1)

    

    def forward(self, x): 
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.bnorm1(out)
        out = self.maxpool1(out)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.bnorm2(out)
        out = self.maxpool2(out)

        out = torch.flatten(out, 1).reshape(-1, 576, 1, 1) # 576 in channels, 10 out channels
        out = self.final_conv(out)
        
        out = torch.flatten(out, 1)
        return out