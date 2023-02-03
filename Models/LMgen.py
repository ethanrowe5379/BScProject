import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset


class LandmarkGenNet(nn.Module):
    "Landmark/face structure recognition Generator"
    def __init__(self):
        super().__init__()
        # input image channel, output channels, square convolution
        self.Conv1 = nn.Conv2d(3,16,5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x
