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
        # kernel
        self.Conv1 = nn.Conv2d(512, 256, 3, padding=2)
        self.Conv2 = nn.Conv2d(256, 128, 3, padding=2)
        self.Conv3 = nn.Conv2d(128, 64, 3, padding=2)
        self.pool = nn.MaxPool2d(3, 2)
        # an affine operation: y = Wx + b
        # (activation shape, activation size)
        self.fc1 = nn.Linear(64, 192)# 5*5 from image dimension
        #self.fc2 = nn.Linear(120, 84)  # 5*5 from image dimension
        #self.fc3 = nn.Linear(84, 64)
    def forward(self, x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = self.pool(F.relu(self.Conv2(x)))
        x = self.pool(F.relu(self.Conv3(x)))
        x = torch.flatten(x, 2)
        x = F.relu(self.fc1(x))
        return x
