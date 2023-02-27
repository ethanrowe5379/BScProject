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
        self.Conv1 = nn.Conv2d(3, 18, 3, padding=2)
        self.Conv2 = nn.Conv2d(18, 34, 3, padding=2)
        self.Conv3 = nn.Conv2d(34, 68, 3, padding=2)
        self.pool = nn.MaxPool2d(3, 2)
        # an affine operation: y = Wx + b
        # (activation shape, activation size)
        self.fc1 = nn.Linear(4096, 1024)# 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 512)  # 5*5 from image dimension
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)
    def forward(self, x):
        x = self.pool(F.relu(self.Conv1(x)))
        x = self.pool(F.relu(self.Conv2(x)))
        x = self.pool(F.relu(self.Conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x
