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
        self.conv1 = nn.Conv2d(324,215,3)
        self.pool = nn.MaxPool2d(1,124)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)


    def forward(self, x):
        print(x)
        return x
