import torch
import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class Wloader(Dataset):
    """300w dataset loader."""

    def __init__(self, pt_files, img_dir, root_dir, transform=None):
        """
        Args:
            pts_files (string): Path to the pts files.
            img_files (string): Path to the pts files.
            root_dir (string): Directory with all content.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pt_file = []
        for f in os.listdir(pt_files):
            if f.endswith(".pts"):
                self.pt_file.append(f)

        self.img_file = []
        for f in os.listdir(img_dir):
            if f.endswith(".png"):
                self.img_file.append(f)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pt_file)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir,
                                self.img_file[idx])
        image = io.imread(img_name)

        pts_name = os.path.join(self.root_dir,
                                self.pt_file[idx])

        file = open(pts_name)
        landmarks = []
        temp = []

        for line in file:
            if line[0] == "v" or line[0] == "n" or line[0] == "{":
                continue
            if line[0] == "}":
                landmarks.append(temp)
                temp = []
                continue

            vals = line.split()
            vals[0] = float(vals[0])
            vals[1] = float(vals[1])
            temp.append(vals)


        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

