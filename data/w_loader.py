import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from skimage import io, transform

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        img = np.float32(img)
        img = torch.tensor(img)
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        lm = np.array(landmarks)
        lm = lm * [new_w / w, new_h / h]
        lm = np.float32(lm)
        lm = torch.tensor(lm)

        return {'image': img, 'landmarks': lm}

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

        res = Rescale((512, 512))

        sample = {'image': image, 'landmarks': landmarks}

        return res(sample)


