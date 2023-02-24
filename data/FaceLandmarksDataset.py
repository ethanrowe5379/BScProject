import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.image as mpimg

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        print(self.landmarks_frame)
        print(self.root_dir)
        print(self.transform)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = mpimg.imread(img_name)

        # if image has an alpha color channel, get rid of it
        if (image.shape[2] == 4):
            image = image[:, :, 0:3]

        image = image/255

        landmarks = self.landmarks_frame.iloc[idx, 1:].values
        landmarks = landmarks.astype('float').reshape(-1, 2)
        landmarks = (landmarks - 100)/50.0
        sample = {'image': image, 'landmarks': landmarks}

        #if self.transform:
            #sample = self.transform(sample)

        return sample

