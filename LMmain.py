import torch
import math
from torch import nn
import os
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
from data.FaceLandmarksDataset import FaceLandmarksDataset
from Models.LMgen import LandmarkGenNet

############################################################################################

if(torch.cuda.is_available()):
    torch.device("cuda")
else:
    torch.device("cpu")

############################################################################################

training_data = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                    root_dir='data/faces/')

net = LandmarkGenNet()

epochs = 10
batchsize = 10
learning_rate = 1e-3
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train_dataloader = DataLoader(training_data, batch_size=batchsize)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, net, loss_fn, optimizer)
print("Done!")
