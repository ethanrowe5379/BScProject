import torch
from torch import nn
import torch.optim as optim
from data.w_loader import Wloader
from Models.LMgen import LandmarkGenNet

############################################################################################

if(torch.cuda.is_available()):
    torch.device("cuda")
else:
    torch.device("cpu")

############################################################################################

training_data = Wloader(pt_files='data/300W/01_Indoor/',
                        img_dir='data/300W/01_Indoor/',
                        root_dir='data/300W/01_Indoor/')

print(training_data[0])

trainloader = torch.utils.data.DataLoader(training_data, batch_size=1,
                                          shuffle=True, num_workers=0)

net = LandmarkGenNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')