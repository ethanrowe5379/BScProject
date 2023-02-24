import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from data.w_loader import Wloader
from Models.LMgen import LandmarkGenNet
import matplotlib.pyplot as plt



############################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

############################################################################################


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.show()


############################################################################################

training_data = Wloader(pt_files='data/300W/01_Indoor/',
                        img_dir='data/300W/01_Indoor/',
                        root_dir='data/300W/01_Indoor/')

trainloader = torch.utils.data.DataLoader(training_data, batch_size=5,
                                          shuffle=True, num_workers=0)


testingloader = torch.utils.data.DataLoader(training_data, batch_size=5,
                                          shuffle=True, num_workers=0)

############################################################################################

net = LandmarkGenNet()

criterion = nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001)
scheduler = ExponentialLR(optimizer, gamma=0.9)

############################################################################################

images, landmarks = next(iter(trainloader))

for epoch in range(5):  # loop over the dataset multiple times
    print(f"Epoch {epoch + 1}\n-------------------------------")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # Inputs
        inputs = data['image'][i].requires_grad_()
        target = data['landmarks'][i].requires_grad_()
        # forward + backward + optimize
        outputs = net(inputs)
        # loss = criterion(outputs[0], target)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
        # optimizer.step()
        #
        # # Running loss
        # running_loss += loss.item()
        # print(f'[{i + 1}] loss: {running_loss / 5:.3f}')
        # running_loss = 0.0
        #
        # if i == 4:
        #     break
    #
    # scheduler.step()

############################################################################################

# for i, data in enumerate(testingloader, 0):
#     output = net(data['image'][i])
#     print(output)
#     if (i == 4):
#         break
