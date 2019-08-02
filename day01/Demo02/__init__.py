#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/7/25
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim
import torchvision
import matplotlib.pyplot as plt
import utils



# step1. load dataset
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081, ))
                               ])),
    batch_size = batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                                                     transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                     torchvision.transforms.Normalize(
                                                                         (0.1307, ), (0.3081, ))
                                                                     ])),
     batch_size=batch_size, shuffle=True)
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), y.min())
utils.plot_image(x, y, 'image sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.f2 = nn.Linear(256, 64)
        self.f3 = nn.Linear(64, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return x
net = Net()
train_loss = []
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28*28)
        out = net(x)
        y_one_hot = utils.one_hot(y)
        loss = F.mse_loss(out, y_one_hot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

utils.plot_curve(train_loss)

total_correct = 0
for x, y, in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc', acc)
