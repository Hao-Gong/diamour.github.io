---
layout: post
title:  "安装pytorch cuda-8.0"
categories: pytorch
tags: pytorch install 学习笔记
author: Hao
description: 安装pytorch cuda8.0
---
#### 上周archlinux崩溃了，不想再麻烦用u盘挂载和修复了，重新装了ubuntu 16.10。这里发现一个问题，ubuntu 16.04.03 lts/ubuntu 17.10对于新一代intel KabyLake架构支持不行，U盘启动盘并不能进入界面。同时，低于ubuntu 16.04.01 lts的版本，新的intel wifi不能识别，其内核是4.4版本。所以最后只能选则了16.10，内核版本4.8。跑是能跑了，ros估计不能装kinetic的版本了。本文讲一下怎么安装pytorch和cuda8.0,至于支持卷积的cudnn暂时先用不到，下次再装：[cudnn](https://developer.nvidia.com/cudnn)，这个需要在Nvidia官网注册账户。

#### 我的电脑配置是：
	i7 7700HQ 
	GTX 1050ti 4g

其实4g的显卡内存一般的CNN还真可以勉强跑一跑，以后工作了配一台好电脑。

[pytorch官网传送门](http://pytorch.org/)，选择python3.5版本(我的是3.5)和cuda8.0，直接把命令贴出来

```
pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip3 install torchvision
```
目前Nvidia的官网默认最新的Cuda9.0,但是我们想要用支持比较好的Cuda8.0,上链接：[cuda8.0](https://developer.nvidia.com/cuda-toolkit-archive)。这里我下载了这个[cuda8.0 ga2 2017](https://developer.nvidia.com/cuda-80-ga2-download-archive)，下载后运行：
	sudo sh cuda_8.0.61_375.26_linux.run

后来我发现ubuntu 16.10并不能安装上官网提供的Cuda8.0, 还有一个办法就是载程序update里面就能够更新显卡驱动。
安装好以后，需要重启，然后：

	nvidia-smi

```
Sun Nov 26 11:23:02 2017
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66                 Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 105...  Off  | 0000:01:00.0     Off |                  N/A |
| N/A   47C    P0    N/A /  N/A |    271MiB /  4038MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0      1138    G   /usr/lib/xorg/Xorg                             152MiB |
|    0      1867    G   /usr/bin/compiz                                111MiB |
|    0      2287    G   fcitx-qimpanel                                   6MiB |
+-----------------------------------------------------------------------------+
```

说明安装好了，然后我们来复制粘帖一下这个使用CUDA的程序，注释掉.cuda()部分，你可以发现CPU计算速度慢很多：
```
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


torch.manual_seed(1)
torch.cuda.manual_seed(1)


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=64, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 11):
    train(epoch)
test()
```

看到这个，说明你已经成功了：
```
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.390087
Train Epoch: 1 [640/60000 (1%)]	Loss: 1.998551
Train Epoch: 1 [1280/60000 (2%)]	Loss: 1.436028
Train Epoch: 1 [1920/60000 (3%)]	Loss: 1.122211
Train Epoch: 1 [2560/60000 (4%)]	Loss: 0.915172
Train Epoch: 1 [3200/60000 (5%)]	Loss: 1.100636
Train Epoch: 1 [3840/60000 (6%)]	Loss: 1.035625
Train Epoch: 1 [4480/60000 (7%)]	Loss: 1.069292
Train Epoch: 1 [5120/60000 (9%)]	Loss: 0.802876
Train Epoch: 1 [5760/60000 (10%)]	Loss: 0.790949
Train Epoch: 1 [6400/60000 (11%)]	Loss: 0.652440
Train Epoch: 1 [7040/60000 (12%)]	Loss: 0.727161
Train Epoch: 1 [7680/60000 (13%)]	Loss: 0.734073
Train Epoch: 1 [8320/60000 (14%)]	Loss: 0.586786
Train Epoch: 1 [8960/60000 (15%)]	Loss: 0.687886
Train Epoch: 1 [9600/60000 (16%)]	Loss: 0.708015
Train Epoch: 1 [10240/60000 (17%)]	Loss: 0.539205
Train Epoch: 1 [10880/60000 (18%)]	Loss: 0.323217
Train Epoch: 1 [11520/60000 (19%)]	Loss: 0.541503
Train Epoch: 1 [12160/60000 (20%)]	Loss: 0.553394
Train Epoch: 1 [12800/60000 (21%)]	Loss: 0.821560
Train Epoch: 1 [13440/60000 (22%)]	Loss: 0.533104
Train Epoch: 1 [14080/60000 (23%)]	Loss: 0.400535
Train Epoch: 1 [14720/60000 (25%)]	Loss: 0.759426
Train Epoch: 1 [15360/60000 (26%)]	Loss: 0.738509
Train Epoch: 1 [16000/60000 (27%)]	Loss: 0.493532
Train Epoch: 1 [16640/60000 (28%)]	Loss: 0.281890
Train Epoch: 1 [17280/60000 (29%)]	Loss: 0.397414
Train Epoch: 1 [17920/60000 (30%)]	Loss: 0.825475

```

##### 版权归@Hao所有，转载标记来源。

