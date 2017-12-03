---
layout: post
title:  "pytorch图像(二)：使用Resnet18的训练tiny-imagenet-200分类器"
categories: pytorch
tags: pytorch resnet imagenet 学习笔记
author: Hao
description: 超详细的使用Resnet18的训练tiny-imagenet-200分类器
---
### 使用轻量级的Resnet18来训练tiny-imagenet-200数据集
#### 什么是tiny-imagenet-200？

[tiny-imagenet-200下载链接](https://tiny-imagenet.herokuapp.com/),tiny-imagenet-200是基于ImageNet的downsampling的数据集，图片尺寸3×64×64 rgb图像或者1×64×64灰白图像（这个是个坑，竟然有灰度图，导致我读入的时候数据尺寸不匹配，调试了一会才发现，一共有10万张图片，200个类别，每张图片还配有单词描述和bounding box的坐标。在这个数据集上，我们可以完成很多任务。

### 读取数据集
```python
#放图片的根目录
ROOT_PATH='/home/diamous'
#200个标签目录
TINY_PATH_ROOT=ROOT_PATH+'/tiny-imagenet-200/'
#train数据集目录
TINY_PATH_TRAIN=ROOT_PATH+'/tiny-imagenet-200/train/'
#验证数据集目录
TINY_PATH_VAL=ROOT_PATH+'/tiny-imagenet-200/val/'
```

首先我们将数据的标签读入，在train的文件夹下面，每个标签是一个文件夹，里面一共有500张该标签的的数据，这里我们只是将图片的path放入list中，如果全部图片一下子读取到内存的话，占用太高：

```python
#读取train_data,输出格式[path,label,(x_s,y_s,x_e,y_e)]的list
def read_train_data():
    image_list = []
    with open(TINY_PATH_ROOT + 'wnids.txt', 'r') as f:
        data = f.read()
    file_name = data.split()

    for i in range(200):
        with open(TINY_PATH_TRAIN + file_name[i] + '/' + file_name[i] + '_boxes.txt', 'r') as f:
            data = f.read()
        image_name_data = data.split()
        for j in range(500):
            image_path = TINY_PATH_TRAIN + file_name[i] + '/images/' + image_name_data[j * 5]
            x_s = int(image_name_data[j * 5 + 1])
            y_s = int(image_name_data[j * 5 + 2])
            x_e = int(image_name_data[j * 5 + 3])
            y_e = int(image_name_data[j * 5 + 4])
            tuple = np.array([x_s, y_s, x_e, y_e])
            image_list.append([image_path, i, tuple])
    random.shuffle(image_list)
    return image_list
```

同理，验证数据集可以也这样读入：

```python
#读取validate_data,输出格式[path,label,(x_s,y_s,x_e,y_e)]
def read_validate_data():
    image_list = []
    with open(TINY_PATH_ROOT + 'wnids.txt', 'r') as f:
        data = f.read()
    file_name = data.split()

    with open(TINY_PATH_VAL + 'val_annotations.txt', 'r') as f:
        data = f.read()
    file_data = data.split()
    for i in range(10000):
        image_path = TINY_PATH_VAL + 'images/' + file_data[i * 6]
        label=file_name.index(file_data[i * 6 + 1])
        x_s = int(file_data[i * 6 + 2])
        y_s = int(file_data[i * 6 + 3])
        x_e = int(file_data[i * 6 + 4])
        y_e = int(file_data[i * 6 + 5])
        tuple = np.array([x_s, y_s, x_e, y_e])
        image_list.append([image_path, label, tuple])
    return image_list
```

接下来，我们就用生成器函数来读取数据集，我有一篇文章讲[yield表达式的用法](/python/2017/12/02/python-learning1.html)，我们只将需要的图片从硬盘中读取，每次数据集随机打乱：

```python
def train_batch_load(batch_size=50):
    random.shuffle(image_train)
    image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for cursor in range(0, len(image_train), batch_size):
        image_batch = torch.unsqueeze(check_grey(image_trans(Image.open(image_train[cursor][0]))), 0)
        label_batch = torch.torch.LongTensor([image_train[cursor][1]])
        box_batch = torch.unsqueeze(torch.from_numpy(image_train[cursor][2]), 0)
        batch=[]
        for i in range(1,batch_size):
            image = torch.unsqueeze(check_grey(image_trans(Image.open(image_train[cursor+i][0]))), 0)
            label = torch.torch.LongTensor([image_train[cursor+i][1]])
            box = torch.unsqueeze(torch.from_numpy(image_train[cursor+i][2]), 0)
            image_batch = torch.cat((image_batch, image), 0)
            label_batch = torch.cat((label_batch, label), 0)
            box_batch = torch.cat((box_batch, box), 0)
        batch.append(image_batch)
        batch.append(label_batch)
        batch.append(box_batch)
        yield batch


def val_batch_load(batch_size=1000):
    random.shuffle(image_val)
    image_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    for cursor in range(0, len(image_val), batch_size):
        image_batch = torch.unsqueeze(check_grey(image_trans(Image.open(image_val[cursor][0]))), 0)
        label_batch = torch.torch.LongTensor([image_val[cursor][1]])
        box_batch = torch.unsqueeze(torch.from_numpy(image_val[cursor][2]), 0)
        batch=[]
        for i in range(1,batch_size):
            image = torch.unsqueeze(check_grey(image_trans(Image.open(image_val[cursor+i][0]))), 0)
            label = torch.torch.LongTensor([image_val[cursor+i][1]])
            box = torch.unsqueeze(torch.from_numpy(image_val[cursor+i][2]), 0)
            image_batch = torch.cat((image_batch, image), 0)
            label_batch = torch.cat((label_batch, label), 0)
            box_batch = torch.cat((box_batch, box), 0)
        batch.append(image_batch)
        batch.append(label_batch)
        batch.append(box_batch)
        yield batch
```

这里注意，我们会读到灰度图，我们就把灰度图的值直接复制成3通道：

```python
def check_grey(im_t):
    if(im_t.size(0)==3):
        return im_t
    else:
        return torch.cat((im_t,im_t,im_t),0)
```

### 准备Resnet18：
因为我们这次训练的是3×64×64的图像，上文的Resnet是基于3×224×224的原版ImageNet的图像，所以我们调整一下Resnet最后一个pooling层：

	#self.avgpool = nn.AvgPool2d(7, stride=1)变成如下
	self.avgpool = nn.AvgPool2d(2, stride=1)

训练的时候，我们不采用固定的learning rate，而是让它随着epoch从大变到小，因为一开始太小的话，可能导致收敛到局部最优解，而不能大范围找到更好的解，但是如果始终很大的话，就会导致所搜步长太大，不能精确收敛：

```python
def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1 ** (int(epoch/2)+1))
    print("learning rate",lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

### 使用cuda来训练：
使用cuda只需要在定义net和Variable的时候在后面加上，在使用validate数据集的时候，需要关闭packpropogation：

```python
BATCH_SIZE=500
VAL_BATCH_SIZE=500
image_train=read_train_data()
image_val=read_validate_data()
LR=0.01

resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
#使用cuda
resnet18.cuda()

optimizer = torch.optim.Adam(resnet18.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()

for epoch in range(10):
    step=0
    adjust_learning_rate(optimizer, epoch)
    for batch in train_batch_load(batch_size=BATCH_SIZE):
	#使用cuda
        b_x = Variable(batch[0].cuda())
        b_y = Variable(batch[1].cuda())
        output = resnet18(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(step%50==0):
            for batch_val in val_batch_load(batch_size=VAL_BATCH_SIZE):
		#使用cuda，关闭BP
                x_test=Variable(batch_val[0].cuda(), volatile=True)
                y_test = Variable(batch_val[1].cuda())
                test_output = resnet18(x_test)
                pred = test_output.data.max(1, keepdim=True)[1]
                correct = pred.eq(y_test.data.view_as(pred)).cpu().sum()/VAL_BATCH_SIZE
                print('Epoch: ', epoch, 'step:', step, '| test accuracy: %.2f' % correct)
                break
        step = step + 1
```

最后我们可以保存网络数据：

```python
torch.save(resnet18, 'resnet18.pkl')
torch.save(resnet18.state_dict(), 'resnet18_params.pkl')
```

然后用我的GTX1050TI运行了20个epoch两个小时，这个是输出：

	learning rate 0.01
	Epoch:  0 step: 0 | test accuracy: 0.01
	Epoch:  0 step: 200 | test accuracy: 0.01
	Epoch:  0 step: 400 | test accuracy: 0.03
	learning rate 0.01
	Epoch:  1 step: 0 | test accuracy: 0.04
	Epoch:  1 step: 200 | test accuracy: 0.11
	Epoch:  1 step: 400 | test accuracy: 0.14
	learning rate 0.001
	Epoch:  2 step: 0 | test accuracy: 0.17
	Epoch:  2 step: 200 | test accuracy: 0.22
	Epoch:  2 step: 400 | test accuracy: 0.23
	learning rate 0.001
	Epoch:  3 step: 0 | test accuracy: 0.22
	Epoch:  3 step: 200 | test accuracy: 0.24
	Epoch:  3 step: 400 | test accuracy: 0.26
	learning rate 0.00010000000000000002
	Epoch:  4 step: 0 | test accuracy: 0.26
	Epoch:  4 step: 200 | test accuracy: 0.27
	Epoch:  4 step: 400 | test accuracy: 0.28
	learning rate 0.00010000000000000002
	Epoch:  5 step: 0 | test accuracy: 0.27
	Epoch:  5 step: 200 | test accuracy: 0.30
	Epoch:  5 step: 400 | test accuracy: 0.27
	learning rate 1.0000000000000003e-05
	Epoch:  6 step: 0 | test accuracy: 0.27
	Epoch:  6 step: 200 | test accuracy: 0.26
	Epoch:  6 step: 400 | test accuracy: 0.29
	learning rate 1.0000000000000003e-05
	Epoch:  7 step: 0 | test accuracy: 0.28
	Epoch:  7 step: 200 | test accuracy: 0.27
	Epoch:  7 step: 400 | test accuracy: 0.29
	learning rate 1.0000000000000002e-06
	Epoch:  8 step: 0 | test accuracy: 0.30
	Epoch:  8 step: 200 | test accuracy: 0.28
	Epoch:  8 step: 400 | test accuracy: 0.28
	learning rate 1.0000000000000002e-06
	Epoch:  9 step: 0 | test accuracy: 0.27
	Epoch:  9 step: 200 | test accuracy: 0.26
	Epoch:  9 step: 400 | test accuracy: 0.29
	learning rate 1.0000000000000002e-07
	Epoch:  10 step: 0 | test accuracy: 0.28
	Epoch:  10 step: 200 | test accuracy: 0.29
	Epoch:  10 step: 400 | test accuracy: 0.32
	learning rate 1.0000000000000002e-07
	Epoch:  11 step: 0 | test accuracy: 0.29
	Epoch:  11 step: 200 | test accuracy: 0.30
	Epoch:  11 step: 400 | test accuracy: 0.26
	learning rate 1.0000000000000004e-08
	Epoch:  12 step: 0 | test accuracy: 0.28
	Epoch:  12 step: 200 | test accuracy: 0.28
	Epoch:  12 step: 400 | test accuracy: 0.29
	learning rate 1.0000000000000004e-08
	Epoch:  13 step: 0 | test accuracy: 0.29
	Epoch:  13 step: 200 | test accuracy: 0.27
	Epoch:  13 step: 400 | test accuracy: 0.27
	learning rate 1.0000000000000005e-09
	Epoch:  14 step: 0 | test accuracy: 0.26
	Epoch:  14 step: 200 | test accuracy: 0.30
	Epoch:  14 step: 400 | test accuracy: 0.32
	learning rate 1.0000000000000005e-09
	Epoch:  15 step: 0 | test accuracy: 0.28
	Epoch:  15 step: 200 | test accuracy: 0.28
	Epoch:  15 step: 400 | test accuracy: 0.31
	learning rate 1.0000000000000006e-10
	Epoch:  16 step: 0 | test accuracy: 0.28
	Epoch:  16 step: 200 | test accuracy: 0.28
	Epoch:  16 step: 400 | test accuracy: 0.26
	learning rate 1.0000000000000006e-10
	Epoch:  17 step: 0 | test accuracy: 0.28
	Epoch:  17 step: 200 | test accuracy: 0.27
	Epoch:  17 step: 400 | test accuracy: 0.28
	learning rate 1.0000000000000004e-11
	Epoch:  18 step: 0 | test accuracy: 0.27
	Epoch:  18 step: 200 | test accuracy: 0.28
	Epoch:  18 step: 400 | test accuracy: 0.29
	learning rate 1.0000000000000004e-11
	Epoch:  19 step: 0 | test accuracy: 0.29
	Epoch:  19 step: 200 | test accuracy: 0.30
	Epoch:  19 step: 400 | test accuracy: 0.28

前面几次收敛的还是很快的，但是后面准确度一直停留在30%左右，但是放在训练集上会有60%的准确率，看来还需要调整一下。

[我的github](https://github.com/diamour/tf_learning/blob/master/pt_learning/models/resnet_tiny_imagenet_200_cuda2.py),这次只是做了分类，后面要使用bounding box regression来做Fast RCNN或者Faster RCNN的定位问题了。

### All Reference:
[1.Resnet 原版论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

[2.pytorch model源码](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)

##### 版权归@Hao所有，转载标记来源。



