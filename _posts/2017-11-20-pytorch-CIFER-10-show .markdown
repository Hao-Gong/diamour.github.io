---
layout: post
title:  "如何在python当中显示CIFAR-10图片"
categories: pytorch
tags: pytorch CIFAR-10 学习笔记
author: Hao
description: python当中显示CIFAR-10图片
---
#### 第一次写个人博客有点小紧张，先搞点水货上去发了再说。。。
### 简单介绍CIFAR10

CIFAR-10包含了10种标签，尺寸大小为32x32，深度为3的RGB图像格式，有6个数据集，其中一个测试集，用于交叉验证正确率。每个集合10000个样本，总共60000个样本。样本包含：

airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck

### 首先先从网上下载CIFAR10的batch包
下面是CIFAR10官网地址：

[CIFAR-10 Python](http://www.cs.toronto.edu/~kriz/cifar.html) 

注意，我们要下载的是CIFAR-10 python version这个压缩包。

```python
#使用pickle工具加载CIFAR-10的batch包
import pickle
#可以使用PIL来显示图片，也可以使用plt来显示图片
#本人推荐plt，显示的时候可以自动调节大小，CIFAR-10图片相当小
from PIL import Image
import matplotlib.pyplot as plt

#加载单个CIFAR包
def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        #加载出来的数据是一个10000大小的字典，有'batch_label'，'data'，'filenames'，'labels'
        datadict = pickle.load(f, encoding='bytes')
        #'batch_label'是0-9的数字
        batch_label = datadict[b'batch_label']
        #'data'是图片的数据，numpy.ndarray格式，需要从(10000,3,32×32)->(10000, 3, 32, 32)
        data = datadict[b'data'].reshape(10000, 3, 32, 32)
        #图片名字，用于保存图像使用
        filename=datadict[b'filenames']
        #图片的序号，第几张图片
        labels=datadict[b'labels']
        return data, labels, batch_label, filename

#需要将图片读出到RGB通道，混合形成RGB图片
def ConvetToImg(data):
    i0 = Image.fromarray(data[0])
    i1 = Image.fromarray(data[1])
    i2 = Image.fromarray(data[2])
    return Image.merge("RGB", (i0, i1, i2))

#第几张图片
SAMPLE_NUM=0
#你的cifar文件夹路径
CIFAR_PATH="/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/data_batch_1"

data = load_CIFAR_batch(CIFAR_PATH)

img=ConvetToImg(data[0][SAMPLE_NUM])

plt.imshow(img)

plt.text(2, -3, "image name:  %s  "%data[3][SAMPLE_NUM], fontdict={'size': 8, 'color':  'red'})
plt.text(2, -1, "image label:  %s  "%data[1][SAMPLE_NUM], fontdict={'size': 8, 'color':  'red'})
plt.show()

#也可以使用这个显示，不过图片相当小，是原始像素尺寸
# img.show()
```

直接运行一下，就行显示了，后面讲讲怎么把CIFAR-10读取然后转换成Pytorch tensor的格式，用于torch.utils.data.DataLoader()进行样本加载，虽然pytorch自带CIFAR-10样本集，非常方便，这样做是为了解tensor的格式，便于将来使用自己的图片。

[My GIT link](https://github.com/diamour/tf_learning/tree/master/pt_learning)



