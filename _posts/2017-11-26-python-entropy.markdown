---
layout: post
title:  "数学基础学习笔记（二）：关于几种信息熵的直白解释"
categories: 数学基础
tags: 数学基础 学习笔记
author: Hao
description: 关于几种信息熵的直白解释
---
### 首先我先不讲几种熵，首先讲一下我们经常需要在神经网络的生成概率分布，如何生成呢？
在神经网络终端我们不是一般输出浮点数值吗？那我们如何转换成概率期望呢？这就需要使用Softmax函数，它能将一个含任意实数的K维的向量V的“压缩”到另一个K维实向量中，使得每一个元素的范围都在0-1之间，并且所有元素的和为1。这样就满足了概率的要求，公式如下：

![math_entropy1](/assets/images/math/entropy1.png)

假设我们输入的向量为[10,-1,6,1,-2]，我们用pytorch来实现一下：

```python
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

#init
Vector=np.array([10,-1,6,1,-2],dtype=float)

Vector_pt=autograd.Variable(torch.from_numpy(Vector))
print("Original Tensor:",Vector_pt)
Probs = F.softmax(Vector_pt)
print("Sotfmax posibility:",Probs)
print("Sum of Sotfmax:",sum(Probs))

```

输出：

	Original Tensor: Variable containing:
	 10
	 -1
	  6
	  1
	 -2
	[torch.DoubleTensor of size 5]

	Sotfmax posibility: Variable containing:
	 9.8187e-01
	 1.6399e-05
	 1.7984e-02
	 1.2117e-04
	 6.0328e-06
	[torch.DoubleTensor of size 5]

	Sum of Sotfmax: Variable containing:
	 1.0000
	[torch.DoubleTensor of size 1]

### Entropy 信息熵

### Cross entropy 交叉熵

### Relative entropy 相对熵

### Maximum entropy 最大熵



### All Reference:

[知乎](https://www.zhihu.com/question/41252833) 

##### 版权归@Hao所有，转载标记来源。

