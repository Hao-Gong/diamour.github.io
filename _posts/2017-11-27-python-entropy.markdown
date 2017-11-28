---
layout: post
title:  "数学基础学习笔记（二）：关于几种熵及其python实现"
categories: 数学基础
tags: 数学基础 学习笔记
author: Hao
description: 关于几种熵的直白解释及其python实现
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

### Information entropy 信息熵
我们很早就知道了熵(entropy)这个概念，高中化学里面的。熵的本质是混乱度的意思，一个系统越是有序，信息熵就越低；反之，一个系统越是混乱，信息熵就越高。
这个概念非常有意思，在咨询公司实习的时候，我尝试量化一家公司员工的涉及领域的集中度，将各个领域的业务量，求信息熵，熵越高，说明该业务员的精力越分散。其实用这个来做透视表筛选效果还不错，只不过很难向客户解释这个概念，因为毕竟非常抽象，所以还是采用了传统的筛选方式。
下面就是信息熵的公式：

![math_entropy2](/assets/images/math/entropy2.png)

```python
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

Vector=np.array([10,-1,6,1,-2],dtype=float)
# Vector=np.array([1,1,1,1,1],dtype=float)
Vector_pt=autograd.Variable(torch.from_numpy(Vector))
print("Original Tensor:",Vector_pt)
Probs = F.softmax(Vector_pt)
print("Sotfmax posibility:",Probs)
# log(1/p(i))=-log(p(i)),pytorch有直接算出Softmax再log的
log_entropy=-F.log_softmax(Vector_pt)
# 每个元素相乘
Information_entropy=Probs*log_entropy
print("Information Entropy:",sum(Information_entropy))

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

	Information Entropy: Variable containing:
	1.00000e-02 *
	  9.1571
	[torch.DoubleTensor of size 1]

可以看出这个熵很小，因为其中第1个元素的概率相对较大，确定性高。

假设Vector=[1,1,1,1,1]，Information Entropy为：1.6094，其中计算的log是e为底的，这个数就是ln(1/0.2)。每个概率相同的时候，是分布最分散的。

### Cross entropy 交叉熵
交叉熵的概念在深度学习分类的时候特别有用，Cross entropy loss作为神经网络最后端的损失函数，计算的是神经网络输出的概率分布与标签单位向量的交叉熵，就是要优化缩小这个损失函数，其公式如下图，真实分布为p，假设分布为q，实际当中，p是神经网络输出后经过softmax的结果，q是标签向量：

![math_entropy3](/assets/images/math/entropy3.png)

直接上代码：

```python
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn

q=autograd.Variable(torch.from_numpy(np.array([1,5,10,2,1],dtype=float)))
#设置一个真实概率分布,其实里面数值随意的，经过softmax都会归一化
print("q:",F.softmax(q))
#我们期望假设的概率分布是，第二个元素是1
p=autograd.Variable(torch.from_numpy(np.array([0,1,0,0,0],dtype=float)))
print("p:",p)
#计算log(1/q(i))
log_entropy=-F.log_softmax(q)

#计算p(i)*log(1/q(i))
cross_entropy=p*log_entropy
print("Cross Entropy:",sum(cross_entropy))

```

输出：

	q: Variable containing:
	 0.0001
	 0.0067
	 0.9927
	 0.0003
	 0.0001
	[torch.DoubleTensor of size 5]

	p: Variable containing:
	 0
	 1
	 0
	 0
	 0
	[torch.DoubleTensor of size 5]

	Cross Entropy: Variable containing:
	 5.0073
	[torch.DoubleTensor of size 1]

假设提高第二个元素的值，这样概率分布逼近假设分布：

	q=autograd.Variable(torch.from_numpy(np.array([1,100,1,1,1],dtype=float)))

输出，这样交叉熵就很小了：
	
	q: Variable containing:
	 0.0001
	 0.9995
	 0.0001
	 0.0001
	 0.0001
	[torch.DoubleTensor of size 5]

	Cross Entropy: Variable containing:
	1.00000e-04 *
	  4.9352
	[torch.DoubleTensor of size 1]

### Relative entropy 相对熵
相对熵(relative entropy)又称为KL散度(Kullback-Leibler divergence)，是两个随机分布间距离的度量，就是指两个概率分布向量之间的距离度量，如图是公式，真实分布为p，假设分布为q：

![math_entropy4](/assets/images/math/entropy4.png)

相对熵 交叉熵 信息熵的关系 交叉熵=P的信息熵+相对熵：

![math_entropy5](/assets/images/math/entropy5.png)

可以看出，交叉熵和相对熵在P的信息熵为常量的时候是等同的，在one-hot类型的P假设概率的时候，P的信息熵为0。

### All Reference:

[1.知乎](https://www.zhihu.com/question/41252833) 

[2.wiki](https://en.wikipedia.org/wiki/Cross_entropy) 

[3.A Maximum Entropy Approach Natural Language Processing](https://www.isi.edu/natural-language/people/ravichan/papers/bergeretal96.pdf)

[4.纯净的天空](https://vimsky.com/article/714.html)

##### 版权归@Hao所有，转载标记来源。

