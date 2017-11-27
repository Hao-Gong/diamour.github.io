---
layout: post
title:  "数学基础学习笔记（一）：关于几种马尔可夫链概念的直白解释"
categories: 数学基础
tags: 数学基础 学习笔记
author: Hao
description: 关于几种马尔可夫链概念的直白解释，并且附上例题和python代码
---

### 简单的马尔可夫链
马尔可夫性质（markov property）：在一个随机过程中，t+1时刻的状态只和t时间状态有关，而与t实践之前的状态无关，则称该过程具有马尔可夫性。上面的文字是我觉得解释的最简单的，再量化一下，就是t时间状态是一个向量Vt，状态t到t+1就是一个矩阵M(里面的数值都是概率)，怎么从t变成t+1状态呢？就是向量Vt左乘矩阵M，注意是左乘。上例题：

	例子1： 小明去一个城市旅游，目的就是逛逛吃吃玩玩。小明一天早上出门使用交通工具的概率：
	公交车bus,自行车Bicycle,步行walk,其他other
	19%, 14%, 56%, 11%

	小明坐完一种交通工具，下一次乘坐的交通工具的概率：	

	      bus    bicycle  walk  other
	Bus      90      4      2     4
	bicycle   7     86      1     6
	walk      8      7     80     5
	other    10      2      3     85 
 
	假设小明各出行方式的转移概率不变，
	问题：
	（1） 预测小明第三次乘坐的交通工具概率？
	（2） 经历足够长的时间，求小明出行乘坐的交通工具概率是多少？

这个矩阵的意思就是，这次小明坐bus，下面可能有 90%换乘bus 4%换乘bicycle 2%换乘walk 4%换乘other的概率。

第一问：预测小明第三次乘坐的交通工具概率？

```python
import numpy as np

M=np.array([[90, 4, 2, 4],\
           [7, 86, 1, 6],\
           [8, 7, 80, 5],\
           [10, 2, 3, 85]], dtype=float)/100.

print("Transform Matrix:")
print(M)

V1=np.array([19,14,56,11],dtype=float)/100.
print("第一次:")
print(V1)

V2=np.dot(V1,M)
print("第二次:")
print(V2)

V3=np.dot(V2,M)
print("第二次:")
print(V3)
```
输出：

	Transform Matrix:
	[[ 0.9   0.04  0.02  0.04]
	 [ 0.07  0.86  0.01  0.06]
	 [ 0.08  0.07  0.8   0.05]
	 [ 0.1   0.02  0.03  0.85]]
	第一次:
	[ 0.19  0.14  0.56  0.11]
	第二次:
	[ 0.2366  0.1694  0.4565  0.1375]
	第二次:
	[ 0.275068  0.189853  0.375751  0.159328]

第二问：经历足够长的时间，求小明出行乘坐的交通工具概率是多少？？这个就是求一个稳态的马尔可夫过程，其本质是求转移矩阵的左特征向量，其特征根为1(因为概率总和为1嘛)。

```python
V_init=np.array([19,14,56,11],dtype=float)/100.
print("原初向量:")
print(V_init)
#循环10000次
for i in range(10000):
    V_init=np.dot(V_init,M)

print("原初始化的向量最后结果:")
print(V_init)

V_init2=np.array([100,0,0,0],dtype=float)/100.
print("原初向量:")
print(V_init2)
#循环10000次
for i in range(10000):
    V_init2=np.dot(V_init2,M)

print("一开始全部坐bus最后结果:")
print(V_init2)
```

输出是：

	原初向量:
	[ 0.19  0.14  0.56  0.11]
	原初始化的向量最后结果:
	[ 0.45911291  0.21117362  0.09211043  0.23760304]
	原初向量:
	[ 1.  0.  0.  0.]
	一开始全部坐bus最后结果:
	[ 0.45911291  0.21117362  0.09211043  0.23760304]

可以看出，无论怎么初始化向量，最后的向量都收敛于同一个特征向量。

### Hidden Markov models (HMMs)隐式马尔可夫链

HMM问题可由下面五个元素描述：

观测序列（observations）：实际观测到的现象序列
隐含状态（states）：所有的可能的隐含状态
初始概率（start_probability）：每个隐含状态的初始概率
转移概率（transition_probability）：从一个隐含状态转移到另一个隐含状态的概率
发射概率（emission_probability）：某种隐含状态产生某种观测现象的概率

我们先把上面的题目来改一下：

	例子2： 小明去一个城市旅游，目的就是逛逛吃吃玩玩。城市里可以使用这些交通工具：
	公交车bus,共享自行车Bicycle,步行walk,其他other
	小明每次坐完一种交通工具，下一次使用的交通工具概率如下：

	      bus    bicycle  walk  other
	Bus      90      4      2     4
	bicycle   7     86      1     6
	walk      8      7     80     5
	other    10      2      3     85 

	上面的矩阵就是，转移概率（transition_probability）：从一个隐含状态转移到另一个隐含状态的概率
 
	在城市中，不同的交通工具可以到达不同的地方，例如 餐馆 博物馆 公园：
	小明使用每一种交通工具的去的地方的概率是：

	        餐馆    博物馆   公园
	Bus      10       60     40
	bicycle  60       20     20
	walk     30       10     60
	other    30       30     40
	
	上面的矩阵就是，发射概率（emission_probability）：某种隐含状态产生某种观测现象的概率

	可以看出来小明walk的话，很有可能去公园，很符合常理，哈哈。
	今天小明准备去玩3个地方，刚出门坐了bus，坐bus的概率为1,这个就是初始概率（start_probability）：
	每个隐含状态的初始概率
	请问，小明最后一个地方去的地方的概率为？

在题目中，我们假设我们只知道小明去了餐馆博物馆公园中的一个地方，但是不知道小明每次坐了什么交通工具过来。隐式马尔可夫链，其中的隐就是其中无法被观测到的中间状态。

```python
import numpy as np

#交通工具转移矩阵
Hidden=np.array([[90, 4, 2, 4],\
           [7, 86, 1, 6],\
           [8, 7, 80, 5],\
           [10, 2, 3, 85]], dtype=float)/100.

#交通工具-地点转移矩阵
Seen=np.array([[10, 60, 40],\
           [60, 20, 20],\
           [30, 10, 60],\
           [30, 30, 40]], dtype=float)/100.

print("交通工具转移矩阵:")
print(Hidden)
print("交通工具-地点转移矩阵:")
print(Seen)
print("餐馆 博物馆 公园")
#bus出门
Init_State=np.array([100,0,0,0],dtype=float)/100.
#第一次去的地方
print("1 ",np.dot(Init_State,Seen))

#第二次换乘
Hidden_state=np.dot(Init_State,Hidden)
#第二次去的地方
print("2 ",np.dot(Hidden_state,Seen))

#第三次换乘
Hidden_state=np.dot(Hidden_state,Hidden)
#第三次去的地方
print("3 ",np.dot(Hidden_state,Seen))

```

输出：

	交通工具转移矩阵:
	[[ 0.9   0.04  0.02  0.04]
	 [ 0.07  0.86  0.01  0.06]
	 [ 0.08  0.07  0.8   0.05]
	 [ 0.1   0.02  0.03  0.85]]
	交通工具-地点转移矩阵:
	[[ 0.1  0.6  0.4]
	 [ 0.6  0.2  0.2]
	 [ 0.3  0.1  0.6]
	 [ 0.3  0.3  0.4]]
	次数 餐馆 博物馆 公园
	1  [ 0.1  0.6  0.4]
	2  [ 0.132  0.562  0.396]
	3  [ 0.1581   0.53114  0.3926 ]


我们只能观测到他三次分别最有可能去什么地方，但是用的什么交通工具不知道。一般在时间序列或者语义分析的时候，隐式马尔可夫可以从观测的状态中，猜测隐含层的状态。比如这个例子中，发现小明在公园，请问，他最有可能乘了什么交通工具，我们马上就能想到是walk。

### Maximum entropy Markov models (MEMMs) 最大交叉熵的隐式马尔可夫链
#### [关于最大熵可以在我这篇文章中看到](/数学基础/2017/11/26/python-entropy.html)


### Conditional Random Fields(CRFs) 条件随机场

### All Reference:

[张金龙的博客](http://blog.sciencenet.cn/blog-255662-513722.html) 

##### 版权归@Hao所有，转载标记来源。

