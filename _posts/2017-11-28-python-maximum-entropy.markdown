---
layout: post
title:  "数学基础学习笔记（三）：最大熵方法及其python实现"
categories: 数学基础
tags: 数学基础 学习笔记
author: Hao
description: 关于最大熵的直白解释及其python实现
---

### Maximum entropy 最大熵
最大熵与上文的概念不一样，它不是一种定义，而是一种优化方法。所以最大熵本质是信息熵，就是回归求解的时候将优化函数设置为argmax(H(p))。最大熵原理认为，学习概率模型时，在所有可能的概率模型中，熵最大的模型是最好的模型。
最大熵模型在NLP中使用相当广泛，用来干嘛呢，就是经过训练以后，给出上文X，可以猜出下文Y。在一个NLP(可以是语义分析，也可以是翻译)随机过程，它的输出是Y。而输入是上下文信息X。我们的任务就是构造一个统计模型，该模型的任务是：在给定上下文X的情况下，输出y的概率p(Y|X)。因此我们可以收集到大量的样本数据:(X1,Y1),(X2,Y2)...(Xn,Yn)。我们可以用样本的经验分布P~来表示所有样本的分布特性:

![math_entropy6](/assets/images/math/entropy6.png)

其中N为训练样本的大小, num(x,y)是样本中(x,y)重复出现次数。

### All Reference:
[1.纯净的天空](https://vimsky.com/article/714.html)

[2.Maximum_Entropy_Classifiers.pdf](https://web.stanford.edu/class/cs124/lec/Maximum_Entropy_Classifiers.pdf)
##### 版权归@Hao所有，转载标记来源。

