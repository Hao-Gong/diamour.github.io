---
layout: post
title:  "有趣的python系列(一)：yield 表达式"
categories: python
tags: python 学习笔记
author: Hao
description: 关于yield读取图像训练集
---
### 在学习图像处理的过程中，发现有一个读取图像文件的好方法，避免内存占用过高，就是用yield表达式

#### 那么什么是yield呢？
如果一个函数包含yield表达式，那么它是一个生成器函数(generator function)，调用它会返回一个特殊的迭代器(iteration)，称为生成器(generator)。

```python
def generator_function():
    yield 1

print(type(generator_function))
print(type(generator_function()))
```
输出，可以看出生成器函数本质是函数，返回的是生成器：

	<class 'function'>
	<class 'generator'>

#### 那么什么是迭代器(iteration)呢？
首先，迭代其是一个对象，迭代器抽象的是一个数据流，是只允许迭代一次的对象。对迭代器不断调用next()方法，则可以依次获取下一个元素；当迭代器中没有元素时，调用next()方法会抛出StopIteration异常。迭代器的__iter__()方法返回迭代器自身。我们经常用的range函数返回的是一个列表，而xrange返回的是一个迭代器。

#### 回到yield生成器函数
先上一个简单的斐波那切数列函数，[来源IBM developerWorks](https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/index.html)，在for循环执行时，每次循环都会执行fab函数内部的代码，执行到yield a,b时，fab函数就返回一个迭代值，下次迭代时，代码从yield a,b的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield:

```python
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield a,b
        # 先计算后面
        a, b = b, a + b
        n = n + 1

for n in fab(5):
    print(n)
```

输出如下，我们看出，函数内部的参数都会被保存下来。

	(0, 1)
	(1, 1)
	(1, 2)
	(2, 3)
	(3, 5)

### 利用yield读取训练集，我们有时候不需要将训练集batch全部读取，所以可以用yield：

```python
def train_batch_load(batchsize):
    for cursor in range(0, len(training_images), batchsize):
        batch = []
        batch.append(training_images[cursor:cursor + batchsize])
        batch.append(training_labels[cursor:cursor + batchsize])
        yield batch
```

### All Reference:
[1.始终](https://liam0205.me/2017/06/30/understanding-yield-in-python/)

[2.IBM developerWorks](https://www.ibm.com/developerworks/cn/opensource/os-cn-python-yield/index.html)

##### 版权归@Hao所有，转载标记来源。

