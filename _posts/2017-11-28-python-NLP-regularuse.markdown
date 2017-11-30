---
layout: post
title:  "python中NLP的基础操作"
categories: NLP
tags: 学习笔记
author: Hao
description: 整理一下，python中NLP的一些基本操作
---

### String Operation:

```python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"


```


### All Reference:
[1.纯净的天空](https://vimsky.com/article/714.html)

[2.Maximum_Entropy_Classifiers.pdf](https://web.stanford.edu/class/cs124/lec/Maximum_Entropy_Classifiers.pdf)

[3.Logsitic and Maximum Classification](http://ataspinar.com/2016/05/07/regression-logistic-regression-and-maximum-entropy-part-2-code-examples/#maxentfeatures)

[4.Multiclassification](https://www.csie.ntu.edu.tw/~cjlin/courses/optml2014/maxent.pdf)

[5.A Brief Maxent Tutorial](https://www.cs.cmu.edu/afs/cs/user/aberger/www/html/tutorial/tutorial.html)
##### 版权归@Hao所有，转载标记来源。

