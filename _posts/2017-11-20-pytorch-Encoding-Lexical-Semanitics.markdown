---
layout: post
title:  "基于pytorch的词汇语义编码"
categories: pytorch
tags: pytorch NLP 学习笔记
author: Hao
description: 本文是将按照pytorch官网案例进行分析
---
### 写这篇文章的动力
这学期闲来无事，对机械专业的课程也没啥太多的兴趣，加上KIT这边的计算机专业还不错，就报名参加了一门神经网络的实践课程。期末的作业是做一个NLP(Natural Language Processing)的RNN编程作业，所以先从pytorch的官网上学习。

[pytorch官网传送门](http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-encoding-lexical-semantics) 
#### 原文题目：
Word Embeddings: Encoding Lexical Semantics
### 本文是将按照pytorch官网案例进行分析，按照原文加上个人解释的格式来写
#### 原文：
Word embeddings are dense vectors of real numbers, one per word in your vocabulary. In NLP, it is almost always the case that your features are words! But how should you represent a word in a computer? You could store its ascii character representation, but that only tells you what the word is, it doesn’t say much about what it means (you might be able to derive its part of speech from its affixes, or properties from its capitalization, but not much). Even more, in what sense could you combine these representations? We often want dense outputs from our neural networks, where the inputs are |V| dimensional, where V is our vocabulary, but often the outputs are only a few dimensional (if we are only predicting a handful of labels, for instance). How do we get from a massive dimensional space to a smaller dimensional space?
#### 我的理解：
单词编码是，单词一对一的映射到实数向量中去。在自然语言处理的过程中，对象特征就是单词。你可以将单词保存成为ASCII码，但是这种方式你并不能很清楚的知道单词的意思。通常情况下，向神经网络输入我们词汇量大小维度的矩阵，在有标签进行预测的情况下输出低维度的矩阵，而我们希望神经网络能够有高密度(高信息量)的输出。那我们如何将高维度的空间映射到低维度的空间呢？

首先Word embeddings是什么意思呢？字面上来看是“单词嵌入”的意思，那么单词嵌入到哪里呢，这里我参考了
[知乎李韶华](https://www.zhihu.com/question/32275069/answer/109446135)
的理解，词汇在分为近意词(或者在句子中功能相近的单词)和毫不相关的词汇，如何表示远近，可以使用高维的向量来表示，其夹角(如余弦相似度)很大程度上展示了其意思的相近。

#### 原文：
How about instead of ascii representations, we use a one-hot encoding? That is, we represent the word w by 

![nlp1](/assets/images/NLP/nlp1.png)

where the 1 is in a location unique to w. Any other word will have a 1 in some other location, and a 0 everywhere else.There is an enormous drawback to this representation, besides just how huge it is. It basically treats all words as independent entities with no relation to each other. What we really want is some notion of similarity between words. Why? Let’s see an example.

Suppose we are building a language model. Suppose we have seen the sentences

    The mathematician ran to the store.
    The physicist ran to the store.
    The mathematician solved the open problem.

in our training data. Now suppose we get a new sentence never before seen in our training data:

    The physicist solved the open problem.

Our language model might do OK on this sentence, but wouldn’t it be much better if we could use the following two facts:1.We have seen mathematician and physicist in the same role in a sentence. Somehow they have a semantic relation.2.We have seen mathematician in the same role in this new unseen sentence as we are now seeing physicist.

And then infer that physicist is actually a good fit in the new unseen sentence? This is what we mean by a notion of similarity: we mean semantic similarity, not simply having similar orthographic representations. It is a technique to combat the sparsity of linguistic data, by connecting the dots between what we have seen and what we haven’t. This example of course relies on a fundamental linguistic assumption: that words appearing in similar contexts are related to each other semantically. This is called the distributional hypothesis.
#### 我的理解：
如果不用ASCII码来表示，而用单位向量来表示，例如我们可以用如下向量表示单词W：

![nlp1](/assets/images/NLP/nlp1.png)

这个向量有V个元素，V同时也是词汇库的大小，毕竟不同单词都需要用不同的单位向量。这就带来一个很大问题，每个单词的向量与其他向量都是线性无关的(内积为0)。但是我们真正想要的是能够表示出词汇间词义的关系，下面是一个例子。假设我们在搭建一个语言模型，假设我们看到了这样一些句子：

    The mathematician ran to the store.
    The physicist ran to the store.
    The mathematician solved the open problem.

在我们的训练数据中，现在假设我们有一个从来没有见过的句子：

    The physicist solved the open problem.

我们的语言模型这种情况下还可以，但是考虑到下面两个事实就并不好了：1.我们看到数学家和物理学家在句子中的成分是一样的，从某种意义上来说他们有语义上的关系。2.我们看到数学家在新的句子中与物理学家具有相同的角色。这样可以推测物理学家是不是在未知的句子中具有很高的匹配度？这就是所谓的语义上的相近，而不是用相近的正则表示(orthographic representations,就是单位向量表示)。解决语言数据的稀疏性始终是一个技术问题，我们通过连接观测到的或者没观测到的数据点来解决。这个例子依靠的是基本的语义学上的假设：这些单词在相近的环境下出现具有相同的语义关系。这就是所说的分布理论(distributional hypothesis)。

好了，现在来点干货：什么是分布理论(distributional hypothesis)呢？这个概念首先由Zellig Harris提出的。这里先发一下参考的两篇论文链接：

1.[MagnusSahlgren. The distributional hypothesis](http://soda.swedish-ict.se/3941/1/sahlgren.distr-hypo.pdf)

2.[Chris Potts, Ling 236/Psych 236c. Distributional approaches to word meanings](https://web.stanford.edu/class/linguist236/materials/ling236-handout-05-09-vsm.pdf)

简单的操作流程是，你手头有好几篇文档，你统计出你关心的单词w在每个文档(documents)中分别出现的次数，生成矩阵D，如图(图片来自[论文1](http://soda.swedish-ict.se/3941/1/sahlgren.distr-hypo.pdf))：

![nlp2](/assets/images/NLP/nlp2.png)

然后你可以将矩阵D乘以D的转置得到新的矩阵(co-occurrence matrix),如图(图片来自[论文1](http://soda.swedish-ict.se/3941/1/sahlgren.distr-hypo.pdf)):

![nlp3](/assets/images/NLP/nlp3.png)

然后你就可以用PAC，SVD或者现在的cnn encoder来降维，一般情况下，降成2维或者3维，贴上标签，你就能看到高相关性的数据会有聚类现象，例如(图片来自[论文2](https://web.stanford.edu/class/linguist236/materials/ling236-handout-05-09-vsm.pdf))：

![nlp4](/assets/images/NLP/nlp4.png)


我觉得我讲的还是很清楚的。

### 稠密单词编码(Getting Dense Word Embeddings),相对于上文的稀疏编码
#### 原文
How can we solve this problem? That is, how could we actually encode semantic similarity in words? Maybe we think up some semantic attributes. For example, we see that both mathematicians and physicists can run, so maybe we give these words a high score for the “is able to run” semantic attribute. Think of some other attributes, and imagine what you might score some common words on those attributes.

If each attribute is a dimension, then we might give each word a vector, like this:

![nlp5](/assets/images/NLP/nlp5.png)

Then we can get a measure of similarity between these words by doing:

![nlp6](/assets/images/NLP/nlp6.png)

Although it is more common to normalize by the lengths:

![nlp7](/assets/images/NLP/nlp7.png)

Where ϕ is the angle between the two vectors. That way, extremely similar words (words whose embeddings point in the same direction) will have similarity 1. Extremely dissimilar words should have similarity -1.

You can think of the sparse one-hot vectors from the beginning of this section as a special case of these new vectors we have defined, where each word basically has similarity 0, and we gave each word some unique semantic attribute. These new vectors are dense, which is to say their entries are (typically) non-zero.
### All Reference:

[1.pytorch.org](http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#word-embeddings-encoding-lexical-semantics) 

[2.知乎李韶华](https://www.zhihu.com/question/32275069/answer/109446135)

[3.MagnusSahlgren. The distributional hypothesis](http://soda.swedish-ict.se/3941/1/sahlgren.distr-hypo.pdf)

[4.Chris Potts, Ling 236/Psych 236c. Distributional approaches to word meanings](https://web.stanford.edu/class/linguist236/materials/ling236-handout-05-09-vsm.pdf)

