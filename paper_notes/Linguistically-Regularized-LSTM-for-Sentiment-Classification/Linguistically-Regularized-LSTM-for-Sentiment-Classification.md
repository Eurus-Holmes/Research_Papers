> [原文链接](https://arxiv.org/pdf/1611.03949.pdf)

----------
# Abstract

> In this paper, we propose simple models trained with sentence-level annotation, but also attempt to model the linguistic role of sentiment lexicons, negation words, and intensity words. 

提出了一种简单的句子级情感分类模型，把语言学规则（情感词典，否定词和程度副词）融入到现有的句子级LSTM情感分类模型中。

----------
# Related Work

## 1. Neural Networks for Sentiment Classification

 - 通过递归自编码器神经网络建立句子的语义表示，输入文本通常是树结构的，具体工作可参考：
    - [[Socher et al. 2011] Semi-Supervised Recursive Autoencoders for Predicting Sentiment Distributions](http://www.aclweb.org/anthology/D11-1014)
    - [[Socher et al. 2013] Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)
    - [[Dong et al. 2014] Adaptive Multi-Compositionality for Recursive Neural Models with Applications to Sentiment Analysis](http://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/viewFile/8148/8605)
    - [[Qian et al. 2015] Learning Tag Embeddings and Tag-specific Composition Functions in Recursive Neural Network](http://www.aclweb.org/anthology/P15-1132)
 - 通过CNN建立句子的语义表示，输入是文本序列，具体工作可参考：
    - [[Kim 2014] Convolutional Neural Networks for Sentence Classification](http://www.aclweb.org/anthology/D14-1181)
    - [ [Kalchbrenner, Grefenstette, and Blunsom 2014] A Convolutional Neural Network for Modelling Sentences](http://www.aclweb.org/anthology/P14-1062)
    - [[Tao Lei, Regina Barzilay, and Tommi Jaakkola. 2015] Molding cnns for text: non-linear, non-consecutive convolutions. ACL .](https://arxiv.org/pdf/1508.04112.pdf)
 - 通过LSTM模型建立句子的语义表示，可以用在文本序列的建模上，也可以是树结构的输入，具体工作可参考：
    - [[Sepp Hochreiter and Ju ̈rgen Schmidhuber. 1997] Long short-term memory. Neural Computation 9(8):1735–1780.](http://www.bioinf.jku.at/publications/older/2604.pdf) 
    - [[Zhu, Sobhani, and Guo 2015] Long Short-Term Memory Over Tree Structures](https://arxiv.org/pdf/1503.04881.pdf)
    - [[Tai, Socher, and Manning 2015] Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://www.aclweb.org/anthology/P15-1150)

----------


## 2. Applying Linguistic Knowledge for Sentiment Classification

> 语言学知识对情感分析任务至关重要，
主要包括:
情感词典(sentiment lexicons)
否定词（negation words (not, never, neither, etc.))
程度副词(intensity words (very, extremely, etc.))

 - 情感词典（sentiment lexicon）

> 情感词典（英文）应用比较广泛的有Hu and Liu 2004年提出的情感词典和MPQA词典，详细介绍可分别参考以下两篇文章：

[[Hu and Liu 2004]  Mining and Summarizing Customer Reviews](https://www.cs.uic.edu/~liub/publications/kdd04-revSummary.pdf)
<br>
[[Wilson, Wiebe, and Hoffmann 2005] Recognizing Contextual Polarity in Phrase-Level Sentiment Analysis](https://people.cs.pitt.edu/~wiebe/pubs/papers/emnlp05polarity.pdf)

 - 否定词（negation words）

> 否定词在情感分析中也是一个关键元素，它会改变文本表达的情感倾向。有很多相关研究：

[1. [Polanyi and Zaenen 2006] Contextual Valence Shifters](https://www.aaai.org/Papers/Symposia/Spring/2004/SS-04-07/SS04-07-020.pdf)
<br>
[2. [Taboada et al. 2011] Lexicon-Based Methods for Sentiment Analysis](https://www.aclweb.org/anthology/J/J11/J11-2001.pdf)
<br>
[3. [Zhu et al. 2014] An Empirical Study on the Effect of Negation Words on Sentiment](http://acl2014.org/acl2014/P14-1/pdf/P14-1029.pdf)
<br>
[4. [Kiritchenko and Mohammad 2016] Sentiment Composition of Words with Opposing Polarities](https://www.aclweb.org/anthology/N/N16/N16-1128.pdf)
<br>
> 文章1：对否定词的处理是将含有否定词的文本的情感倾向反转；<br>
文章2：由于不同的否定表达以不同的方式不同程度影响着句子的情感倾向，文中提出否定词按照某个常量值的程度改变着文本的情感倾向；<br>
文章3：将否定词作为特征结合到神经网络模型中；<br>
文章4：将否定词和其他语言学知识与SVM结合在一起，分析文本情感倾向。

 - 程度副词 （intensity words）            

> 程度副词影响文本的情感强度，在细粒度情感中非常关键，相关研究可参考：

[1. [Taboada et al. 2011] Lexicon-Based Methods for Sentiment Analysis](https://www.aclweb.org/anthology/J/J11/J11-2001.pdf)
<br>
[2. [Wei, Wu, and Lin 2011] A regression approach to affective rating of chinese words from anew](http://www.infomus.org/Events/proceedings/ACII2011/papers/6975/69750121.pdf)
<br>
[3. [Malandrakis et al. 2013] Distributional semantic models for affective text analysis](http://www.telecom.tuc.gr/~potam/preprints/journal/2013_TASLP_affect_text.pdf)
<br>
[4. [Wang, Zhang, and Lan 2016] Ecnu at semeval-2016 task 7: An enhanced supervised learning method for lexicon sentiment intensity ranking](http://www.aclweb.org/anthology/S16-1080)
<br>
文章1：直接让程度副词通过一个固定值来改变情感强度；<br>
文章2：利用线性回归模型来预测词语的情感强度值；<br>
文章3：通过核函数结合语义信息预测情感强度得分；<br>
文章4：提出一种learning-to-rank模型预测情感强度得分。

----------
# The model proposed in this paper:
# Linguistically Regularized LSTM

> Linguistically Regularized
LSTM是把语言学规则（包括情感词典、否定词和程度副词）以约束的形式和LSTM结合起来。

----------

> 首先，我们来看一个例子。我们用单向LSTM从右往左合成 It's not an interesting movie。
从时刻1到时刻2，输入了一个情感词interesting，仅凭直觉就可以断言，代表着interesting movie的情感分布p2，一定比movie的情感分布p1有一个偏移量，这个偏移量应该与情感词interesting保持一致。所以，在我们的模型中，我们使用一个Sentiment Regularizer去约束p1和p2的关系，当然这个regularizer是与interesting相关的，并且是可学习的。
从时刻2到时刻3，输入了一个非情感词an，凭着我们的先验知识，an interesting movie与interesting movie的情感分布应该是近似的，所以我们使用一个Non-Sentiment Regularizer去约束p2和p3的关系。
从时刻3到时刻4，输入了一个否定词not，我们知道加入not后情感应该会发生很大的变化，通常是一定程度的反转。所以我们为否定词not学习一个Negation Regularizer，并用它去约束p3和p4的关系。


![Figure 1](https://leanote.com/api/file/getImage?fileId=5b1111e6ab644116b7001aa1)

> 本文的核心思想是通过规则化句子的相邻位置的输出来对句子层次的情感分类中的情感词，否定词和程度副词的语言角色进行建模。 
如图所示，在"It’s not an interesting movie"中，"an interesting movie"和"interesting movie"中的预测情感分布应该彼此接近，而预测的情感分布在"interesting movie"应该与前面的位置（在后面的方向） ("movie") 完全不同，因为可以看到一个情感词("interesting") 。
<br>
作者定义了四种规则(一个通用规则和三个特殊规则)来将语言学知识结合进来，每一个规则主要是考虑当前词和它相邻位置词的情感分布来定义的。

----------

 - Non-Sentiment Regularizer(NSR)

 

> NSR的基本想法是:
如果相邻的两个词都是non-opinion（不表达意见）的词，那么这两个词的情感分布应该是比较相近的。
将这种想法结合到模型中，有如下定义：

$$ L_{t} ^{NSR} =max(0,D_{KL}(p_{t} ,p_{t-1} )-M ) $$
其中$M$是边界参数，$D_{KL} (p,q)$是对称$KL$散度，$p_{t}$ 是要预测的位置$t$处的词的分布，它的向量表示是$h_{t}$ 。

----------

> 关于$KL$散度：
<br>
$KL$散度是用来衡量两个函数或者分布之间的差异性的一个指标，其原始定义式如下：
<br>
$$D(p||q)=\sum_{i=1}^{n}{p(x)\cdot log\frac{p(x)}{q(x)} } $$
两个分布的差异越大，$KL$散度值越大；
两个分布的差异越小，$KL$散度值越小；
当两个分布相同时，$KL$散度值为$0$
<br>
这里所用的对称$KL$散度定义如下：
$$D_{KL}=\frac{1}{2}\sum_{l=1}^{C}{p(l)log q(l)+q(l)logp(l)}$$   

> 所以我们可以看到，
当相邻的两个词分布较近，$KL$散度小于$M$时，$NSR$的值为$0$；
随着两个词的分布差异增大时，$NSR$值变大。

 - Sentiment Regularizer(SR)

> SR的基本想法是:
如果当前词是情感词典中的词，那么它的情感分布应该和前一个词以及后一个词有明显不同。
例如：$This$ $movie$ $is$ $interesting.$
在位置$t=4$处的词$“interesting”$是一个表达正向情感的词，
所以在$t=4$处的情感分布应该比$t=3$处要$positive$得多。
这个叫做$sentiment$ $drift（情感漂流）$。
<br>
为了将这种想法结合到模型中，
作者提出一个极性漂流分布$s_{c}$，情感词典中的每一类词，有一个漂流分布值$s_{c}$ 。
例如，情感词典中的词可能划分为以下几类：
strong positive，
weakly positive，
weakly negative
strong negative，
对于每一类情感词，有一个漂流分布，由模型学习得到。
<br>
SR定义如下：
<br>
$$p_{t-1}^{(SR)} =p_{t-1} +s_{c(x_{t} )} $$
$$L_{t}^{(SR)}=max(0,D_{KL}(p_{t},p_{t-1}^{(SR)})-M )$$ 
所以我们可以看到，
当前词$t$是情感词典中的词的时候，
前一个位置$t-1$的情感分布加上漂流分布之后，如果与位置$t$的分布相近的话，$SR$值为$0$，
随着其分布差异的增大，$SR$值增大。

 - Negation Regularizer(NR)

> 否定词通常会反转文本的情感倾向（从正向变为负向，或者是从负向变为正向），
但是具体情况是跟否定词本身和它所否定的对象有关的。
例如：“not good”和“not bad”中，“not”的角色并不一样，前者是将正向情感转为负向情感，后者是将负向情感转为中性情感。
<br>
对否定词的处理，本文是这样做的：
针对每一个否定词，提出一个转化矩阵$T_{m}$ ，这个矩阵是模型自己学出来的。
如果当前词是否定词的话，那么它的前一个词或者后一个词经过转化矩阵之后的分布与当前词的分布应该是比较近的。
$$p_{t-1}^{(NR)}=softmax(T_{x_{j} }\times p_{t-1}  )$$ 
$$p_{t+1}^{(NR)} =softmax(T_{x_{j} }\times p_{t+1}  )$$

![figure 2](https://leanote.com/api/file/getImage?fileId=5b0d4528ab644177f100227e)

<br>
我们可以看到，如果在当前词是否定词的情况下，如果它的前一个词或者后一个词与当前词的分布较近，那么$NR$的值比较小。

----------

 - Intensity Regularizer(IR)

> 程度副词改变文本的情感强度（比如从positive变为very positive），这种表达对细粒度情感分析很重要。
<br>
程度副词对于情感倾向的改变和否定词很像，只是改变的程度不同，这里也是通过一个转化矩阵来定义IR的。
如果当前词是程度副词的话，那么它的前一个词或者后一个词，经过转化矩阵之后得到的分布，应该与当前词的分布比较接近。

----------

 - Applying Linguistic Regularizers to Bidirectional LSTM

> 定义一个新的损失函数将上述规则结合到模型中，
$$E(\theta )=-\sum_{i}^{}{y^{i}log p^{i}  }+\alpha \sum_{i}^{}{\sum_{t}^{}{L_{t}^{i} } }  +\beta ||\theta ||^{2} $$
其中，$y^{i}$ 是样本的实际分布，$p^{i}$ 是预测得到的样本分布，$L_{t}^{i}$ 是上述规则中的一个或者多个的组合，$i$是句子的索引，$t$是位置的索引。
<br>
模型训练的目标是最小化损失函数，让样本的实际分布与预测分布尽可能接近的同时，让模型符合上述四种规则。

----------
# Experiment
## 数据

> 本文实验用了两个数据集来验证模型性能，

 - Movie Review (MR)(with two classes { negative, positive})
 - Stanford Sentiment Treebank (SST) (with five classes { very negative, negative, neutral, positive, very positive})

> 两个数据集的具体统计信息如下图所示，

![Figure 3](https://leanote.com/api/file/getImage?fileId=5b0d482dab644179d800230b)

----------
## 实验结果

![Figure 4](https://leanote.com/api/file/getImage?fileId=5b0d4885ab644177f10022d4)

> 从实验结果可以看出:

 - LR-LSTM和LR-BI-LSTM与对应的LSTM相比都有较大提升；
 - LR-BI-LSTM在句子级标注数据上的结果与BI-LSTM在短语级标注数据上的结果基本持平，表明通过引入LR,可以减少标注成本，并得到差不多的结果；
 - 本文的LR-LSTM和LR-BI-LSTM和Tree-LSTM的结果基本持平，但是本文的模型更简单，效率更高，同时省去了短语级 的标注工作。

----------
## 不同规则的效果分析

![Figure 5](https://leanote.com/api/file/getImage?fileId=5b0d4960ab644177f10022e2)

> 从实验结果可以看出，
NSR和SR对提升模型性能最重要，
NR和IR对模型性能提升重要性没有那么强，
可能是因为在测试数据中只有14%的句子中含有否定词，只有23%的句子含有程度副词。
<br>
为了进一步研究NR和IR的作用，作者又分别在仅包含否定词的子数据集（Neg.Sub）和仅包含程度副词的子数据集（Int.Sub）上做了对比实验，实验结果如下图，

![Figure 6](https://leanote.com/api/file/getImage?fileId=5b0d49c2ab644179d8002323)

从实验结果可以看出，

 - 在这些子数据集上，LR-Bi-LSTM的性能优于Bi-LSTM；
 - 去掉NR或者IR的约束，在MR和SST两个数据集上模型性能都有明显下降。

----------
# Conclusion and Future Work

> 这篇文章通过损失函数将语言学规则引入现有的句子级情感分析的LSTM模型。在没有增大模型复杂度的情况下，有效的利用情感词典、否定词和程度副词的信息，在实验数据集上取得了较好效果。
<br>
随着深度学习的发展，人们慢慢忽略了宝贵的经典自然语言资源，如何有效将这部分知识有效地融入到深度学习模型中是一个非常有意义的工作。

----------

> To preserve the simplicity of the proposed models, we do not consider the modification scope of negation and intensity words, though we partially address this issue by applying a minimization operartor and bi-directional LSTM. As future work, we plan to apply the linguistic regularizers to tree-LSTM to address the scope issue since the parsing tree is easier to indicate the modification scope explicitly.

----------

> [参考链接](https://zhuanlan.zhihu.com/p/23906487?refer=c_51425207)

 

 
 
 
 
 

    
