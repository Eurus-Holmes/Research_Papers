# Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets

----------
> [原文链接](http://www.aclweb.org/anthology/W17-5202)

----------
# Abstract

> It is hard to understand how well a certain model generalizes across different tasks and datasets. 
In this paper, we contribute to this situation by comparing several models on six different benchmarks, which belong to different domains and additionally have different levels of granularity (binary, 3-class, 4-class and 5-class). 
We show that Bi- LSTMs perform well across datasets and that both LSTMs and Bi-LSTMs are particularly good at fine-grained sentiment tasks (i. e., with more than two classes). 
Incorporating sentiment information into word embeddings during training gives good results for datasets that are lexically similar to the training data. 
With our experiments, we contribute to a better understanding of the performance of different model architectures on different data sets. 
Consequently, we detect novel state-of-the-art results on the SenTube datasets.

----------

> 本文通过比较6个不同领域数据集的7个模型，来更好地理解不同模型架构在不同数据集上的性能。
结果显示Bi-LSTM在数据集中表现良好，并且LSTM和Bi-LSTM特别擅长细粒度的情感分析任务（即具有两个以上的类）。
在训练期间将情感信息融入词向量练习对于与训练数据在词汇上相似的数据集产生了良好的结果。 
通过比较实验结果，作者还惊喜地在SenTube数据集上检测到了新颖的最优结构模型。

----------

# Datasets
![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/1.png)


 - The Stanford Sentiment Treebank (SST-fine) is a dataset of movie reviews.

> In order to compare with [[Faruqui et al. (2015)]](http://www.aclweb.org/anthology/N15-1184), we also adapt the dataset to the task of binary sentiment analysis, where strong negative and negative are mapped to one label, and strong positive and positive are mapped to another label, and the neutral examples are dropped. This leads to a slightly different split of 6920/872/1821 (we refer to this dataset as SST- binary).

 - The OpeNER dataset is a dataset of hotel reviews in which each review is annotated for opinions. 
 - The SenTube datasets are texts that are taken from YouTube comments regarding automobiles (SenTube-A) and tablets (SenTube-T).
 - The SemEval 2013 Twitter dataset (SemEval) is a dataset that contains tweets collected for the 2013 SemEval shared task B.

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/2.png)


----------

# Models

## Baselines

> First, we train an L2-regularized logistic regression on a bag-of-words representation (BOW) of the training examples, where each example is represented as a vector of size n, with n = |V | and V the vocabulary. This is a standard baseline for text classification.

----------

> Our second baseline is an L2-regularized logistic regression classifier trained on the average of the word vectors in the training example (AVE). We train word embeddings using the skip-gram with negative sampling algorithm [[Mikolov et al., 2013]](https://arxiv.org/pdf/1301.3781.pdf) on a 2016 Wikipedia dump, using 50-, 100-, 200-, and 600-dimensional vectors, a window size of 10, 5 negative samples, and we set the subsampling parameter to 10−4. Additionally, we use the publicly available 300-dimensional GoogleNews vectors3 in order to compare to previous work.

----------
## Retrofitting

> We apply the approach by [[Faruqui et al. (2015)]](http://www.aclweb.org/anthology/N15-1184) and make use of the [code](https://github.com/mfaruqui/retrofitting) released in combination with the PPDB-XL lexicon, as this gave the best results for sentiment analysis in their experiments. We train for 10 iterations. Following the authors’ setup, for testing we train an L2-regularized logistic re- gression classifier on the average word embeddings for a phrase (RETROFIT).

----------

## Joint Training

> For the joint method, we use the 50-dimensional sentiment embeddings provided by [[Tang et al. (2014)]](http://www.aclweb.org/anthology/P14-1146). Additionally, we create 100-, 200-, and 300-dimensional embeddings using the code that is publicly available. We use the same hyperpa- rameters as [[Tang et al. (2014)]](http://www.aclweb.org/anthology/P14-1146): five million positive and negative tweets crawled using hashtags as prox- ies for sentiment, a 20-dimensional hidden layer, and a window size of three. Following the authors’ setup, we concatenate the maximum, minimum and average vectors of the word embeddings for each phrase. We then train a linear SVM on these representations (JOINT).

----------

## Supervised Training

![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/3.png)
![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/4.png)
![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/5.png)
![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/6.png)

----------

# Results
![Figure 7](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/7.png)
![Figure 8](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/8.png)
![Figure 9](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/9.png)
![Figure 10](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/10.png)
![Figure 11](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Assessing-State-of-the-Art-Sentiment-Models-on-State-of-the-Art-Sentiment-Datasets/images/11.png)

----------




