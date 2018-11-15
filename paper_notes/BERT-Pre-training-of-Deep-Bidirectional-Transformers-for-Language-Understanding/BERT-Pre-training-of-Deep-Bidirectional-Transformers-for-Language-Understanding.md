# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

----------

 - [Paper](https://arxiv.org/abs/1810.04805)
 - [Code (from Google AI --- Tensorflow)](https://github.com/google-research/bert)
 - [Code (from codertimo --- Pytorch)](https://github.com/codertimo/BERT-pytorch)
 - [Code (from Separius --- Keras)](https://github.com/Separius/BERT-keras)

----------
# Abstract

> *We introduce a new language representation model called BERT, which stands for <font color="#dd00dd">Bidirectional Encoder Representations from Transformers</font>. Unlike recent language representation models, BERT is designed to <font color="#dd00dd">pre-train deep bidirectional representations</font> by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be <font color="#dd00dd">fine-tuned</font> with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.* <br><br>
*BERT is conceptually simple and empirically powerful. <font color="#dd00dd">It obtains new state-of-the-art results on eleven natural language processing tasks</font>, including pushing the GLUE bench-mark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7% (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5 absolute improvement), <font color="#dd00dd">outperforming human performance by 2.0.</font>*

----------
# BERT

### Model Architecture

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/1.png)

### Input Representation

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/2.png)

### Pre-training Tasks

 - ***Task #1: Masked LM***
*There are two downsides to such an approach. The first is that we are creating a mismatch between pre-training and fine- tuning, since the [MASK] token is never seen during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual [MASK] token. Instead, the training data generator chooses 15% of tokens at random, e.g., in the sentence my dog is hairy it chooses hairy. It then performs the following procedure:*
![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/3.png)
![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/4.png)
 - ***Task #2: Next Sentence Prediction***
*Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus. For example:*
![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/5.png)
 
### Pre-training Procedure

![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/6.png)

*emmm...However...*

*Training of $BERT_{BASE}$ was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).*

*Training of $BERT_{LARGE}$ was performed on 16 Cloud TPUs (64 TPU chips total). Each pre-training took 4 days to complete.*

### Fine-tuning Procedure

![Figure 7](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/7.png)

### Comparison of BERT and OpenAI GPT

 - *GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M words).*
 - *GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training.*
 - *GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.*
 - *GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.*

----------
# Experiments

> *In this section, we present BERT fine-tuning results on 11 NLP tasks.*

![Figure 8](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/8.png)

### GLUE Datasets

 - **MNLI** *Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task. Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.*
 - **QQP** *Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent.*
 - **QNLI** *Question Natural Language Inference is a version of the Stanford Question Answering Dataset which has been converted to a binary classification task. The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.*
 - **SST-2** *The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment.*
 - **CoLA** *The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not.*
 - **STS-B** *The Semantic Textual Similarity Bench-mark is a collection of sentence pairs drawn from news headlines and other sources. They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning.*
 - **MRPC** *Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.*
 - **RTE** *Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data.*
 - **WNLI** *Winograd NLI is a small natural language inference dataset deriving from. The GLUE webpage notes that there are issues with the construction of this dataset, and every trained system that’s been submitted to GLUE has has performed worse than the 65.1 baseline accuracy of predicting the majority class. We therefore exclude this set out of fairness to OpenAI GPT. For our GLUE submission, we always predicted the majority class.*

![Figure 9](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/9.png)

### SQuAD v1.1

> *The Standford Question Answering Dataset (SQuAD) is a collection of 100k crowdsourced question/answer pairs. Given a question and a paragraph from Wikipedia containing the answer, the task is to predict the answer text span in the paragraph. For example:*
![Figure 10](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/10.png)

![Figure 11](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/11.png)

### Named Entity Recognition

> *To evaluate performance on a token tagging task, we fine-tune BERT on the CoNLL 2003 Named Entity Recognition (NER) dataset. This dataset consists of 200k training words which have been annotated as Person, Organization, Location, Miscellaneous, or Other (non-named entity).*

![Figure 12](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/12.png)

### SWAG

> *The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference.*

![Figure 13](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/13.png)

----------
# Ablation Studies

### Effect of Pre-training Tasks

![Figure 14](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/14.png)

### Effect of Model Size

![Figure 15](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/15.png)

### Effect of Number of Training Steps

![Figure 16](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/16.png)

### Feature-based Approach with BERT

![Figure 17](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding/images/17.png)

----------
# Conclusion

> *Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from very deep unidirectional architectures. <font color="#dd00dd">Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks</font>.*
<br>
*While the empirical results are strong, <font color="#dd00dd">in some cases surpassing human performance</font>, important future work is to investigate the linguistic phenomena that may or may not be captured by BERT.*


----------
# Related Links

 - [最强NLP预训练模型！谷歌BERT横扫11项NLP任务记录](https://www.jiqizhixin.com/articles/2018-10-12-13)
 - [【NLP】Google BERT详解](https://zhuanlan.zhihu.com/p/46652512)
 - [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
 - [NLP的游戏规则从此改写？从word2vec, ELMo到BERT](https://zhuanlan.zhihu.com/p/47488095)
 - [谷歌终于开源BERT代码：3 亿参数量，机器之心全面解读](https://zhuanlan.zhihu.com/p/48266680)
 - [全面超越人类！Google称霸SQuAD，BERT横扫11大NLP测试](https://zhuanlan.zhihu.com/p/46648916)
