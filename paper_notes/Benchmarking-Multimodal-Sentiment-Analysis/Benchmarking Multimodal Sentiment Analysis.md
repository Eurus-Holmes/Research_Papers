# Benchmarking Multimodal Sentiment Analysis

> [论文链接](https://github.com/Eurus-Holmes/Research_Papers/blob/master/papers/Benchmarking%20Multimodal%20Sentiment%20Analysis.pdf)

----------
# Abstract

> **We propose a framework for multimodal sentiment analysis and emotion recognition using convolutional neural network-based feature extraction from text and visual modalities.** We obtain a performance improvement of 10% over the state of the art by **combining visual, text and audio features.** We also discuss some major issues frequently ignored in multimodal sentiment analysis research: **the role of speaker-independent models**, **importance of the modalities and generalizability.** The paper thus serve as a new benchmark for further research in mul- timodal sentiment analysis and also demonstrates the different facets of analysis to be considered while performing such tasks.

----------
# 1 Introduction

----------
# 2 Related Work

> In this paper, we propose CNN-based framework for feature extraction from visual and text modality and a method for fusing them for multimodal sentiment analysis and emotion recognition. 

----------
# 3 Method

 - **Textual Features**
 - **Audio Features**
 - **Visual Features**
 - **Fusion**

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/1.png)

----------
# 4 Experiments and Observations

----------
### 4.1 Datasets

 - Multimodal Sentiment Analysis Datasets
    - MOUD dataset
    - MOSI dataset
 - Multimodal Emotion Recognition Dataset
    - IEMOCAP dataset

----------
### 4.2 Speaker-Independent Experiment

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/2.png)

![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/3.png)

![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/4.png)

----------
### 4.3 Contributions of the Modalities

![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/5.png)

----------
### 4.4 Generalizability of the Models

![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/6.png)

----------
### 4.5 Visualization of the Datasets

![Figure 7](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Benchmarking-Multimodal-Sentiment-Analysis/images/7.png)

----------
# 5 Qualitative Analysis

> The above examples demonstrates the effectiveness and robustness of our model to capture overall video semantics of the utterances for emotion and sen- timent detection. It also shows how bi and multimodal models, given the multiple media input, overcomes the limitations of unimodal networks.
We also explored the misclassified validation utterances and found some interesting trends. A video is constituent of a group of utterances which have contextual dependencies among them. Thus, **our model failed to classify utterances whose emotional polarity was highly dependent on the context described in earlier or later part of the video.** However, such interdependent modeling was out of the scope of this paper and we therefore enlist it as a future work.

----------
# 6 Conclusion

> We have presented a framework for multimodal sentiment analysis and multimodal emotion recognition, which outperforms the state of the art in both tasks by a significant margin. Apart from that, we also discuss some major aspects of multimodal sentiment analysis problem such as the performance of speaker-independent models and cross dataset performance of the models.
**Our future work will focus on extracting semantics from the visual features, relatedness of the cross-modal features and their fusion.** We will also include contextual dependency learning in our model to overcome limitations mentioned in Section 5.

