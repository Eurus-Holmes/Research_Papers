# Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis

----------
> [论文链接](https://github.com/Eurus-Holmes/Research_Papers/blob/master/papers/multiple-attentions-for-multimodal-sentiment-analysis.pdf)

----------
# Abstract

> We propose a recurrent model that is able to capture contextual information among utterances. In this paper, we also introduce **attention-based** networks for improving both context learning and dynamic feature fusion. 

----------
# Introduction

----------
# Method
## Problem Definition

> Let us assume a video to be considered as $V_j = [u_{j,1} u_{j,2} u_{j,3} , ..., u_{j,i} ...u_{j,L_j} ] $where $u_{j,i}$ is the $i^{th}$ utterance in video $v_j$ and $L_j$ is the number utterances in the video. The goal of this approach is to label each utterance $u_{j,i}$ with the sentiment expressed by the speaker. We claim that, in order to classify utterance $u_{j,i}$, the other utterances in the video, i.e., $[u_{j,k} ∣ ∀k ≤ L_j,k ≠ i]$, serve as its context and provide key information for the classification.

## Overview of the Approach

 - Unimodal feature extraction
    - Textual Features Extraction
    - Audio Feature Extraction
    - Visual Feature Extraction
 - AT-Fusion – Multimodal fusion using the attention mechanism
 - CAT-LSTM – Attention-based LSTM model for sentiment classification

![1](https://leanote.com/api/file/getImage?fileId=5b6561f9ab64415f640034b1)

## Training

 - Unimodal Classification
 - Multimodal Classification
    - Single-Level Framework
    - Multi-Level Framework

----------
# Experimental Results

 - Dataset details

![2](https://leanote.com/api/file/getImage?fileId=5b6574bdab64415f64003eb4)

 - Different Models and Network Architectures
 - Single-Level vs Multi-level Framework
![3](https://leanote.com/api/file/getImage?fileId=5b6578c4ab6441615b0045cd)
 - AT-Fusion Performance
 - Comparison Among the Models
    - Comparison with the state of the art
![4](https://leanote.com/api/file/getImage?fileId=5b65781dab6441615b0045c1)
    - unimodal-SVM
    - CAT-LSTM vs Simple-LSTM
![5](https://leanote.com/api/file/getImage?fileId=5b65791dab64415f64003f27)

 - Importance of the Modalities
> As expected, bimodal classifiers dominate unimodal clas- sifiers and trimodal classifiers perform the best among all. Across modalities, textual modality performs better than the other two, thus indicating the need for better feature extraction for audio and video modalities.
 - Qualitative Analysis and Case Studies

![6](https://leanote.com/api/file/getImage?fileId=5b66d9c1ab64413ab400134e)

----------
# Conclusion
 > We developed a new framework that models contextual information obtained from other relevant utterances while classifying one target utterance. 

----------


     

    
