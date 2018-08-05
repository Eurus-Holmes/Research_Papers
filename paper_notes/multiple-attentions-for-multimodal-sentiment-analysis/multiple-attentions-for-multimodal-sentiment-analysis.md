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

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/1.png)


## Training

 - Unimodal Classification
 - Multimodal Classification
    - Single-Level Framework
    - Multi-Level Framework

----------
# Experimental Results

 - Dataset details

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/2.png)

 - Different Models and Network Architectures
 - Single-Level vs Multi-level Framework
 
![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/3.png)
 - AT-Fusion Performance
 - Comparison Among the Models
    - Comparison with the state of the art
    
![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/4.png)
    - unimodal-SVM
    - CAT-LSTM vs Simple-LSTM
    
![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/5.png)

 - Importance of the Modalities
> As expected, bimodal classifiers dominate unimodal clas- sifiers and trimodal classifiers perform the best among all. Across modalities, textual modality performs better than the other two, thus indicating the need for better feature extraction for audio and video modalities.
 - Qualitative Analysis and Case Studies

![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/multiple-attentions-for-multimodal-sentiment-analysis/images/6.png)

----------
# Conclusion
 > We developed a new framework that models contextual information obtained from other relevant utterances while classifying one target utterance. 

----------


     

    
