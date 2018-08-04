# Multi-level Multiple Attentions for Contextual Multimodal Sentiment Analysis
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
