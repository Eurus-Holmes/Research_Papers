# 3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition

----------
> [论文链接](https://github.com/xuanjihe/speech-emotion-recognition/blob/master/3-D.pdf)

----------
# Abstract

> *Speech emotion recognition (SER) is a difficult task due to the complexity of emotion. The SER performances are heavily depend on the effectiveness of emotional features extracted from speech. However, most emotional features are sensitive to emotional irrelevant factors, such as the speaker, speaking styles and environment. In this letter, we assume that calculating the deltas and delta-deltas for personalized features not only preserves effective emotional information but also reduce the influence of emotional irrelevant factors, leading to reduce misclassification. In addition, SER often suffers from the silent frames and emotional irrelevant frames. Meanwhile, attention mechanism has exhibited outstanding performances in learning relevant feature representations for specific tasks. Inspired by this, we propose a 3-D attention-based convolutional recurrent neural networks (ACRNN) to learn discriminative features for SER, where the Mel-spectrogram with deltas and delta-deltas are used as input. Experiments on IEMOCAP and Emo-DB corpus demonstrate the effectiveness of the proposed method and achieve the state-of-the-art performance in terms of unweighted average recall.*

----------
# I. INTRODUCTION

----------
# II. PROPOSED METHODOLOGY

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/1.png)

 - *3-D Log-Mels generation*

>  *We use the log-Mels with deltas and delta-deltas as the ACRNN input, where the deltas and delta-deltas reflect the process of emotional change.*

 - *Architecture of ACRNN*
    - *CRNN Model*
    - *Attention Layer*
    
![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/2.png)
  
----------
# III. EXPERIMENTS

 - *Experiment Setup*
    - *Interactive Emotional Dyadic Motion Capture database (IEMOCAP)* 
    - *Berlin Emotional Database (Emo-DB)*

> *Since both databases contain 10 speakers, we employ a 10-fold cross- validation technique in our evaluations. Specifically, for each evaluation, 8 speakers are selected as the training data and 1 speaker is select as the validation data, while the remaining 1 speaker is used as the test data......We repeat each evaluation for five times with different random seeds and report the average and standard deviation to get more reliable results.Since the test class distributions are imbalanced, we report unweighted average recall (UAR) on the test set.Note that, all model architectures, including the number of epochs are selected by maximizing the UAR on the validation set.*

 - *Baselines*
    - *DNN-ELM*
    - *2-D ACRNN*
 - *Comparison of Network Architectures*

> *We use ARCNN to execute SER on the Log- Mels with increasing convolutional layer.*

![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/3.png)

  - *Experiment Results*

  ![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/4.png)
  
  ![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/5.png)
  
  ![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/3-D-Convolutional-Recurrent-Neural-Networks-with-Attention-Model-for-Speech-Emotion-Recognition/images/6.png)

     

----------
# IV. CONCLUSION

> *In this letter, we proposed a 3-D attention-based convolut- ional recurrent neural networks (ACRNN) for SER. We first extract log-Mels (static, deltas and delta-deltas) from speech signals as the 3-D CNN input. Next, we combine 3-D CNN with LSTM for high-level features extraction. Finally, an attention layer is used to focus on the emotional relevant parts and produce utterance-level affective-salient features for SER. Experiments on the IEMOCAP and Emo-DB databases show the superiority of our proposed approach compared with the baselines in terms of UAR.*

----------


