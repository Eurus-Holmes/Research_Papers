# Acoustic Features Fusion using Attentive Multi-channel Deep Architecture

----------

 - [Paper](https://arxiv.org/pdf/1811.00936.pdf)
 - [Code](https://github.com/DeepLearn-lab/Acoustic-Feature-Fusion_Chime18)

----------
# Abstract

> *In this paper, we present a novel deep fusion architecture for audio classification tasks. <font color="#dd00dd">The multi-channel model presented is formed using deep convolution layers where different acoustic features are passed through each channel.</font> <font color="blue">To enable dissemination of information across the channels, we introduce attention feature maps that aid in the alignment of frames.</font> <font color="green">The output of each channel is merged using interaction parameters that non-linearly aggregate the representative features.</font> Finally, we evaluate the performance of the proposed architecture on three benchmark datasets :- DCASE-2016 and LITIS Rouen (acoustic scene recognition), and CHiME-Home (tagging). Our experimental results suggest that the architecture presented outperforms the standard baselines and achieves outstanding performance on the task of acoustic scene recognition and audio tagging.*

----------
# Complementary Acoustic Features

> *The Mel and log-Mel are a set of complementary acoustic features (CAF). <font color="#FF8C00">The Mel frequencies capture classes which lie in the higher frequency domain and log-Mel frequencies capture classes that lie in the lower frequency domain.</font> We conjecture that passing the features via a multi-channel model it is possible to efficiently combine the complementary properties inhibited by these features.*

----------
# Multi-Channel Deep Fusion

![1](https://leanote.com/api/file/getImage?fileId=5bfbc6f2ab6441789a003d2f)

 - ***Early fusion***
![2](https://leanote.com/api/file/getImage?fileId=5bfbc761ab644176aa00448f)
![3](https://leanote.com/api/file/getImage?fileId=5bfbc70aab6441789a003d36)
 - ***Late Fusion***
![4](https://leanote.com/api/file/getImage?fileId=5bfbc782ab644176aa004499)
 - ***Parameters Sharing***
![5](https://leanote.com/api/file/getImage?fileId=5bfbc799ab6441789a003d74)

----------
# Experiments and Results

### *Datasets*

 - ***DCASE-2016***
 - ***LITIS Rouen***
 - ***CHiME-Home***
 
### *Hyperparameters*
![6](https://leanote.com/api/file/getImage?fileId=5bfbc800ab6441789a003d8d)

### *Baselines*

> *For <font color="blue">DCASE-2016</font> we use the <font color="blue">Gaussian Mixture Model (GMM) with MFCC (including acceleration and delta coefficients)</font> as the baseline system. This baseline is provided by DCASE-2016 organizers. The other baseline used is <font color="blue">DNN with mel-components</font>. For <font color="#FF8C00">LITIS-Rouen</font> we use the <font color="#FF8C00">HOG+CQA and DNN + MFCC </font>results as the baseline. For the <font color="#FF00FF">CHiME-Home</font> dataset, we use the standard baseline of the <font color="#FF00FF">MFCC+GMM system and mel+DNN</font>.*

### *Results*

![7](https://leanote.com/api/file/getImage?fileId=5bfbc9c1ab6441789a003e04)

![8](https://leanote.com/api/file/getImage?fileId=5bfbc9dcab644176aa004545)
> *Note: vanilla (no fusion), early fusion (EF), late fusion (LF) and hybrid (EF+LF)*

![9](https://leanote.com/api/file/getImage?fileId=5bfbca3aab644176aa004563)
![10](https://leanote.com/api/file/getImage?fileId=5bfbca4dab644176aa004569)
![11](https://leanote.com/api/file/getImage?fileId=5bfbca9cab6441789a003e46)

----------
# Conclusion

> *In this paper, we present a multi-channel architecture for the fusion of complementary acoustic features. Our idea is based on the fact that the introduction of attention parameters between the channels results in better convergence. The proposed technique is general and can be applied to any audio classification task. <font color="#dd00dd">A possible extension to our work would be to use the pairs or triplets of audio samples of similar classes and pass them through the multi-channel architecture.</font> This could help to align the diverse audio samples of similar classes, making the model robust to audio samples that are difficult to classify.*
