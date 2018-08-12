# SoundNet-Learning-Sound-Representations-from-Unlabeled-Video

----------
> [***Source***](http://soundnet.csail.mit.edu/)

----------

> [***github***](https://github.com/cvondrick/soundnet)

----------
# Abstract

> *We learn rich natural sound representations by capitalizing on large amounts of unlabeled sound data collected in the wild. We leverage the natural synchronization between vision and sound to learn an acoustic representation using two-million unlabeled videos. Unlabeled video has the advantage that it can be economically acquired at massive scales, yet contains useful signals about natural sound. We propose a student-teacher training procedure which transfers discriminative visual knowledge from well established visual recognition models into the sound modality using unlabeled video as a bridge. Our sound representation yields significant performance improvements over the state-of-the-art results on standard benchmarks for acoustic scene/object classification. Visualizations suggest some high-level semantics automatically emerge in the sound network, even though it is trained without ground truth labels.*

----------
# 1 Introduction

> *We present a deep convolutional network that learns directly on raw audio waveforms, which is trained by transferring knowledge from vision into sound\......Since we can leverage large amounts of unlabeled sound data, it is feasible to train deeper networks without significant overfitting, and our experiments suggest deeper models perform better. Visualizations of the representation suggest that the network is also learning high-level detectors, such as recognizing bird chirps or crowds cheering, even though it is trained directly from audio without ground truth labels\......The primary contribution of this paper is the development of a large-scale and semantically rich representation for natural sound.* 

## Related Work

 - *Sound Recognition*
 - *Transfer Learning*
 - *Cross-Modal Learning and Unlabeled Video*

----------
# 2 Large Unlabeled Video Dataset

> *We downloaded over two million videos from Flickr by querying for popular tags and dictionary words, which resulted in over one year of continuous natural sound and video, which we use for training.*

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/1.png)


> *We wish to process sound waves in the raw. Hence, the only post-processing we did on the videos was to convert sound to MP3s, reduce the sampling rate to 22 kHz, and convert to single channel audio.* 

----------
# 3 Learning Sound Representations

 - *Deep Convolutional Sound Network*
    - *Convolutional Network*
    - *Variable Length Input/Output*
    - *Network Depth*

*We visualize the eight-layer network architecture in Figure 1, which conists of 8 convolutional layers and 3 max-pooling layers. We show the layer configuration in Table 1 and Table 2.*

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/2.png)

![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/3.png)

 - *Visual Transfer into Sound*
 - *Sound Classification*
 - *Implementation*

    
----------
# 4 Experiments

 - *Acoustic Scene Classification*
    - *DCASE*
    - *ESC-50 and ESC-10*
    - *Comparison to State-of-the-Art*

![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/4.png)

 - *Ablation Analysis*
    - *Comparison of Loss and Teacher Net*
    - *Comparison of Network Depth*
    - *Comparison of Supervision*
    ![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/5.png)
    - *Comparison of Layer and Teacher Network*
    ![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/6.png)
  - *Multi-Modal Recognition*
    - *Vision vs. Sound Embeddings*
    ![Figure 7](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/7.png)
    - *Object and Scene Classification*
    ![Figure 8](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/8.png)
 - *Visualizations*
 ![Figure 9](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/SoundNet-Learning-Sound-Representations-from-Unlabeled-Video/images/9.png)
    
----------
# 5 Conclusion

> *We propose to train deep sound networks (SoundNet) by transferring knowledge from established vision networks and large amounts of unlabeled video. The synchronous nature of videos (sound + vision) allow us to perform such a transfer which resulted in semantically rich audio representations for natural sounds. Our results show that transfer with unlabeled video is a powerful paradigm for learning sound representations. All of our experiments suggest that one may obtain better performance simply by downloading more videos, creating deeper networks, and leveraging richer vision models.*

----------
