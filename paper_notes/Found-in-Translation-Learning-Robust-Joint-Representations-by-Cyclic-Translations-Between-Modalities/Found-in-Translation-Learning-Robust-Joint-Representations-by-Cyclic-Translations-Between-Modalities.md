# Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities

----------

 - [Paper](https://arxiv.org/pdf/1812.07809.pdf)

----------
# Abstract

> *Multimodal sentiment analysis is a core research area that studies speaker sentiment expressed from the language, visual, and acoustic modalities. The central challenge in multimodal learning involves inferring joint representations that can process and relate information from these modalities. However, existing work learns joint representations by requiring all modalities as input and as a result, the learned representations may be sensitive to noisy or missing modalities at test time. With the recent success of sequence to sequence (Seq2Seq) models in machine translation, there is an opportunity to explore new ways of learning joint representations that may not require all input modalities at test time. In this paper, <font color="#FF00FF">we propose a method to learn robust joint representations by translating between modalities.</font> <font color="#FF8C00">Our method is based on the key insight that translation from a source to a target modality provides a method of learning joint representations using only the source modality as input.</font> <font color="blue">We augment modality translations with a cycle consistency loss to ensure that our joint representations retain maximal information from all modalities.</font> <font color="red">Once our translation model is trained with paired multimodal data, we only need data from the source modality at test time for final sentiment prediction.</font> <font color="green">This ensures that our model remains robust from perturbations or missing information in the other modalities.</font> We train our model with a coupled translation- prediction objective and it achieves new state-of-the-art results on multimodal sentiment analysis datasets: CMU-MOSI, ICT- MMMO, and YouTube. <font color="#FF00FF">Additional experiments show that our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to missing or perturbed modalities.</font>*

----------
# Introduction

> *Existing prior work learns joint representations using multiple modalities as input (Liang et al. 2018; Morency, Mihalcea, and Doshi 2011; Zadeh et al. 2016). However, these joint representations also regain all modalities at test time, making them sensitive to noisy or missing modalities at test time (Tran et al. 2017; Cai et al. 2018).*

![1](https://leanote.com/api/file/getImage?fileId=5c433340ab644154ed000f3a)

----------
# Related Work

![2](https://leanote.com/api/file/getImage?fileId=5c43339fab644154ed000f3d)
![3](https://leanote.com/api/file/getImage?fileId=5c4333e9ab644152f9000f55)
![4](https://leanote.com/api/file/getImage?fileId=5c433419ab644154ed000f41)

----------
# Proposed Approach

 - ***Problem Formulation and Notation***
 - ***Learning Joint Representations***
 - ***Multimodal Cyclic Translation Network***

![5](https://leanote.com/api/file/getImage?fileId=5c433486ab644154ed000f44)

 - ***Coupled Translation-Prediction Objective***
 - ***Hierarchical MCTN for Three Modalities***

![6](https://leanote.com/api/file/getImage?fileId=5c4334c0ab644152f9000f64)

----------
# Experimental Setup

 - ***Dataset and Input Modalities***
 - ***Multimodal Features and Alignment***
 - ***Evaluation Metrics***
 - ***Baseline Models***

----------
# Results and Discussion

 - ***Comparison with Existing Work***

![7](https://leanote.com/api/file/getImage?fileId=5c433512ab644152f9000f65)
![8](https://leanote.com/api/file/getImage?fileId=5c43351fab644154ed000f58)

 - ***Adding More Modalities***

![9](https://leanote.com/api/file/getImage?fileId=5c4335abab644154ed000f5e)
![10](https://leanote.com/api/file/getImage?fileId=5c4335bbab644154ed000f60)
![11](https://leanote.com/api/file/getImage?fileId=5c4335d0ab644152f9000f6b)

  - ***Ablation Studies***

![12](https://leanote.com/api/file/getImage?fileId=5c4335edab644152f9000f6e)
![13](https://leanote.com/api/file/getImage?fileId=5c433605ab644152f9000f70)

----------
# Conclusion

> *This paper investigated learning joint representations via cyclic translations from source to target modalities. During testing, we only need the source modality for prediction which ensures robustness to noisy or missing target modalities. We demonstrate that cyclic translations and seq2seq models are useful for learning joint representations in multimodal environments. <font color="#FF00FF">In addition to achieving new state-of-the-art results on three datasets, our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to all target modalities.</font>*
