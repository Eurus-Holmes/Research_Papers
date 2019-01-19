# Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities

----------

 - [Paper](https://arxiv.org/pdf/1812.07809.pdf)

----------
# Abstract

> *Multimodal sentiment analysis is a core research area that studies speaker sentiment expressed from the language, visual, and acoustic modalities. The central challenge in multimodal learning involves inferring joint representations that can process and relate information from these modalities. However, existing work learns joint representations by requiring all modalities as input and as a result, the learned representations may be sensitive to noisy or missing modalities at test time. With the recent success of sequence to sequence (Seq2Seq) models in machine translation, there is an opportunity to explore new ways of learning joint representations that may not require all input modalities at test time. In this paper, <font color="#FF00FF">we propose a method to learn robust joint representations by translating between modalities.</font> <font color="#FF8C00">Our method is based on the key insight that translation from a source to a target modality provides a method of learning joint representations using only the source modality as input.</font> <font color="blue">We augment modality translations with a cycle consistency loss to ensure that our joint representations retain maximal information from all modalities.</font> <font color="red">Once our translation model is trained with paired multimodal data, we only need data from the source modality at test time for final sentiment prediction.</font> <font color="green">This ensures that our model remains robust from perturbations or missing information in the other modalities.</font> We train our model with a coupled translation- prediction objective and it achieves new state-of-the-art results on multimodal sentiment analysis datasets: CMU-MOSI, ICT- MMMO, and YouTube. <font color="#FF00FF">Additional experiments show that our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to missing or perturbed modalities.</font>*

----------
# Introduction

> *Existing prior work learns joint representations using multiple modalities as input (Liang et al. 2018; Morency, Mihalcea, and Doshi 2011; Zadeh et al. 2016). However, these joint representations also regain all modalities at test time, making them sensitive to noisy or missing modalities at test time (Tran et al. 2017; Cai et al. 2018).*

![Figure 1](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/1.png))

----------
# Related Work

![Figure 2](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/2.png))
![Figure 3](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/3.png))
![Figure 4](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/4.png))

----------
# Proposed Approach

 - ***Problem Formulation and Notation***
 - ***Learning Joint Representations***
 - ***Multimodal Cyclic Translation Network***

![Figure 5](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/5.png))

 - ***Coupled Translation-Prediction Objective***
 - ***Hierarchical MCTN for Three Modalities***

![Figure 6](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/6.png))

----------
# Experimental Setup

 - ***Dataset and Input Modalities***
 - ***Multimodal Features and Alignment***
 - ***Evaluation Metrics***
 - ***Baseline Models***

----------
# Results and Discussion

 - ***Comparison with Existing Work***

![Figure 7](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/7.png))
![Figure 8](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/8.png))

 - ***Adding More Modalities***

![Figure 9](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/9.png))
![Figure 10](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/10.png))
![Figure 11](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/11.png))

  - ***Ablation Studies***

![Figure 12](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/12.png))
![Figure 13](https://github.com/Eurus-Holmes/Research_Papers/raw/master/paper_notes/Found-in-Translation-Learning-Robust-Joint-Representations-by-Cyclic-Translations-Between-Modalities/images/13.png))

----------
# Conclusion

> *This paper investigated learning joint representations via cyclic translations from source to target modalities. During testing, we only need the source modality for prediction which ensures robustness to noisy or missing target modalities. We demonstrate that cyclic translations and seq2seq models are useful for learning joint representations in multimodal environments. <font color="#FF00FF">In addition to achieving new state-of-the-art results on three datasets, our model learns increasingly discriminative joint representations with more input modalities while maintaining robustness to all target modalities.</font>*
