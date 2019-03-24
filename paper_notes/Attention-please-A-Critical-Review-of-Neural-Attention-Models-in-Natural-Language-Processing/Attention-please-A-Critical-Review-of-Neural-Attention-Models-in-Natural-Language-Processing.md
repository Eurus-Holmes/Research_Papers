# Attention, please! A Critical Review of Neural Attention Models in Natural Language Processing

----------

 - [Paper](https://github.com/Eurus-Holmes/Research_Papers/blob/master/papers/Attention-please-A-Critical-Review-of-Neural-Attention-Models-in-Natural-Language-Processing.pdf)

----------
# Abstract

> *Attention is an increasingly popular mechanism used in a wide range of neural architectures. Because of the fast-paced advances in this domain, a systematic overview of attention is still missing. In this article, we define a unified model for attention architectures for natural language processing, with a focus on architectures designed to work with vector representation of the textual data. We discuss the dimensions along which proposals differ, the possible uses of attention, and chart the major research activities and open challenges in the area.*

----------
# 1. Introduction

![1](https://leanote.com/api/file/getImage?fileId=5c94773fab6441495d0015cb)

----------
# 2. The Attention Function

> *The core idea behind attention is to compute a weight distribution on the input sequence, assigning higher values to more relevant elements.*

----------
## 2.1. An Example for Machine Translation and Alignment

![2](https://leanote.com/api/file/getImage?fileId=5c94779eab64414762001505)

----------
## 2.2. A Unified Attention Model

![3](https://leanote.com/api/file/getImage?fileId=5c947979ab6441495d0015e3)

![4](https://leanote.com/api/file/getImage?fileId=5c947bb8ab6441495d0015ee)

![5](https://leanote.com/api/file/getImage?fileId=5c947bd8ab64414762001513)

![6](https://leanote.com/api/file/getImage?fileId=5c947d09ab6441476200152d)

![7](https://leanote.com/api/file/getImage?fileId=5c947d65ab6441495d0015fe)

![8](https://leanote.com/api/file/getImage?fileId=5c948504ab6441476200156c)

----------
## 2.3. The Uses of Attention

> *Attention enables to estimate the relevance of the input elements as well as to combine said elements into a compact representation – the context vector – that condenses the characteristics of the most relevant elements. Because the context vector is smaller than the original input, it requires fewer computational resources to be processed at later stages, yielding a computational gain.*

----------
# 3. A Taxonomy for Attention Models

----------
## 3.1 Input Representation

![9](https://leanote.com/api/file/getImage?fileId=5c95fea6ab64417c52000371)

 - ***3.1.1 HIERARCHICAL INPUT ARCHITECTURES***

![10](https://leanote.com/api/file/getImage?fileId=5c96019dab64417c520003bd)

----------
## 3.2 Compatibility Functions

![11](https://leanote.com/api/file/getImage?fileId=5c96023eab64417c520003be)

----------
## 3.3 Distribution functions

> *Attention distribution maps energy scores to attention weights. The choice of the distribution function depends on the properties the distribution is required to have—for instance, whether it is required to be a probability distribution, a set of probability scores, or a set of Boolean scores—on the need to enforce sparsity, and on the need to take into account the keys’ positions.*

----------
## 3.4 Multiplicity

 - ***3.4.1 MULTIPLE OUTPUTS***

> *Some applications suggest that the data could, and should, be interpreted in multiple ways. This can be the case when there is ambiguity in the data, stemming, for example, from words having multiple meanings, or when addressing a multi-task problem. For this reason, models have been defined that jointly compute not only one, but multiple attention distributions over the same data.*

  - ***3.4.2 MULTIPLE INPUTS: CO-ATTENTION***
    - ***Coarse-grained co-attention***
![12](https://leanote.com/api/file/getImage?fileId=5c97273dab64417e5f0006a1)
    - ***Fine-grained co-attention***
![13](https://leanote.com/api/file/getImage?fileId=5c97286bab64417c52000749)    
![14](https://leanote.com/api/file/getImage?fileId=5c9728caab64417e5f0006b9)

----------
# 4. Combining Attention and Knowledge

----------
## 4.1 Supervised Attention

 - ***4.1.1 PRELIMINARY TRAINING***
 - ***4.1.2 AUXILIARY TRAINING***
 - ***4.1.3 TRANSFER LEARNING***

----------
## 4.2 Attention tracking

> *When attention is applied multiple times on the same data, as in sequence-to-sequence models, a useful piece of information could be how much relevance has been given to the input along different model iterations. Indeed, one may need to keep track of the weights that the attention model assigns to each input. For example, in machine translation it is desirable to ensure that all the words of the original phrase are taken into account. One possibility to maintain this information is to use a suitable structure and provide it as an additional input to the attention model. Tu et al. (2016) exploit a piece of symbolic information called coverage to keep track of the weight associated to the inputs. Every time attention is computed, such information is fed to the attention model as a query element, and it is updated according to the output of the attention itself. In Mi et al.’s (2016a) work, the representation is enhanced by making use of a sub-symbolic representation for the coverage.*

----------
## 4.3 Modelling the distribution function according to background knowledge

> *Another component of the attention model where background knowledge can be exploited is the distribution function. For example, constraints can be applied on the computation of the new weights to enforce the boundaries on the weights assigned to the inputs. In Martins and Kreutzer’s (2017), Malaviya et al.’s (2018) work, the coverage information is exploited by a constrained distribution function, regulating the amount of attention that the same word can receive over time.*

----------
# 5. Challenges and Future Directions

----------

 - ***5.1 Attention for deep networks investigation***
 - ***5.2 Attention for outlier detection and sample weighing***
 - ***5.3 Attention analysis for model evaluation***
 - ***5.4 Unsupervised learning with attention***

> *To properly exploit unsupervised learning is widely recognized as one of the most important long-term challenges of AI (LeCun et al., 2015). As already mentioned in Section 4, attention is typically trained in a supervised architecture, although without a direct supervision on the attention weights. Nevertheless, a few works have recently attempted to exploit attention within purely unsupervised models. We believe this to be a promising research direction, as the learning process of humans is indeed largely unsupervised.*


----------
# 6. Conclusion

> ***Attention models have nowadays become widespread in NLP applications. By integrating attention in neural architectures, two positive effects are jointly obtained: a performance gain, and a means of investigating the network’s behaviour.***

----------

> ***We have shown how attention can be applied to different input parts, different representations of the same data, or different features. The attention mechanism enables to obtain a compact representation of the data as well as to highlight relevant information. The selection is performed through a distribution function, which may take into account locality in different dimensions, such as space, time, or even semantics. Attention can also be modeled so as to compare the input data with a given element (a query) based on similarity or significance. But it can also learn the concept of relevant element by itself, thus creating a representation to which the important data should be similar.***

----------


> ***We have also discussed the possible role of attention in addressing fundamental AI challenges. In particular, we have shown how attention can be a means of injecting knowledge into the neural model, so as to represent specific features, or to exploit knowledge acquired previously, as in transfer learning settings. We speculate that this could pave the way to new challenging research avenues, where attention could be exploited to enforce the combination of sub-symbolic models with symbolic knowledge representations, especially to perform reasoning tasks, or to address natural language understanding. In a similar vein, attention could be a key ingredient of unsupervised learning architectures, as recent works suggest, by guiding and focusing the training process where no supervision is given in advance.***

----------


