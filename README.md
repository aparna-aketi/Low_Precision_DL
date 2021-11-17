# Low Precision Decentralized Training with Heterogenous Data

Official PyTorch implementation for  "**Low Precision Decentralized Distributed Learning with Heterogenous Data**"
[[Paper]]()

## Abstract 
Decentralized distributed learning is the key to enabling large-scale machine learning (training) on the edge devices utilizing private user-generated local data, without relying on the cloud. However, practical realization of such on-device training is limited by the communication bottleneck, computation complexity of training deep models and significant data distribution skew across devices. Many feedback-based compression techniques have been proposed in the literature to reduce the communication cost and a few works propose algorithmic changes to aid the performance in the presence of skewed data distribution by improving convergence rate. To the best of our knowledge, there is no work in the literature that applies and shows compute efficient training techniques such quantization, pruning etc., for peer-to-peer decentralized learning setups. In this paper, we analyze and show the convergence of low precision decentralized training that aims to reduce computational complexity of training and inference. Further, We study the effect of degree of skew and communication compression on the low precision decentralized training over various computer vision and Natural Language Processing (NLP) tasks. Our experiments indicate that 8-bit decentralized training has minimal accuracy loss compared to its full precision counterpart even with heterogeneous data. However, when low precision training is accompanied by communication compression through sparsification we observe 1-2% drop in accuracy. The proposed low precision decentralized training decreases computational complexity, memory usage, and communication cost by ~4x while trading off less than a 1% accuracy for both IID and non-IID data. In particular, with higher skew values, we observe an increase in accuracy (by ~0.5%) with low precision training, indicating the regularization effect of the quantization. 

## Experiments
This repository currently contains experiments reported in the paper for Low precision CHOCO-SGD and Deep-Squeeze.

### Datasets
* CIFAR-10
* CIFAR-100
* Imagenette

### Models
* ResNet
* VGG
* MobileNet

```python
sh run.sh
```

### Citation
```
@inproceedings{
aketi2021,
title={Low Precision Decentralized Distributed Learning with Heterogenous Data},
author={Sai Aparna Aketi, Sangamesh Kodge, and Kaushik Roy},
booktitle={arXiv pre-print},
year={2021},
url={}
}
```
