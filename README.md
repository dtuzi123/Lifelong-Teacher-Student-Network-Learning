>📋 The implementation of Lifelong Teacher-Student Network Learning

# Title : Lifelong Teacher-Student Network Learning

This repository is the implementation of [Lifelong Teacher-Student Network Learning](https://ieeexplore.ieee.org/document/9465640).

# Abstract

A unique cognitive capability of humans consists in their ability to acquire new knowledge and 
skills from a sequence of experiences. Meanwhile, artificial intelligence systems are good
at learning only the last given task without any ability of remembering databases learnt 
in the past. We propose a novel lifelong learning methodology by employing a Teacher-Student
network framework. While the Student module is trained with a new given database, the Teacher 
module would remind the Student about the information learnt in the past. The Teacher, implemented 
by a Generative Adversarial Network (GAN), is trained to preserve and replay past knowledge corresponding 
to the probabilistic representations of previously learn databases. Meanwhile, the Student module is 
implemented by a Variational Autoencoder (VAE) which infers its latent variable representation from both
the output of the Teacher module as well as from the newly available database. Moreover, the Student module
is trained to capture both continuous and discrete underlying data representations across different domains. 
The proposed lifelong learning framework is applied in  supervised, semi-supervised and unsupervised training.


# Environment

1. Tensorflow 1.5
2. Python 3.6


## Training

To train the model(s) in the paper, run this command:

```train
python TeacherStudent_xxx.py
```

# BibTeX

@ARTICLE{9465640,  author={Ye, Fei and Bors, Adrian},  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   title={Lifelong Teacher-Student Network Learning},   year={2021},  volume={},  number={},  pages={1-1},  doi={10.1109/TPAMI.2021.3092677}}

# The network architecture of Lifelong Teacher-Student Network

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/mainStructure.png)

# Learning new concepts without forgetting

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t1.png)
![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t2.png)

# Learning on the complicated tasks

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t3.png)

# Capture both the shared and task-specific information

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t4.png)

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t5.png)

# Learning disentangled representation across domain under lifelong learning

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t6.png)

![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t7.png)
![image](https://github.com/dtuzi123/Lifelong-Teacher-Student-Network-Learning/blob/main/t8.png)





