# Lifelong-Teacher-Student-Network-Learning
The implementation of Lifelong Teacher-Student Network Learning

Title : Lifelong Teacher-Student Network Learning

# Paper link



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


# The network architecture of Lifelong Teacher-Student Network



