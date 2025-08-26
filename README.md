# CvT - Convolution Vision Transformer
This is the implementation of CvT model architecture introduced by microsoft in research paper : "CvT : Introducing Convolution to vision transformers"

The architecture consist of 3 stages:

a. Convolution Token Embedding : Instead of positional encoding like traditiona transformer , this model uses convolution embedding to capture local spatial features
  
b. Convolution Attention layer : In this Q,K,V are convolutional projected instead of linear projections
  
c. Transformer Block - It implements the multi-head attention layers like traditional transformer for capturing global dependencies

The Final fucntion builds all the transformer block all together building all the layers of model.

I do not claim any copyright of this.
