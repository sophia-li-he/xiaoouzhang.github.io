---
layout: post
title: Deep Learning + Collaborative Filtering for Recommender Systems
---
## Introduction

This post is a follow up discussion about [an earlier work](https://github.com/xiaoouzhang/Collaborative-Deep-Learning-for-Recommender-Systems) about recommender systems using denoised autoencoder. I will focus on how to use the model to interpret the patterns of the dataset.

## What Can We Get from the Dataset
The key information extracted from the dataset is the rating matrix and the side information of users. The size of the rating matrix is (user number) $$\times$$ (product number). Each matrix element is the number of months that each user has used each product from 2015-01-28 to 2016-03-28. The record from 2016-03-28 to 2016-03-28 is used for validation and testing process. Note that we can build a recommender system only using the rating matrix by collaborative filtering (specifcally, matrix factoriztion).

Besides, the user information is extracted to enhance the performance of collaborative filtering for new users. Both categorical (gender, nationality, etc.) and numerical (age, income, etc.) variables exist in the dataset. The categorical variables are represented by one-hot encodings, and the numerical variables can be binned and become categorical variables. The one-hot encodings are concatenated into a 314-dimensional vector as the input of the denoising autoencoder.


## Model Description

![an image alt text]({{ site.baseurl }}/images/rs/AE.png "an image title")

A hybrid model is applied combining the stacked denoising autoencoder (SDAE) with matrix factorization (MF) algorithm. Here [(Wang et al. (2015))](https://dl.acm.org/citation.cfm?id=2783273) is a good article on this model.

The structure of a typical SDAE is shown in the upper part of the figure above. The one-hot encoding in the input layer is followed by a corruption layer, where a Gaussian noise is added to the input.  X0 is the original user information using one-hot encoding, and Xc is the corrupted input with Gaussian noise. Note that we use tied weight in the SDAE, such that the SDAE has a symmetric structure. The Xencode layer with the least number of hidden units is the encoding of the user information, which is also called “bottleneck” in this post.
