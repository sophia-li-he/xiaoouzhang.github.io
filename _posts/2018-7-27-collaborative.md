---
layout: post
title: Collaborative Deep Learning for Recommender Systems
---
## Introduction

This post is a follow up discussion about [an earlier work](https://github.com/xiaoouzhang/Collaborative-Deep-Learning-for-Recommender-Systems) about recommender systems using denoised autoencoder. I will focus on how to use the model to interpret the patterns of the dataset.

## What Can We Get from the Dataset
The key information extracted from the dataset is the rating matrix and the side information of users. The size of the rating matrix is (user number $$\times$$ product number). Each matrix element is the number of months that each user have used each product from 2015-01-28 to 2016-03-28. The record from 2016-03-28 to 2016-03-28 is used for validation and testing process. Note that we can build a recommender system merely using the rating information by collaborative filtering.

Besides, the user information is extracted to enhance the performance of collaborative filtering for new users. The categorical variables are represented by one-hot encodings, and the numerical variables can be binned and become categorial variables. The one-hot encodings are concatenated into a 314 dimensional vector as the input of the denoising autoencoder.


## Model Description

