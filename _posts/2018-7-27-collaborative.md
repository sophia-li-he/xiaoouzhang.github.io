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

The figure above shows the hybrid model combining the stacked denoising autoencoder (SDAE) with matrix factorization (MF) algorithm. Here [(Wang et al. (2015))](https://dl.acm.org/citation.cfm?id=2783273) is a good article on this model.

The structure of a typical SDAE is shown in the upper part of the figure above. The one-hot encoding $$X_0$$ in the input layer is followed by a corruption layer $$X_c$$, where a Gaussian noise is added to the input. Note that we use tied weight in the SDAE, such that the SDAE has a symmetric structure. The $$X_{encode}$$ layer with the least number of hidden units is the encoding of the user information and will be fed into the MF algorithm.

In the MF algorithm, we need to handle the rating matrix $$r_{ij}$$ containing the user behavior history, which are implicit feedbacks. This is different from the explicit feedbacks such as the rating in Amazon and Netflix, as discussed in [Hu et al. (2008)](https://dl.acm.org/citation.cfm?id=1510528.1511352).

For implicit feed backs, we can construct preference $$p_{ij}$$ and confidence $$c_{ij}$$ according to the rating matrix. If a customer has used a certain service before, it is possible that the user has preference to this service. Accordingly, the definition of preference $$p_{ij}$$ is given by
\begin{equation}
p_{ij}=\begin{cases}
1 & \text{if }r_{ij}>0\\\
0 & \text{otherwise}
\end{cases}\;.
\end{equation}
On the other hand, we are not completely sure about the preference. Therefore, we need to define the confidence
\begin{equation}
c_{ij}=1+\alpha r_{ij}\;,
\end{equation}
where $$\alpha$$ describes how the confidence grows with the history of using the service $$j$$. 

The MF algorithm for implicit feed back is applied in the following way. We define user matrix $$U_{i,:}$$ and item matrix $$V_{j,:}$$, where each row, written as $$\mathbf{u}_i$$ and $$\\mathbf{v}_j$$, is the vector in the latent factor representation for each customer and service, respectively. We predict the preference $$p_{i,j}$$ by $$\mathbf{u}_i\cdot\mathbf{v}_j$$. The preferences with different levels of confidence are not treated equally in the loss function, which is given by:
\begin{equation}
l=\sum_{i,j}c_{i,j}(p_{i,j}-\mathbf{u}_i\cdot\mathbf{v}_j)^2+\lambda(\sum_{i}||\mathbf{u}_i||^2+\sum_{j}||\mathbf{v}_j||^2)\;.
\end{equation}
From a probability point of view, the confidence $$c_{i,j}$$ measures the standard deviation of the prediction to the preference.

The update rule for is given by setting the derivivatives of the loss funciton with respect to $$\mathbf{u}_i$$ and $$\mathbf{v}_j$$ to zero:
\begin{equation}
U_{i,:}\rightarrow P^iC^iV(\lambda I+V^TC^iV)^{-1}\;,
\end{equation}
where $$C^i_{jj}=c_{ij}$$ is a diagonal matrix, and $$P^i=\{p_{ij}\}_\text{all j}$$ is the preference vector for customer $$i$$;
\begin{equation}
V_{i,:}\rightarrow \tilde{P}^j\tilde{C}^jU(\lambda I+U^TC^jU)^{-1}\;,
\end{equation}
where $$\tilde{C}^j_{ii}=c_{ij}$$ is a diagonal matrix, and $$\tilde{P}^j=\{p_{ij}\}_\text{all i}$$ is the preference vector for service $$j$$.

Each time after the $U$ and $V$ matrices are updated, we also update the parameters in the SDAE using gradient decent, such that collaborate filtering not only receive the prediction from the SDAE, but it also provide feedbacks to the SDAE.

\begin{equation}
l=\sum_{i,j}c_{i,j}(p_{i,j}-\mathbf{u}_i\cdot\mathbf{v}_j)^2
\end{equation}
