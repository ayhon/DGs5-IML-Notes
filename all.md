# Machine Learning

**Machine learning:** The study of algorithms which seek to provide knowledge to computers through data, observations and interaction with the world. It is used to make accurate predictions given new observations.

In short, a machine learning consists on **algorithms** which take **data** and give an **insight** on that data:

* **data** which can be supervised or unsupervised
* **algorithms** such as:
  * Supervised learning: Data samples come with labels. Consists on two parts:
    * Training (With a training set)
    * Testing (With a testing set)
  * Unsupervised learning
  * Reinforcement learning
* **insight** which can be a regression (Predict continuous value) or classification (Predict one discrete label)

# Data

## Polynomial feature expansion

Problems impossible to treat linearly may become doable with a transformation of the data.

There is a balance of dimensions which must be achieved as neither too many nor too little are helpful.[$^{\color{orange}*}$](#overfitting-and-underfitting)

In general, for $D$-dimensional imports, we can obtain
$$
\phi(\overline{x}) = \left[\begin{matrix}
\prod_{k=1}^{D} x^{(k)\alpha_k} 
\end{matrix}\right]_{\alpha\in [0\ldots K]^D}
$$

## Kernel

A kernel is nothing but a function that provides a notion of similarity between
two samples. In the following examples, $x^Tx$ has been used as an implicit 
kernel. However, this can be generalized

* **Polynomial kernel**
  
  $$
  k(\overline{x}_i,\overline{x}_j) 
= \overline{x}_i^T\overline{x}_j + c)^d
  $$
  
  With hyper-parameters:
  * bias $c$
  * degree $d$
    For $\phi(\overline{x}) = \left[(x^{(1)})^2,(x^{(2)})^2, \sqrt{2}x^{(1)}x^{(2)},\sqrt{2}x^{(1)},\sqrt{2}x^{(2)},1 \right]^T$, we have the correspondence $\phi(\overline{x}_i^T)^T\phi(\overline{x}_j) = (x_i^{(1)}x_j^{(1)} + x^{(2)}_ix^{(2)}_j + 1)^2 = k(\overline{x}_i,\overline{x}_j)$
* **Gaussian kernel**, or **R**adical **B**asis **F**unction kernel
  
  $$
  k(\overline{x}_i,\overline{x}_j) 
= \exp \left( -\frac{\norm{\overline{x}_i - \overline{x}_j}^2}{2\sigma^2} \right) 
  $$
  
  With hyper-parameters:
  * bandwith $\sigma$, which is data dependent.

### Representer theorem

The _minimizer of a regularized empirical risk function_ can be represented as a _linear combination_ of the points in the higher dimensional space, so
$$
w^* = \sum_{i=1}^{N} \alpha _i^*\phi(\overline{x}_i) = \Phi^T\alpha^*
$$
where $\alpha^* = (\alpha_i^*)_{i=1}^N$

### Data representations

* Bag of words
* Histograms
* Image features / Bag of visual words
  
  ### Cleaning noise
* **Binning**
  Sort the data, partition it into _bins_ and take the mean or median of each bin as the new data.
* **Clustering**
  Detect and suppress outliers as the clusters with too few elements.

### Data normalization

* **min-max** $\tilde{x}^{(d)} = \frac{x_i^{(d)} - x_{\min}^{(d)}}{x_{\max}^{(d)} - x_{\min}^{(d)}}$
* **z-score** $\tilde{x}^{(d)} = \frac{x_i^{(d)} - \mu^{(d)}}{\sigma^{(d)}}$
* **max** $\tilde{x}^{(d)} = \frac{x_i^{(d)}}{x_{\max}^{(d)}}$
* **decimal scaling** $\tilde{x}^{(d)} = \frac{x_i^{(d)}}{10^k}$ where $k := \arg\underset{k \in \Z}{\min}\underset{i \in\ldots N}{\max} \abs{\frac{x_i^{(d)}}{10^k}} \le 1$

### Dealing with imbalanced data

The imbalance may reside within one class or even between classes

We use method which act on the data

* Oversampling randomly or by using data on neighbours _(SMOTE, AdaSyn)_
* Undersampling randomly or by using data on neighbours
  or on the cost function (Manually adding a weight $\beta _i$ for each label $i$ proportional to each categories size)

### Deep learning tricks

* Pre-train your neural network
* Use learning rate scheduling:
  * Decrease the learning rate after a certain number of epochs (iterations)
  * Use a cyclic learning rate
* Data augmentation (Use elemental transformations on the given samples to augment the number of them. _For example, rotating or moving images_)

# Algorithms (Models)

## Linear model

For regression and classification

### Prediction

$$
\hat{y} = w^T\overline{x_i}
$$

### Loss and risk function

$$
l(\hat{y}_i,y_i) = \|w^T\overline{x_i} - y_i\|^2
$$

$$
R(w) = \frac{1}{N}\sum_{i=1}^N l(\hat{y}_i,y_i) \\
\text{or alternatively }
\begin{cases}
MSE(w)  &= \frac{1}{N}\sum_{i=1}^N \norm{\hat{y}_i-y_i} \\
RMSE(w) &= \frac{1}{N}\sqrt{MSE(w)} \\
MAE(w)  &= \frac{1}{N}\sum_{i=1}^N \abs{\hat{y}_i - y_i} \\
MAPE(w) &= \frac{1}{N}\sum_{i=1}^{N} \abs{\frac{\hat{y}_i - y_i}{y_i}}
\end{cases}
$$

### Finding w

$$
w^* = (X^TX)^{-1}X^Ty = X^†y
$$

With a regularizer and polynomial expansion,
$$
w^* = (\Phi^T \Phi + \lambda I_F)^{-1} \Phi^T y
$$
where $\Phi = \phi(X) \in Mat(N\times F,\R)$ and $\lambda$ is the hyperparameter of the regularizer.[$^{\color{orange}*}$](#penalizing-model-complexity)

Regularized linear regression is also referred to as **ridge regression**.

Thanks to the [representer theorem](#representer-theorem), we can introduce a 
kernel in the risk function
$$
\begin{align}
R(w) &= \sum_{i=1}^{N} (w^T\phi(\overline{x}_i) - y_i )^2 \\
 &= \sum_{i=1}^{N} (\alpha^T\Phi\phi(\overline{x}_i) - y_i )^2 \\
 &= \sum_{i=1}^{N} (\alpha^Tk(x,\overline{x}_i) - y_i )^2 \\
\end{align}
$$

And with $K := (k(\overline{x}_i,\overline{x}_j))_{i=1,j=1}^{N,N}$, we can deduce
the factors $\alpha^*$ as $K^{-1}y$, so $w^* = \Phi^TK^{-1}y$. With regularizers, $\alpha^* = (K + \lambda I_N)^{-1}y$

## Logistic regression

Mainly for classification

### Prediction

$$
\hat{y}_i = \sigma(w^T\overline{x}_i) = \frac{1}{1 + \exp(-w^T\overline{x}_i)}
$$

or, in case of multi-class classification, we can use the $\text{softmax}$ function instead: 
$$
\hat{y}^{(k)}_i = \frac{\exp(w_{(k)}^T\overline{x}_i)}{\sum_{j=1}^C \exp(w_{(j)}^T\overline{x}_i)}
$$

### Loss and risk function

$$
\begin{align}
l(\hat{y}_i,y_i) &= \log\left(p(\hat{y}_i = y_i | w)\right) = \\
 &= \log\left( \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i}  \right)=\\
 &= \left( y_i\log(\hat{y}_i) + (1-y_i)\log (1-\hat{y}_i) \right) 
\end{align}
$$

$$
\begin{align}
R(w) &= - \sum_{i=1}^{N} l(\hat{y}_i,y_i) =\\
 &= - \sum_{i\in \{i : y_i = 1\} } \log (\hat{y}_i)
- \sum_{i\in \{i : y_i = 0\} } \log (1-\hat{y}_i)
\end{align}
$$

or for multi-class classification
$$
R(W) = - \sum_{i=1}^{N} \sum_{k=1}^{C} y_i^{(k)}\ln \hat{y}_i^{(k)}
$$

### Finding w

Use gradient descent:

* **convergence criteria**:
  * Small enough improvements of $R$
  * Small enough changes in $w$
  * Fixed number of iterations

## Support Vector Machine

For classification

To use in multi-class classification, use the approaches:

* OAASVM (**O**ne **A**gainst **A**ll)
* OAOSVM (**O**ne **A**gainst **O**ne)

### Prediction

$$
\hat{y}_i = w^T\overline{x}_i = \sum_{i\in S} \alpha^*_iy_ix_i^Tx + w^{(0)*}
$$

where $\alpha_i$ are the dual coefficients, $\alpha_iy_ix_i^T$ are the primal coefficients and $S := \{i : y_i(w^T\overline{x}_i) = 1-\xi_i\}$ are the
support vectors.

The $\overline{x}_i^T\overline{x}$ can be substituted by a kernel $k(\overline{x}_i,\overline{x})$.

### Loss and risk function

$$
\text{HingeLoss}(\hat{y}_i,y_i) = \max(0, 1-y_i(w^T\overline{x}_i))
$$

### Finding w

$$
\underset{w,\{\xi_i\} }{\min}
\frac{1}{2}\norm{\tilde{w}}^2 +
C\sum_{i=1}^{N} \xi_i
$$

subject to $y_i(w^T\overline{x}_i)\ge 1-\xi_i, \xi_i \ge 0 \forall i = 1..N$,
where $C$ is a hyperparameter which sets the severity of using slack variables.

Equivalently using the **hinge loss** $\max(0, 1-y_i(w^T\overline{x}_i)$
$$
\underset{w}{\min}\frac{1}{2c}\norm{\tilde{w}}^2 +
\sum_{i=1}^{N} \max(0, 1-y_i(w^T\overline{x}_i))
$$
There is no need for a regularizer, since it's built in into the algorithm.

Kernelizing the algorithm, we get 
$$
\underset{\alpha}{\max}\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i\alpha_jy_iy_j\overline{x}_i^T\overline{x}_j = \\
\underset{\alpha}{\max}\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i\alpha_jy_iy_j\phi(\overline{x}_i)^T\phi(\overline{x}_j) = \\
\underset{\alpha}{\max}\sum_{i=1}^{N} \alpha_i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_i\alpha_jy_iy_jk(\overline{x}_i,\overline{x}_j) = 
$$
subject to $\sum_{i=1}^N \alpha _iy_i = 0$, $0\le \alpha_i \le C, \forall i$

## K-nearest neighbours

Depends on the distance used (Euclidean $d(x_i,x) = \sqrt{\sum_{d=1}^{D} (x_i^{(d)} - x^{(d)})^2}$, $\chi^2$ distance $d(x_i,x) = \sqrt{\sum_{d=1}^{D} \frac{(x_i^{(d)}-x^{(d)})^2}{x_i^{(d)}-x^{(d)}}}$) and the hyper-parameter $k$.

| Pros                                | Cons                                  |
| ----------------------------------- | ------------------------------------- |
| Simple method                       | Results depend on $k$                 |
| Effective to handle non-linear data | Becomes expensive as $N$ and $D$ grow |
| Only requires defining $k$          | May be unreliable with large $D$      |

## Neural networks

The idea is to use various layers of $w_{(i)}$, and interconnect them with activation functions (non-linear functions), so
$$
\begin{align}
y &= {\color{yellow} f_{(L)}(w^T_{(L)}z_{L}) }= \\
 &= f_{(L)}(w^T_{(L)} {\color{yellow} f_{(L-1)}(w^T_{(L-1)}z_{L-1})}) = \\
 &= f_{(L)}(w^T_{(L)} f_{(L-1)}(w^T_{(L-1)} · \ldots 
 {\color{yellow}f_{(2)}(w^T_{(2)}z_{2})}
\ldots )) = \\
 &= f_{(L)}(w^T_{(L)} f_{(L-1)}(w^T_{(L-1)} · \ldots 
f_{(2)}(w^T_{(2)}·
{\color{yellow}f_{(1)}(w^T_{(1)}x)}
)
\ldots ))
\end{align}
$$

We call $(z_{(l)})_{l = 2}^{L}$ the outputs of the hidden layers.

### Activation functions

* Sigmoid $f(a) = \frac{1}{1 + \exp(-a)}$
* Hyperbolic tangent $f(a) = \tanh (a)$
* **Re**ctified **L**inear **U**nit $f(a) = \begin{cases}
  a & \text{if }a>0\\
   0 & \text{otherwise}
   \end{cases}$
* Leaky ReLU, with $0.001x$ instead of $0$
* Parametric ReLU, with $\lambda x$ instead of 0
* **E**xponential **L**inear **U**nit $f(a) = \begin{cases}
  a & \text{if }a > 0\\
  \lambda (e^a - 1) & \text{otherwise}
   \end{cases}$

Usually _softmax_ is used for classification and _linear activation function_
for regression _(i.e., no activation function)_

### Finding $\{w_{(l)}\}$

We use gradient descent. The gradients can be computed efficiently using
the algorithm of backpropagation:

1. Propagate $\overline{x}_i$ forward through the network
2. Compute $\delta_{(L)} := \frac{\part l_i}{\part y^{(k)}}$ depending on the loss
3. Propagate $\delta_{(L)}$ backward to obtain $\{\delta_{(l)}\}$
4. At each layer, compute the partial derivatives $\frac{\part l_i}{\part w_{(l)}^{(k,j)}} = \delta_{(l)}^{(k)} z_{(l-1)}^{(j)}$, where $\{z_{(l)}\}$ where computed on step 1

Then, we can just do $W_k \leftarrow W_{k-1} - \mu \nabla R(W_{k-1})$. However, since
this is too expensive, usually we use the gradient of a single sample or a batch $B$ of samples in each iteration.
$$
W_k \leftarrow W_{k-1} - \mu \sum_{b=1}^B\nabla l_{i(b,k)}(W_{k-1})
$$
This is called **stochastic gradient descent**, abbreviated to SGD.

It can be extended by adding:

* momentum: $W \leftarrow W - v$ where $v \leftarrow \gamma v + \mu \nabla R(W)$
* Adaptive learning rate _(Adagrad, Adadelta, RMSprop, ADMA)_
* Glorot & Bengio initialization strategy

### Regularization of deep neural networks

We can use _Dropout_: randomly remove units and the connections during training.

## Convolutional neural networks

Neural networks which handle "locally-relevant" data, that is, data where its
less important each individual values, but how the values change across a small
group of related values _(For example, an image)_

### Convolutions

Primary method of identifying features

Applies a filter to each layer (to which we can add a global bias term).

Multiple filters can be used to achieve different channels (with the byproduct of
increasing dimensionality, see [pooling](#pooling) for decreasing dimensionality)

We use padding to center the convolution in the relevant data, and may use a
stride greater than one to speed up computation (That is, taking every $k$-th point
instead of going one by one)

### Pooling

Primary method of reducing dimensionality. Has no learnable parameters, consists
just of a filter for dimensionality reduction:

* Max pooling (Take the max of values)
* Average pooling (Take the average of values)

### Normalization

Normalization can also be carried over (seeing as it helped in previous models),
using one of the following:

* Batch normalization
* Layer normalization
* Instance normalization
* Group normalization
  👁

### Reversing the operations

We can say that each channel is trying to distinguish features in the original data. We can
therefore use values in the feature space to recreate a given data point. 

The idea is to invert the operations we've done up until down. We can use transpose 
convolutions for this.

## Dimensionality reduction

### Principal Component Analysis

New dimension is projection on space with highest variance in vectors (That is,
higher _sparsity_ in that dimension).

#### Computing reduced dimensionality

$$
\hat{x}_i = W^T(x_i - \overline{x})
$$

where $\overline{x} = \frac{1}{N}\sum_{i=1}^{N} x_i$ and $W^TW = 1_d$, for a
$d$-dimensional representation

#### Finding W

For a $d$-dimensional representation, choose the $d$ biggest eigenvalues of the
input covariance matrix 
$$
C := \frac{1}{N}\sum_{i=1}^{N} (x_i - \overline{x})(x_j \overline{x}) = \left( \frac{1}{N}\sum_{i=1}^{N} (x_i^{(a)} - \overline{x})(x_i^{(b)} \overline{x}) \right)^{N,N}_{a = 1, b = 1}
$$

### Kernelized Principal Component Analysis

Let's us handle non-linear data, but fails to provide a way to retrieve the
original dimension (Since it may be unknown or impossible to represent, like with
the RBF kernel)

#### Computing reduced dimensionality

$$
\hat{x}_i = \sum_{j=1}^{N} k(x_i,x_j)
$$

#### Finding W

For a $d$-dimensional representation, choose the $d$ biggest eigenvalues of the
following eigenvalue problem
$$
Ka = \lambda_k N a
$$
(which means, of the matrix $\frac{1}{N}K$, where $K := \left(k(x_i,x_j\right))^{N,N}_{i=1,j=1}$)

### Autoencoder

Use a neural network to find both an encoder and decoder to a lower dimensional
space.

#### Computing reduced dimensionality

$$
\hat{x} = W·f(W^Tx)
$$

or for multilayered, use $z_{(j)} = f(W^T_{(j)}z_{(j-1)})$

#### Finding ${W_{(j)}}$

Use SGD with the risk function
$$
R(\{W_{(j)}\}) = \frac{1}{N}\sum_{i=1}^{N} \norm{\hat{x}_i - x_i}^2
$$
where $\norm{\hat{x}_i - x_i}^2$ is know as the reconstruction error $e_i$.

We can also use $E_r(\{W_{(j)}\}) = \sum_{i=1}^{N} \sum_{j=1}^{M} \abs{y_i^{(k)}}$
as a regularizer

## Clustering

### K-means clustering

Given an hyperparameter K (the number of classes, assumed to be correct) and a
metric of closeness between two data points (euclidean metric, for example), assign
every point to the category it's closest to.

To find the best such categories, we can look for the centers. Start with some
randomly chosen centers and iteratively assign each sample to one center and
move the previous centers to the new centers of the categories they formed. This
process can be repeated, and it's guaranteed to converge (Though it may not be 
to the best solution)

To run the algorithm without needing to find k, we can iteratively apply the
algorithm to different values of k, and choose the one in which the error 
starts to decelerate.

### Fisher Lineal Discriminant Analysis

Clustering with dimensionality reduction

For $d$ dimensions, choose the d largest eigenvalues associated to the eigenvalue
problem
$$
S_B w_{(k)} = \lambda_k S_W w_{(k)}
$$

### Spectral clustering

The idea is to use connectivity instead of compactness (euclidean or $L^1$ measures) [$^{\color{yellow}*}$](https://www.geeksforgeeks.org/ml-spectral-clustering/)

# Insights

## Evaluation metrics

### Binomial classification

* Accuracy: $\frac{TP +TN}{P + N}$
* Precision: $\frac{TP}{TP + FP}$
* Recall: $\frac{TP}{P}$
* False positive rate: $\frac{FP}{N}$
* F1 score: $2\frac{\text{Precision}·\text{Recall}}{\text{Precision} + \text{Recall}}$
* AUC: Area under ROC (Receiver operating characteristic) curve

### Multi-class classification

* Confusion matrix
* Accuracy: $\frac{\sum ✔}{\sum✔ + \sum✘}$
* Precision: Average over doing it for $(C_i,\overline{C_i})$ for all categories
* Recall: Average over doing it for $(C_i,\overline{C_i})$ for all categories
* False positive rate: Average over doing it for $(C_i,\overline{C_i})$ for all categories
* F1 score: Average over doing it for $(C_i,\overline{C_i})$ for all categories

## Overfitting and Underfitting

Overfitting is the phenomenon of having a model too specialized for the training data.

Underfitting is the phenomenon of having a model still not trained enough.

### Cross validation

A method of model selection, to prevent under-fitting or over-fitting.

* **$k$-fold cross validation**  
  Partition the training data into $k$ parts. For each part $P_i$, train the model in $\overline{P_i}$ and calculate the loss on $P_i$. Then use the average of all errors.
* **validation set cross validation** train on part of your data and compute the loss on the other part. Like $2$-fold cross validation, but only training on one partition, without alternation.
* **leave-one-out cross validation** is exactly the same as $N$-fold cross validation.

### Penalizing model complexity

Add a **regularizer** to the risk function to train, which measures model complexity. The, we use $E$ as our new risk function, where

* $E(w) = R(w) + \lambda E_w(w)$.
* $E_w(w)$ is called the regularizer, measures model complexity. 
   _(i.e. $E_w(w) = \norm{w}^2$)_
* $\lambda$ is a hyperparameter which defines the regularizer's influence.

### Insights from convolutional neural networks
