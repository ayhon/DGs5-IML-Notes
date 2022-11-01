[toc]

### From regression to classification
Predict one discrete label for a given sample

### Binary classification as regression
##### Example
Classify a tumor as benign vs malignant. (1D input, 1D output)

We can use a linear model to fit the data, but then the output we'll get won't
be just 0 or 1.

We can achieve this by making a threshold on the model.
$$
\text{label} = \begin{cases}
1 & \text{if }\hat{y} \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$
This forms a **decision boundary**
 * 1D, a point
 * 2D, a line

However, using the least-square binary classification can fail if one of the
samples to classify is too far away from the decision boundary.

This is because we are working with non-linear data.

We would like to add the notion of non-linearity to our model.

$$
\text{y} = \begin{cases}
1 & \text{if }w^Tx \ge 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

The problem with the previous function is that it's hard to optimize. We could
use the Perceptron algorithm, but we won't use that.

The most used alternative is approximating the step function with a sigmoid
$$
f(a) = \frac{1}{1+\exp(-a)}
$$
With a 1D input $x$, applying the logistic function to the output of a linear
model gives a prediction
$$
\hat{y} = \frac{1}{1 + \exp (-w^{(1)}x - w^{(0)})}
$$
 * $w^{(0)}$ shifts the transition boundary
 * $w^{(1)}$ sharpens or softens the transition boundary

In the binary case, with $D$ dimensional inputs, we can write the prediction of
the model as
$$
\hat{y} = \sigma(w^Tx) = \frac{1}{1 + \exp (-w^Tx)}
$$
such a prediction can be interpreted as the probability that $x$ belongs to the
positive class (or as a score for the positive class)

The probability to belong to the negative class is therefore $1-\hat{y}$

#### Logistic regression
$$
\hat{y} = \frac{1}{1 + \exp (-w^{(1)}x - w^{(0)})}
$$
The name is misleading, as this is a model used for binary classification. At
its heart, logistic regression still makes use of the linear model, but followed
by a non-linearity to better encode the underlying non-linear data.

##### Training
We need a loss function. We need something better than the $L^2$ loss. We 
therefore use the probabilistic interpretation.

The probability of guessing $y$, the test labels, given $w$ can be written as:
$$
p(y|w) = \prod_{n=1}^{N} \hat{y}_i^{y_i}(1-\hat{y}_i)^{1-y_i} 
$$
where, again:
 * $\hat{y}_i$ is the prediction of the model
 * $y_i$ are values between 0 and 1 (The actual labels)

However, as a product of numbers is really expensive, and can disappear rapidly
when multiplying multiple small numbers, we use the logarithmic version as the
risk function.
$$
R(w) = - \sum_{i=1}^{N} (y_i \ln (\hat{y}_i) + (1-y_i)\ln (1-\hat{y}_i))
$$
Since the true $y_i$ is either 0 or 1, the cross entropy can be rewritten as:
$$
R(w) = -\sum_{i\in\text{positive samples}} \ln (\hat{y}_i) - \sum_{i\in\text{negative samples}} \ln (1-\hat{y}_i)
$$

We want to minimize this using [gradient descent](#iterative%2C-gradient-based-optimization)

The derivative of the logistic sigmoid function is:
$$
\frac{d \sigma(a)}{d a} = \sigma(a)\cdot (1-\sigma(a))
$$

From this we derive:
$$
\nabla \left( - \sum_{i=\text{positive}} \ln (\sigma(w^Tx_i))\right)  =
\sum_{i=\text{postive}}(\hat{y}_i-1)x_i
$$
$$
\nabla \left( - \sum_{i=\text{negative}} \ln (\sigma(w^Tx_i))\right)  =
\sum_{i=\text{negative}}\hat{y}_i x_i
$$

Which can be put together to get:
$$
\nabla R(w) = \sum_{i=1}^{N} (\hat{y}_i - y_i)x_i
$$
where $y_i$ is the true label for sample $i$.

> #### Exercise
> ##### Prompt
>  1. Describe two main differences between linear regression and logistic
>     regression.
>  2. Describe one similarity between linear regression and logistic regression.
> ##### Solution
>  1. One is used for regression, the other one for classification
>  2. The both depend on linear data.

### Iterative, gradient-based optimization

In some cases where we cannot compute the closed form of the solution, we can 
nevertheless use the derivative as a measure of where to move to find the 
minima.

The algorithms goes as follows:
 1. Initialize $w^{(0)}$ (i.e. randomly)
 2. While not converged:  
    Update $w_k$ ← $w_{k-1}-\eta \frac{d R(w_{k-1}}{d w}$  
    We call $\eta$ the learning rate

We say that it has converged if: 
 * Change in function values less than threshold:  
   $\mid R(w_{k-1}) - R(w_k)\mid < \delta  _R$
 * Change in parameter value less than a threshold:  
   $\mid w_{k-1} - w_k| < \delta  _w$
 * Maximum number of iterations reached (Doesn't guarantee to have reached minimum)

For non-convex functions, this method can be used, but it's more finicky. 

In practice, we have more than one parameter
$$
\nabla R(w) = \begin{pmatrix} 
\frac{\delta R}{\delta w^{(1)}} \\ \vdots \\ \frac{\delta R}{\delta w^{(D)}}
\end{pmatrix} 
$$
With multiple variables, the algorithms goes as follows:
 1. Initialize $w^{(0)}$ (i.e. randomly)
 2. While not converged:  
    Update $w_k$ ← $w_{k-1}-\eta \nabla R(w_{k-1})$  

### Model evaluation
We say TP is true positives, TN true negatives, FP and FN false positives and
negatives.
#### Classification metrics
##### Accuracy
$$
Acc = \frac{TP + TN}{P+N}
$$
The problem is it doesn't differentiate between strong or weak positive 
recognition rates. Because we mix TP and TN, we cannot differentiate, we cannot differentiate
##### Precision/Recall
$$
Precision = \frac{TP}{TP+FP}
$$
The problem with this model is that it doesn't take into account the negative
recognition rate.

##### ROC curve
Many classifiers output a score/confidence level for their predictions, so the
decision between positive and negative can be computed by thresholding this score,
that is:
$$
lable_i = 1 \text{ if ] \hat{y}_i \ge  \pi
$$
The **R**eceiver **O**perating **C**haracteristic curve plots the true positive
rate as a function of the false positive rate, obtained by varying the score 
threshold.

We can use the area under a ROC curve as a metric. The closer to 1, the better.
