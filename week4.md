### Goals for this lecture
 - Move from binary to multi-class classification

###  Dealing with multiple classes

Encode the class label as a vector with a single 1 at the index corresponding
to the category, and 0s elsewhere. This is called the one-hot encoding, or 
1-of-C encoding.

Predicting such a one-hot encoding is then similar to making the model output
multiple values.

We therefore use a matrix of parameters

$$
\hat{y}_i = W^Tx_i = \begin{pmatrix} w_{(1)}^T \\ w_{(2)}^T \\ \vdots 
\\ w_{(N)}^T \end{pmatrix} 
$$

We can then rewrite the training as:
$$
\underset{min}{W} \sum_{i=1}^{N} \| \ldots \|
$$


We can again group the inputs and outpus in two matrices as
$$
X = \begin{pmatrix}  x_1^T \\ x_2^T \\ \vdots\\x_N^T \end{pmatrix} 
$$
$$
Y = \begin{pmatriy}  y_1^T \\ y_2^T \\ \vdots\\y_N^T \end{pmatriy} 
$$

Given the optimal parameters, the prediction for a new sample is computed as
$$
\hat{y} = (W^*)^Tx
$$
The sample is then assigned to a class $k$ if
$$
\hat{y}^{(k)} > \hat{y}^{(j)} \forall j \neq k
$$

or
$$
k = argmax_j\hat{y}^{(j)}
$$

### Decision boundaries

In the binary case, the point where the predicted label changes from 0 to 1 
forms a decision boundary (which is a hyperplane).

The decision boundary between two classes $k$ and $j$ is defined as the set
of points $x$ such that $\hat{y}^{(k)}(x) = \hat{y}^{(j)}(x)$


The probability for a class k is given by the softmax function
$$
\hat{y}^{(k)}(x) = \frac{\exp (w_{(k)}^Tx)}{\sum_{j=1}^{C} \exp (w^T_{(j)}x}
$$

The empirical risk can then be derived from the multi-class version of the cross
entropy, which yields
$$
R(W) = -\sum_{i=1}^{N} \sum_{k=1}^{C} y_i^{(k)}\ln \hat{y}_i^{(k)}
$$

As in the binary case, the training is done by minimizing the loss using a 
gradient descent

With multiple classes, the parameters are shaped as a matrix $W \in \mathbb{R}^{(D+1)\times C}$,
not as a vector anymore. This does not affect the algorithm, it just means the
gradient itself will also be a matrix
$$
\nabla R(W_{k-1}) \in \mathbb{R}^{(D+1)xC}
$$
So the whole algorithm works with matrix entities.

By using the chain rule and the gradient of the softmax function, we can derive 
the gradient of the multi-class cross-entropy
$$
\nabla R(W) = \sum_{i=1}^{N} x_i(\hat{y}_i - y_i)^T
$$
The gradient follows the same intuition as the binary case.

At test time, given a new input sample $x$, the proability for any class $k$ is
computed via the softmax function as
$$
\hat{y}^{(k)}(x) = \frac{\exp (w_{(k)}^Tx)}{\sum_{j=1}^{C} \exp (w^T_{(j)}x}
$$
and the final label is then predicted as $k = argmax_j\hat{y}^{(j)}(x)$.

### Evaluation methods for multi-class classification

#### Confusion matrix

#### Accuracy
Summing the correct predictions for all classes, and dividing by the total number
of samples

#### Precision, recall, Fp-rate and F1-score
Compute them for each individual class, and then average over the classes.
