# Week 2
## Recap
### What data?
 * Data
 * Sound
 * Images
### What insight?
 * Category
 * Description
 * Vector-valued quantity
 * Different representation
### What algorithms?
 * Supervised learning
 * Unsupervised learning
 * Reinforcement learning _(Not talked about in this course)_
### A simple parametric model
Defined by 2 parameters:
 * The y-intercept $w^{(0)}$
 * The slope $w^{(1)}$

Given $\{(x_i,y_i\}$ of noisy measurements, find the line
that best fits the data points.

In essence, it consists of getting the best line parameters
$w^{(1)*}$ and $w^{(0)*}$ for the given data.

This corresponds to the training stage.

Typically for error, we used the squared euclidean distance.
$$
d²(\hat{y}_i,y_i) = (\hat{y}_i - y_i)^2
$$

Training can be expressed as the least-squares minimization problem
$$
min \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2
$$

## Today's lecture
### Minimizing the risk
Minimizing risk involves computing derivatives. Therefore, let's review the 
notion of derivatives/gradients.

#### Derivatives and gradients
Given an $R$ function of one function $w$, the derivative of $R(w)$ is the rate
at which $R$ changes as $w$ changes

It is measured for an infinitesimal change in $w$, starting from a point 
$\overline{w}$

The derivative at $\overline{w}$ is the tangent at that point.

The derivative banishes at stationary points:
 * Minima
 * Maxima
 * Saddle points
 
To minimize a function, we try to find the point:
$$
\frac{dR(w^*)}{dw} = 0
$$
For some simple, convex functions, $w^*$ can be obtained algebraically.

For some other functions, non-convex, there are multiple local minima, and they
are hard to find without first trying all of them.

#### Properties of derivatives
 1. The derivative of a sum is the sum of the derivatives  
   Useful because we are trying to find an average
   $$
   \frac{dr(\overline{w})}{dw} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2
   $$

#### Gradient
Formally, the gradient is denoted as:
$$
\nabla R(w) = \begin{pmatrix} \frac{dR_1(w)}{dw}\\ \vdots\\ \frac{dR_m(w)}{dw} \end{pmatrix}
$$

The gradient vanishes at the stationary points of the function

TO minimize a function, we seek $w^*$ were
$$
\nabla R(w^*) = 0
$$

### Back to linear regression
$$
R = \frac{1}{N }\sum_{i=1}^{N} d*2(\hat{y}_i,y_i) = \frac{1}{N}\sum_{i=1}^{N} (w^{(1)}x_i + w^{(0)}-y_i)^2

$$
This gives us the partial derivatives:
$$
\frac{dR}{dw^{(0)}} = \frac{2}{N}\sum_{i=1}^{N} (w^{(1)}x_i + w^{(0)} - y_i)
$$
$$
\frac{dR}{dw^{(1)}} = \frac{2}{N}\sum_{i=1}^{N} (w^{(1)}x_i + w^{(0)} - y_i)\cdot x_i
$$
We would like to find $w^{(1)}$ and $w^{(0)}$ such that this derivatives become 0

To do this it's easier to group the parameters $w^{(1)}$ and $w^{(0)}$ into a 
single parameter vector $w \in \mathbb{R}^2$. Then:
$$
R = \frac{1}{N }\sum_{i=1}^{N} w^T \begin{pmatrix} x_i\\ 1 \end{pmatrix} \ldots
$$

To compute the gradient of this empirical risk, we can use the general vector
derivative rules:
    ...

Use Matrix cookbook if unsure

With our empirical risk
$$
R = \frac{1}{N}\sum_{i=1}^{N} (x_i^Tw-y_i)²
$$

We can use the chain rule:
$$
\nabla _w R = \frac{1}{n}\sum_{i=1}^{N} \frac{d(x_i^Tw-y_i)^2}{d(x_i^Tw-y_i)} \cdot 
\frac{d(x_i^Tw-y_i)}{dw} = \frac{2}{N}\sum_{i=1}^{N} x_i(x_i^Tw - y_i)
$$

We want
$$
\nabla _wR = 0
$$
this means
$$
(\sum_{i=1}^{N} x_ix_i^T)w^* = \sum_{i=1}^{N} x_iy_i
$$

Let us now group all $\{x_i\} $ and $\{y_i\} $ as matrices
$$
X = \begin{bmatrix} x_1^T\\ \vdots\\ x_N^T \end{bmatrix} =
\begin{bmatrix} x_1 & 1 \\ \vdots & \vdots \\ x_N & 1 \end{bmatrix} \in \mathbb{R}^{N\times_2}
$$
$$
y = \begin{bmatrix} y_1 \\ \vdots\\ y_N \end{bmatrix} \in \mathbb{R}^N
$$

Then, we can re-write the solution as
$$
X^TXw^* = X^T  y
$$
This finally gives us
$$
w^* = (X^TX)^{-1}X^T y = X^† y
$$
were $X^†$ is the Moore-Penrose pseudo-inverse of X

### Testing phase
Once we have $w^*$, we can predict $\hat{y}_t$ for any new $x_t$.

The predicted value is given by
$$
\hat{y}_t = \begin{bmatrix} x_t\\ 1\end{bmatrix}w^* = (w^*)^T \begin{bmatrix} x_t \\ 1 \end{bmatrix} 
$$

> #### Exercise
> ##### Prompt
> From the trend data given
> ```
> |                                                                                             
> |    X                                                                                        
> |    X                                                                                        
> |      X    X                                                                                 
> |               X    X                                                                        
> |                       X    X                                                                
> |                             X       X                                                       
> |                        X                  X                                                 
> |                                X                                                            
> |                                        X    X  X   X       X   X                            
> |                                                         X         X   X                     
> |                                                                         X                   
> |                                                                            X                
> |                                                                                 X    X X    
> +---------------------------------------------------------------------------------------------
> ```
> Try to predict the values of $X$ and $y$
> ##### Solution

In general, an input observation $x_i$ is not represented by a single value:
 * A greyscale image can be represented as a $W\cdot H$ dimensional vector

When there are 2 input dimensions, instead of a line, we can define a plane.
Mathematically
$$
y = w^{(0)} + w^{(1)}x^{(1)}  + w^{(2)}x^{(2)}  = w^T \begin{bmatrix} x^{(1)} \\ x^{(2)} \\ 1 \end{bmatrix} 
$$
#### Plane fitting
Given $N$ noisy pairs $\{(x_i,y_i\}$ with $x_i \in \mathbb{R}^2$, find the plane
that best fits these observations

This can all be generalized to higher dimensions:
$$
y = w^{(0)} + w^{(1)}x^{(1)} + w^{(2)}x^{(2)} + \ldots + w^{(D)}x^{(D)} =
    w^T \begin{bmatrix} x^{(1)} \\ x^{(2)} \\ \vdots \\ x^{(D)} \\ 1 \end{bmatrix} 
$$
Note that
$$
w = \begin{bmatrix} w^{(1)} \\ \vdots \\ w^{(D)} \\ w^{(0)} \end{bmatrix} 
$$

Ultimately, whatever the dimension we write
$$
y = w^Tx
$$
Because the output remains 1D, we can use the same least-square loss function
$$
min(w) \sum_{i=1}^{N} d^2(\hat{y}_i,i) \iff min(w) \sum_{i=1}^{N} (x_i^Tw-y_i)^2
$$

Note that this has the same form as the 1D input case when we grouped $w^{(0)}$
and $w^{(1)}$ in a single vector. Therefore, the solution obtained earlier by 
zeroing out the gradient is exactly the same as before
$$
w^* = (X^TX)^{-1}X^Ty = X^†y
$$
but now, the matrix $X \in \mathbb{R}^{N \times (D+1)}$, instead of 
$\mathbb{R}^{N\times 2}$

You can play with linear regression in this [demo](https://playground.tensorflow.org/#activation=linear&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=&seed=0.45772&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=regression&initZero=false&hideText=false)
### Interlude: Interpreting a linear model
How can we interpret the results?
![](/tmp/evince-251626/image.444DA1.png)
One can then look at the coefficient values to see the influence of each
attribute

When interpreting a linear model, the magnitude of the coefficient will depend
on the magnitude of the corresponding feature/attribute:
 * A coefficient might be very small simple to compensate for the fact that the
   range of the feature is very large

This can be addressed by normalizing the data

### Dealing with multiple output dimensions
We've assumed that the output was a single value $y_I \in \mathbb{R}$. In practice,
one may want to output multiple values for a given input $y_i\in \mathbb{R}^C$ 
(where $C > 1$)

To output multiple values, the linear model cannot just rely on a vector $w$.
We need a matrix $W \in \mathbb{R}^{(D+1)\times C}$, such that
$$
\hat{y}_i = W^Tx_i = \begin{bmatrix} w^T_{(1)} \\ \vdots \\ w^T_{(C)} \end{bmatrix}x_i
$$

Where each $w_{(j)}$ is a (D+1)-dimensional vector

We need to modify the loss function
$$
min(w)\sum_{i=1}^{N} \|W^Tx_i-y_i\|^2
$$
where $\|a\| = \sqrt{\sum_{j=1}^{C} (a^{(j)})^2}$ indicates the norm of a vector

Using:
 * $\|a\|^2 = a^Ta$, so $ \frac{\delta \|a\|^2}{\delta a} = 2a$
 * $\frac{\delta a^T B}{\delta B} = a$ (WRONG, see notes again)

We can use the chain rule replacing $(W^Tx_i-y_i)$ with $(x_i^TW-y_i^T)$) to get
the gradient:
$$
\nabla _WR = w \sum_{i=1}^{N} x_i(x_i^TW-y_i^T)
$$
We can again group the outputs in a matrix
$$
Y = \begin{bmatrix} y^T_1 \\ \vdots \\ y^T_C \end{bmatrix}  = 
\begin{bmatrix}
y_1^{(1)} & y_1^{(2)} & \ldots & y_1^{(C)} \\
y_2^{(1)} & y_2^{(2)} & \ldots & y_2^{(C)} \\
\vdots & \vdots & & \vdots \\
y_N^{(1)} & y_N^{(2)} & \ldots & y_N^{(C)} \\
\end{bmatrix} 
$$

> #### Exercise
> ##### Prompt
> The task is given the top half of a face, predict the bottom half.
> 
> Assuming that there are 35 training subjects with 10 images per subject, and
> that a complete face image is of size 28 × 28 pixels, what is the shape of
> the matrices X and Y? How many parameters are there to learn?
> ##### Solution
> Image = 28*28
> 10 images per person
> 35 training subjects
> The dimension of the input and output is 28x14
> The number of parameters to learn would be (28x14+1)x(28x14)
