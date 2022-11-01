## Week  8
[toc]
### The Kernel as a similarity function
The solution to may ML algorithms ends up relying on a dot product between inputs, since $\overline{x}_i^T\overline{x}_j \propto \cos \angle(\overline{x}_i,\overline{x}_j)$, which helps encode a notion of similarity between two samples.

Sometimes, however, we want to compare different features, like when we do $\phi(\overline{x}_i)^T\phi(\overline{x}_j)$.

This we can generalize in a similarity function $k(\overline{x}_i,\overline{x}_j)$, which we call 
**kernel function**, and is characterized for being some sort of $\phi(\overline{x}_i)^T\phi(\overline{x}_j)$ where $\phi$ is unknown.

#### Linear Kernel
This is the typical cross product we've been using all along
$$
k(\overline{x}_i, \overline{x}_j) = \left<\overline{x}_i, \overline{x}_j\right>
$$
It has no hyper-parameters

#### Polynomial Kernel
This is the polynomial kernel
$$
k(\overline{x}_i,\overline{x}_j) = (\overline{x}_i^T\overline{x}_j + c)^{d}
$$
It has two hyper-parameters:
 * The bias $c$ (often set to 1)
 * The degree of the polynomial $d$ (often set to 2)

#### Gaussian Kernel
The most common kernel, the Gaussian (or **R**adian **B**asis **F**unction)
kernel
$$
k(\overline{x}_i,\overline{x}_j) = \exp \left(\frac{\|\overline{x}_i - \overline{x}_j\|^2}{2\sigma ^2}\right)
$$
It has one hyper-parameter:
 * The bandwidth $\sigma$ (which is data-dependent)

### Kernelizing
The act of changing a dot product for a kernel on an algorithms is referred as 
_kernelizing_ the algorithm.

For that, we use the **Representer theorem**

##### Representer theorem
The minimizer of a regularized empirical risk function can be represented as a 
linear combination of the points in the high dimensional space. That is, we can
write
$$
w^* = \sum_{i=1}^{N} \alpha_i^* \phi(\overline{x}_i) = \Phi^T\alpha^*
$$

Which lets us rewrite the empirical risk in terms of the variables $\{\alpha _i\}$

### Kernel regression
$$
R(w) = \sum_{i=1}^{N} (w^T\phi (\overline{x}_i)-y_i)^2
$$
Can be re-written as 
$$
R(\alpha ) = \sum_{i=1}^{N} (\alpha ^T\Phi\phi (\overline{x}_i) - y_i)^2 = \quad \\ 
 = \sum_{i=1}^{N} (\alpha ^Tk(X,\overline{x}_i - y_i)^2
$$
And the gradient is given by
$$
\nabla R(\alpha ) = 2 \sum_{i=1}^{N} k(X,\overline{x}_i)(k(X,\overline{x}_i)^T\alpha -y_i)
$$

Where we can generalize as in linear regression into the following expression
$$
\nabla _\alpha R(\alpha ) = 2KK\alpha - 2Ky
$$
where $K$ is the $N\times N$ kernel matrix, obtained by evaluating the kernel 
function between all pairs of training samples.
$$
2KK\alpha ^* - 2Ky = 0 \iff K\alpha^* = y
$$
Therefore, $a^* = (K)^{-1}y$ and
$$
\hat{y} = (w^*)^T \phi (\overline{x}) = y^T(K)^{-1}\Phi\phi(\overline{x}) = y^T(K)^{-1}k(X,\overline{x})
$$

> **NOTE** The quantity $y^T(K)^{-1}$ only depend on the training data

If we add a regularizer, like $\lambda w^Tw$, we can (using the Representer's
theorem) rewrite it as
$$
E(\alpha ) = \sum_{i=1}^{N} (w^T\phi (\overline{x}_i) - y_i)^2 + \lambda \|w\|^2 = \\
\quad\quad\quad = \sum_{i=1}^{N} (\alpha ^Tk(X,\overline{x}_i) - y_i)^2 + \lambda \alpha ^TK\alpha \\
\\ \ \\
\nabla _\alpha E(\alpha ) = 2K(K+\lambda I_N)\alpha -2Ky\quad\quad\quad\quad\ \ \ 
$$

Where the solution is
$$
\alpha^* = (K+\lambda I_N)^{-1}y
$$

##### Multiple outputs
To handle multiple outputs, we can again work is a different feature space
$\phi(\;\cdot\;)$
and use the Representer theorem to derive a solution that does not explicitely
depend  on $\phi ( \;\cdot\; )$ but in $k(x_i,x_j)$.
$$
W^* = \Phi^T(K)^{-1}Y
$$
The prediction becomes
$$
\hat{y} = (W^*)^T\phi (\overline{x}) = Y^T(K)^{-1}\Phi\phi(\overline{x}) = Y^T(K)^{-1}k(\overline{X},\overline{x})
$$

### Kernelizing SVM
The problem is
$$
\underset{\{\alpha _i\}}{max}\sum_{i=1}^{N} \alpha _i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha _i\alpha _jy_iy_j\phi(\overline{x}_i)^T\phi(\overline{x}_j)
$$
subject to $\sum_{i=1}^{N} \alpha _iy_i = 0$ for $\forall i,\ 0\le \alpha _i \le  C$
which already shows the product we need to replace to get:
$$
\underset{\{\alpha _i\}}{max}\sum_{i=1}^{N} \alpha _i - \frac{1}{2}\sum_{i=1}^{N} \sum_{j=1}^{N} \alpha _i\alpha _jy_iy_jk(\overline{x}_i,\;\overline{x}_j)
$$
subject to $\sum_{i=1}^{N} \alpha _iy_i = 0$ for $\forall i,\ 0\le \alpha _i \le  C$
where $\tilde{w}^* = \sum_{i=1}^{N} \alpha_i^*y_i\phi(\overline{x}_i)$ and 
$$
\hat{y}(\overline{x}) = (\tilde{w}^*)^T\phi(\overline{x}) + w^{(0)*} = \\
\quad\quad\quad\quad\quad\quad= \sum_{i=1}^{N} \alpha _i^*y_i\phi(\overline{x}_i)^T\phi(x) + w^{(0)*} = \\
\quad\quad\quad\ = \sum_{i=1}^{N} \alpha _i^*y_ik(\overline{x}_i,\overline{x}) + w^{(0)*}
$$
But since on all samples that are not support vectors we have $\alpha^*_i= 0$, we can perform the computation only in the set of support vectors $\mathit{S}$


