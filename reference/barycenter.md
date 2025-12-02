# Barycenter algorithm

Barycenter algorithm to solve for entropy-regularized Optimal Transport
Barycenter problems. For a more detailed explaination, please refer to
[`vignette("barycenter")`](https://fangzhou-xie.github.io/rwig/articles/barycenter.md).

## Usage

``` r
barycenter(
  A,
  C,
  w,
  b_ext = NULL,
  barycenter_control = list(reg = 0.1, with_grad = FALSE, n_threads = 0, method = "auto",
    threshold = 0.1, max_iter = 1000, zero_tol = 1e-06, verbose = 0)
)
```

## Arguments

- A:

  numeric matrix, source discrete densities (M \* S)

- C:

  numeric matrix, cost matrix between source and target (M \* N)

- w:

  numeric vector, weight vector (S)

- b_ext:

  numeric vector, only used to calculate quadratic loss against the
  computed barycenter (default = NULL)

- barycenter_control:

  list, control parameters for the computation

  - reg: double, regularization parameter (default = .1)

  - with_grad: bool, whether to calculate the gradient w.r.t. A

  - n_threads: int, number of threads (only used for `method = "log"`,
    ignored by `method = "parallel"`, default = 0)

  - threshold: double, threshold value below which "auto" method will
    default to `method = "log"` for stablized computation in log-domain
    (default = .1)

  - max_iter: int, maximum iteration of `barycenter` algorithm (default
    = 1000)

  - zero_tol: double, determine covergence (default = 1e-6)

  - verbose: int, print out debug info for the algorithm for every
    `verbose` iteration (default to 0, i.e. not printing anything)

## Value

list of results

- b: numeric vector, computed barycenter

- grad_A: gradient w.r.t. A (only with `with_grad = TRUE`)

- grad_w: gradient w.r.t. w (only with `with_grad = TRUE`)

- loss: double, quadratic loss between `b` and `b_ext` (only with
  `with_grad = TRUE`)

- U, V: scaling variables for the Sinkhorn algorithm (only with
  `method = "parallel"`)

- F, G: scaling variables for the Sinkhorn algorithm (only with
  `method = "log"`)

- iter: iterations of the algorithm

- err: condition for convergence

- return_status: 0 (convergence), 1 (max iteration reached), 2 (other)

## Details

This is the general function to solve OT Barycenter problem, and it will
use either parallel (`method = "parallel"`) or log-stablized Barycenter
algorithm (`method = "log"`) for solving the problem.

## References

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. https://doi.org/10.1561/2200000073

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
https://doi.org/10.48550/arXiv.2504.08722

## See also

[`vignette("gradient")`](https://fangzhou-xie.github.io/rwig/articles/gradient.md),
[`vignette("threading")`](https://fangzhou-xie.github.io/rwig/articles/threading.md)

## Examples

``` r
A <- rbind(c(.3, .2), c(.2, .1), c(.1, .2), c(.1, .1), c(.3, .4))
C <- rbind(
  c(.1, .2, .3, .4, .5),
  c(.2, .3, .4, .3, .2),
  c(.4, .3, .2, .1, .2),
  c(.3, .2, .1, .2, .5),
  c(.5, .5, .4, .0, .2)
)
w <- c(.4, .6)
b <- c(.2, .2, .2, .2, .2)
reg <- .1

# simple barycenter example
sol <- barycenter(A, C, w, barycenter_control = list(reg = reg))
#> `method` is automatically switched to "log"
#> Forward pass:
#> iter: 1, err: 0.3207, last speed: 0.015, avg speed: 0.015
#> iter: 11, err: 0.0020, last speed: 0.000, avg speed: 0.001
#> iter: 21, err: 0.0000, last speed: 0.000, avg speed: 0.001

# you can also supply arguments to control the computation
# for example, including the loss and gradient w.r.t. `A`
sol <- barycenter(A, C, w, b, barycenter_control = list(reg = reg, with_grad = TRUE))
#> `method` is automatically switched to "log"
#> Forward pass:
#> iter: 1, err: 0.3207, last speed: 0.000, avg speed: 0.000
#> iter: 11, err: 0.0020, last speed: 0.000, avg speed: 0.000
#> iter: 21, err: 0.0000, last speed: 0.000, avg speed: 0.000
#> Backward pass:
```
