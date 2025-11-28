# Sinkhorn algorithm

Sinkhorn algorithm to solve entropy-regularized Optimal Transport
problems.

## Usage

``` r
sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(reg = 0.1, with_grad = FALSE, method = "auto", threshold = 0.1,
    max_iter = 1000L, zero_tol = 1e-06, verbose = 0L)
)
```

## Arguments

- a:

  numeric vector, source discrete density (probability vector)

- b:

  numeric vector, target discrete density (probability vector)

- C:

  numeric matrix, cost matrix between source and target

- sinkhorn_control:

  list, control parameters for the computation

  - reg double, regularization parameter (default = .1)

  - with_grad: bool, whether to calculate the gradient w.r.t. a

  - n_threads: int, number of threads (only used for `method = "log"`,
    ignored by the `method = "vanilla"`, default = 0)

  - method: character, which method to use: "auto", "vanilla", "log"
    "auto" with try to calculate minimum value of the Gibbs kernel K and
    switch to `method = "log"` if the minimum value is less than
    `threshold` (default = "auto")

  - threshold: double, threshold value below which "auto" method will
    default to `method = "log"` for stablized computation in log-domain
    (default = .1)

  - max_iter: int, maximum iteration of `sinkhorn()` algorithm (default
    = 1000)

  - zero_tol: double, determine covergence (default = 1e-6)

  - verbose: int, print out debug info for the algorithm for every
    `verbose` iteration (default to 0, i.e. not printing anything)

## Value

list of results

- P: optimal coupling matrix

- grad_a: gradient of loss w.r.t. `a` (only with `with_grad = TRUE`)

- u, v: scaling vectors

- loss: regularized loss

- iter: iterations of the algorithm

- err: condition for convergence

- return_status: 0 (convergence), 1 (max iteration reached), 2 (other)

## Details

This is the general function to solve the OT problem, and it will use
either vanilla (`method = "vanilla"`) or log-stabilized Sinkhorn
algorithm (`method = "log"`) for solving the problem.

## References

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. https://doi.org/10.1561/2200000073

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
https://doi.org/10.48550/arXiv.2504.08722

## Examples

``` r
# simple sinkhorn example
a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1
sol <- sinkhorn(a, b, C, sinkhorn_control = list(reg = reg, verbose = 0))

# you can also supply arguments to control the computation
# for example, calculate the gradient w.r.t. a
sol <- sinkhorn(a, b, C,
sinkhorn_control = list(reg = reg, with_grad = TRUE, verbose = 0))
```
