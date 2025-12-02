# Sinkhorn Algorithms

## Get Started with Sinkhorn Algorithm

``` r
library(wig) |> suppressPackageStartupMessages()
```

Suppose we have two probability vectors **a** and **b**, for example,

``` r
a <- c(.3, .4, .1, .1, .1)
b <- c(.4, .5, .1)
```

and the cost matrix **C** between them

``` r
C <- rbind(
  c(.1, .2, .3),
  c(.2, .3, .4),
  c(.4, .3, .2),
  c(.3, .2, .1),
  c(.5, .5, .4)
)
reg <- .1
```

and the regularization parameter `reg = .1`. We can then compute the
optimal transport plan by the
[`sinkhorn()`](https://fangzhou-xie.github.io/wig/reference/sinkhorn.md)
algorithm:

``` r
sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    verbose = 0
  )
)
#> $P
#>             [,1]       [,2]        [,3]
#> [1,] 0.153872665 0.13773502 0.008392325
#> [2,] 0.205163553 0.18364670 0.011189767
#> [3,] 0.009441141 0.06244481 0.028114038
#> [4,] 0.009441141 0.06244481 0.028114038
#> [5,] 0.022081501 0.05372866 0.024189833
#> 
#> $f
#> [1] -0.06969114  0.05907707 -0.04879600 -0.14879600  0.13617034
#> 
#> $g
#> [1] -0.01747185  0.07144879 -0.10835262
#> 
#> $loss
#> [1] -0.08551829
#> 
#> $iter
#> [1] 10
#> 
#> $err
#> [1] 2.149571e-07
#> 
#> $return_status
#> [1] 0
#> 
#> $method
#> [1] "log"
```

The optimal coupling matrix (optimal transport plan) is **P**, and two
scaling vectors **f**, **g** with the method being “log”.

## Vanilla, Log, and Auto Sinkhorn Algorithms

There are two different methods for the Sinkhorn algorithm implemented
in the `wig` package for the computation of the regularized OT problem:
the vanilla Sinkhorn and the log-stabilized Sinkhorn.

The vanilla algorithm is usually faster, as it is inherently parallel
and the computation can be done purely in matrix-vector and
matrix-matrix operations.

However, the vanilla algorithm is commonly known for its instability
when the regularization parameter is very small, and the log-stabilized
Sinkhorn will handle any arbitrarily small `reg`. But the cost of doing
so is that the log algorithm is typically slower than the vanilla one,
as it requires column-by-column and row-by-row soft-minimums.

For example, we can have

``` r
sol_vanilla <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "vanilla",
    verbose = 0
  )
)
sol_log <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "log",
    verbose = 0
  )
)
```

To speed up the log algorithm, I used multi-threading to speed up the
computation.

``` r
sol_thread <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    method = "log",
    n_threads = 4,
    verbose = 0
  )
)
```

Moreover, I also provided an “auto” method, trying to provide a
heuristics for determining whether to use “vanilla” or “log” method
automatically. This is also the default algorithm.

Although you mostly likely should avoid doing so, you can also tweak the
`threshold` argument to tune the “auto” method. The default value is
`.1`: smaller means more likely to use “vanilla” method, bigger means
more likely to use “log” method. If you know which algorithm to use, you
should directly set `method` argument instead of doing this.

``` r
sol_auto <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    threshold = .1,
    verbose = 0
  )
)
```

See also
[`vignette("barycenter")`](https://fangzhou-xie.github.io/wig/articles/barycenter.md).

## Gradient of Loss w.r.t. **a**

As derived in Xie (2025), I also provide the gradient of the Sinkhorn
algorithm w.r.t. to the source density **a**. The default for gradient
calculation is `FALSE`, but you can turn it on by the following:

``` r
sol_grad <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    with_grad = TRUE,
    verbose = 0
  )
)

sol_grad$grad_a
#> [1] -0.06627247  0.06249575 -0.04537730 -0.14537729  0.13958900
```

See also
[`vignette("barycenter")`](https://fangzhou-xie.github.io/wig/articles/barycenter.md),
[`vignette("gradient")`](https://fangzhou-xie.github.io/wig/articles/gradient.md).

## Other Control Parameters

This is the default `sinkhorn_control` argument, and we have discussed
`reg`, `with_grad`, `threshold` already.

``` r
sinkhorn_control = list(
  reg = .1,
  with_grad = FALSE,
  method = "auto",
  threshold = .1,
  max_iter = 1000L,
  zero_tol = 1e-6,
  verbose = 0L
)
```

The `max_iter` and `zero_tol` arguments are the parameters to determine
maximum number of iterations and the convergence condition for the
Sinkhorn algorithms.

The `verbose` are useful for printing out some useful information during
the computation. You can set it as an non-negative integer (0 meaning
not verbose), and the algorithm will update info for every `verbose`
steps. For example,

``` r
sol_verbose <- sinkhorn(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = reg,
    verbose = 10
  )
)
#> `method` is automatically switched to "log"
#> Forward pass:
#> iter: 1, err: 0.3201, last speed: 0.000, avg speed: 0.000
```

See also
[`vignette("barycenter")`](https://fangzhou-xie.github.io/wig/articles/barycenter.md).

## Reference

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
