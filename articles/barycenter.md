# Barycenter Algorithms

``` r
library(rwig) |> suppressPackageStartupMessages()
```

Suppose we have a matrix **A** of size $M \times S$, the cost matrix
**C** of size $M \times N$, and we are aiming to compute the barycenter
vector of size $N$. For example,

``` r
A <- rbind(
  c(.3, .2),
  c(.2, .1),
  c(.1, .2),
  c(.1, .1),
  c(.3, .4)
)
C <- rbind(
  c(.1, .2, .3, .4, .5),
  c(.2, .3, .4, .3, .2),
  c(.4, .3, .2, .1, .2),
  c(.3, .2, .1, .2, .5),
  c(.5, .5, .4, .0, .2)
)
w <- c(.4, .6)
reg <- .1

sol <- barycenter(A, C, w, barycenter_control = list(reg = reg))
#> `method` is automatically switched to "log"
#> Forward pass:
#> iter: 1, err: 0.3207, last speed: 0.000, avg speed: 0.000
#> iter: 11, err: 0.0020, last speed: 0.000, avg speed: 0.000
#> iter: 21, err: 0.0000, last speed: 0.000, avg speed: 0.000
```

## Difference from `sinkhorn()`

The interface for
[`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md)
is almost identical to
[`sinkhorn()`](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.md)
(see
[`vignette("sinkhorn")`](https://fangzhou-xie.github.io/rwig/articles/sinkhorn.md)),
except for the name of the algorithm.
[`sinkhorn()`](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.md)
accepts three parameters for the `method` argument: `vanilla`, `log`,
and `auto`; whereas
[`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md)
accepts `parallel`, `log`, and `auto`.

You can still set the gradient, threading (only for `log`), and all
other parameters to control the computation as in
[`sinkhorn()`](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.md),
but you will also need to supply an external vector for `b_ext` to
compute the quadratic loss between the output barycenter and `b_ext`.

``` r
b <- c(.2, .2, .2, .2, .2)
sol <- barycenter(A, C, w, b_ext = b, barycenter_control = list(reg = reg, with_grad = TRUE))
```

## See Also

See also
[`vignette("sinkhorn")`](https://fangzhou-xie.github.io/rwig/articles/sinkhorn.md).

## Reference

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
