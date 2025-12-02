# Computation of the Gradients

``` r
library(rwig) |> suppressPackageStartupMessages()
```

To set up the computation for the
[`sinkhorn()`](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.md)
and
[`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md)
algorithms, you will need to set `with_grad = TRUE` for
`sinkhorn_control` and `barycenter_control`.

The exact formulae of gradients were given by Xie (2025), and have been
checked by the Automatic Differentiation library `ForwardDiff` in Julia.

## See Also

See also
[`vignette("sinkhorn")`](https://fangzhou-xie.github.io/rwig/articles/sinkhorn.md),
[`vignette("barycenter")`](https://fangzhou-xie.github.io/rwig/articles/barycenter.md).

## References

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
