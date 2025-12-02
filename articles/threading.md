# Multi-threading Support

``` r
library(wig) |> suppressPackageStartupMessages()
```

The log methods for
[`sinkhorn()`](https://fangzhou-xie.github.io/wig/reference/sinkhorn.md)
and
[`barycenter()`](https://fangzhou-xie.github.io/wig/reference/barycenter.md)
both require row-by-row and column-by-column *soft-minimum* operations
for each iteration of the algorithm, and therefore suffer from slow
computation time compared to the vanilla/parallel version.

If the dimensions of $M$ and $N$ are large (recall that cost matrix
**C** is of size $M \times N$), we can use multi-threading to process
the rows and columns simultaneously.

This can be done by setting `n_threads` to an integer bigger than 0. By
default it is 0, and it means that threading is disabled. Also, setting
`n_threads` for “vanilla” or “parallel” methods will be ignored
automatically.

But you might ask: if multi-threading is so wonderful, why don’t you set
threading as the default? This is because threading comes with an
overhead, and sometimes for small problems, it can even be slower than
serial processing. So be sure to benchmark your code and see if
threading actually helps.

## See Also

See also
[`vignette("sinkhorn")`](https://fangzhou-xie.github.io/wig/articles/sinkhorn.md),
[`vignette("barycenter")`](https://fangzhou-xie.github.io/wig/articles/barycenter.md).

## References

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
