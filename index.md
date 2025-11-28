# wig

The `wig` package implements the Sinkhorn algorithms for regularized
Optimal Transport problems, Wasserstein Barycenter algorithms for the
regularized Wasserstein Barycenter problems, Wasserstein Dictionary
Learning (WDL) model, and **W**asserstein **I**ndex **G**eneration
(**WIG**) model (see references below).

All the methods are implemented from the ground up with C++ and
Armadillo (with Rcpp and RcppArmadillo), with additional support for
multi-threading for the log-stablized methods for
[sinkhorn](https://fangzhou-xie.github.io/wig/reference/sinkhorn.html)
and
[barycenter](https://fangzhou-xie.github.io/wig/reference/barycenter.html).
See the
[vignette](https://fangzhou-xie.github.io/wig/articles/threading.html)
on multi-threading for faster processing.

## Installation

The package is currently under heavy development and can only be
considered as alpha stage. You can install the development version of
wig from [GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("fangzhou-xie/wig")
```

## Get Started

Please check out all the vignettes for the examples of using this
package under the “Articles” drop down menu.

## Reference

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. Foundations and Trends® in Machine
Learning, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Schmitz, M. A., Heitz, M., Bonneel, N., Ngolè, F., Coeurjolly, D.,
Cuturi, M., Peyré, G., & Starck, J.-L. (2018). Wasserstein dictionary
learning: Optimal transport-based unsupervised nonlinear dictionary
learning. SIAM Journal on Imaging Sciences, 11(1), 643–678.
<https://doi.org/10.1137/17M1140431>

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with applieion to economic policy uncertainty.
Economics Letters, 186, 108874.
<https://doi.org/10.1016/j.econlet.2019.108874>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). arXiv.
<https://doi.org/10.48550/arXiv.2504.08722>
