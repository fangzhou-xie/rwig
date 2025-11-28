
<!-- README.md is generated from README.Rmd. Please edit that file -->

# wig

<!-- badges: start -->

[![DOI:
10.1016/j.econlet.2019.108874](https://img.shields.io/badge/doi-10.1016/j.econlet.2019.108874-yellow.svg)](https://doi.org/10.1016/j.econlet.2019.108874)
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://cran.r-project.org/web/licenses/MIT)
[![experimental
version](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R build
status](https://github.com/fangzhou-xie/rethnicity/workflows/R-CMD-check/badge.svg)](https://github.com/fangzhou-xie/rethnicity/actions)
<!-- badges: end -->

The `wig` package implements the Sinkhorn algorithms for regularized
Optimal Transport problems, Wasserstein Barycenter algorithms for the
regularized Wasserstein Barycenter problems, Wasserstein Dictionary
Learning (WDL) model, and **W**asserstein **I**ndex **G**eneration
(**WIG**) model (see references below).

All the methods are implemented from the ground up with C++ and
Armadillo, with additional support for multi-threading for the
log-stablized methods for
[sinkhorn](https://fangzhou-xie.github.io/wig/reference/sinkhorn.html)
and
[barycenter](https://fangzhou-xie.github.io/wig/reference/barycenter.html).
See the
[vignette](https://fangzhou-xie.github.io/wig/articles/threading.html)
on multi-threading.

## Installation

<!-- This package is on [CRAN](https://cran.rstudio.com/), and I recommened  -->

<!-- to use the `pak` to install it: -->

<!-- ``` r -->

<!-- # install pak if not already done so -->

<!-- # install.packages("pak") -->

<!-- pak::pak("wig") -->

<!-- # or you can install it in the classic way -->

<!-- install.packages("wig") -->

<!-- ``` -->

You can install the development version of wig from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("fangzhou-xie/wig")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(wig)
#> ══ WARNING: ═════════════════════════════════════════════════════════════ wig ══
#> ! Your current BLAS backend is: /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3. For better performance, it's recommended to use an optimized BLAS library, such as Inter MKL or OpenBLAS. For example, you can consider ropenblas (<https://github.com/prdm0/ropenblas>).
#> ! Automatically setting BLAS thread to be 1, and this is recommended for most users. If you want to set up threading for faster processing, you can set `n_threads` argument in functions: sinkhorn() (<https://fangzhou-xie.github.io/wig/reference/sinkhorn.html>) and barycenter() (<https://fangzhou-xie.github.io/wig/reference/barycenter.html>) Please read the vignette (<https://fangzhou-xie.github.io/wig/articles/threading.html>) for advanced usage on threading.
#> 
#> ══ CITATION: ════════════════════════════════════════════════════════════ wig ══
#> ℹ Please use the following to cite my works, thanks!
#> 
#>  @article{xie2020,
#>    title = {Wasserstein Index Generation Model: Automatic Generation of Time-Series Index with Application to Economic Policy Uncertainty},
#>    author = {Xie, Fangzhou},
#>    year = 2020,
#>    journal = {Economics Letters},
#>    volume = {186},
#>    pages = {108874},
#>    issn = {0165-1765},
#>    doi = {10.1016/j.econlet.2019.108874},
#>    urldate = {2019-12-10},
#>  }

## basic example code
```

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
of time-series index with application to economic policy uncertainty.
Economics Letters, 186, 108874.
<https://doi.org/10.1016/j.econlet.2019.108874>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). arXiv.
<https://doi.org/10.48550/arXiv.2504.08722>
