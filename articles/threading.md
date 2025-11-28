# Advanced Topic: Multi-threading

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
```
