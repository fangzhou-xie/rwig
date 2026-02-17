# CUDA Acceleration

``` r
library(rwig) |> suppressPackageStartupMessages()
```

The Wasserstein Index Generation (WIG) model relies on the
computationally expensive Optimal Transport methods, and could be
potentially very slow if the dimension of the problem is very large. In
order to speed up the computation, I have implemented some functions in
CUDA for faster processing with GPU. This is currently only available on
Linux machines, with CUDA installed.

To see if the `rwig` package on your machine is built with CUDA, you can
use the following function to check

``` r
check_cuda()
#> [1] FALSE
```
