# WIG Model

``` r
library(wig) |> suppressPackageStartupMessages()
```

The Wasserstein Index Generation (WIG) model leverages the WDL model for
topic modeling and generates time series sentiment index, given that the
docs are associated with timestamps. This was used to automatically
reconstruct the Economic Policy Uncertain (EPU) index.

``` r
# create a small dataset
wigdf <- data.frame(
  ref_date = as.Date(c("2012-01-01", "2012-02-01")),
  docs = c("this is a sentence", "this is another sentence")
)

wigfit <- wig(wigdf, ref_date, docs)
#> `method` is automatically switched to "log"
wigfit
#> WIG model from 2012-01-01 to 2012-02-01
#> WDL model topics:
#> 
#> Topic 1:
#>        a     this  another     </s> sentence 
#>     0.31     0.27     0.21     0.15     0.04
```

## See Also

See also
[`vignette("wdl-model")`](https://fangzhou-xie.github.io/wig/articles/wdl-model.md),
[`vignette("specs")`](https://fangzhou-xie.github.io/wig/articles/specs.md).

## References

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic
policy uncertainty. *The Quarterly Journal of Economics*, 131(4),
1593–1636. <https://doi.org/10.1093/qje/qjw024>

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with application to economic policy uncertainty.
*Economics Letters*, 186, 108874.
<https://doi.org/10.1016/j.econlet.2019.108874>
