# WIG Model

``` r
library(rwig) |> suppressPackageStartupMessages()
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
#> sentenc    </s>   anoth 
#>   0.794   0.132   0.074
```

## See Also

See also
[`vignette("wdl-model")`](https://fangzhou-xie.github.io/rwig/articles/wdl-model.md),
[`vignette("specs")`](https://fangzhou-xie.github.io/rwig/articles/specs.md).

## References

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic
policy uncertainty. *The Quarterly Journal of Economics*, 131(4),
1593–1636. <https://doi.org/10.1093/qje/qjw024>

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with application to economic policy uncertainty.
*Economics Letters*, 186, 108874.
<https://doi.org/10.1016/j.econlet.2019.108874>
