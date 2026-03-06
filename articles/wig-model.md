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

wigfit <- wig(wigdf, ref_date, docs, specs = wig_specs(
  wdl_control = list(num_topics = 2),
  word2vec_control = list(min_count = 1)
))
#> Preprocessing the data...
#> Running tokenizer on the sentences...
#> Running Word2Vec for the embeddings and distance matrix...
#> `method` is automatically switched to "log"
#> Running WDL in CPU mode...
#> This might take a while depending on the problem size...
#> Running in serial mode...
#> Initializing WDL model with 3 vocabs, 2 docs, and 2 topics...
#> Training WDL model with 2 epochs, 1 batches
#> Epoch 1 of 2, batch 1 of 1:
#> avg speed: 0.00 sec, last speed: 0.00 sec
#> Epoch 2 of 2, batch 1 of 1:
#> avg speed: 0.00 sec, last speed: 0.00 sec
#> Inference on the dataset
wigfit
#> WIG model from 2012-01-01 to 2012-02-01
#> WDL model topics:
#> 
#> Topic 1:
#> sentenc    </s>   anoth 
#>   0.796   0.130   0.074
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
