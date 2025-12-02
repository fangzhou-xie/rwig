# WDL and WIG Model Specs

``` r
library(rwig) |> suppressPackageStartupMessages()
```

In this vignette, I will show how to set up the control parameters
(hyper-parameters) needed for the WDL and WIG models.

The
[`wdl_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md)
is a list of lists, and consists of 5 parts (lists): `wdl_control`,
`tokenizer_control`, `word2vec_control`, `barycenter_control`,
`optimizer_control`.

The
[`wig_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md)
is the same as
[`wdl_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md),
with additional `wig_control`.

## `wig_control`

This is the options only needed for
[`wig_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md).
By default, it is

``` r
wig_control = list(
  group_unit = "month",
  svd_method = "docs",
  standardize = TRUE
)
```

1.  `group_unit` dictates at which level of time to group the documents,
    and it will be passed to
    [`lubridate::floor_date()`](https://lubridate.tidyverse.org/reference/round_date.html)
    as the `unit` argument. The default option is “month” to obtain
    monthly time series index, and other options can be specified
    following the `unit` argument in
    [`lubridate::floor_date()`](https://lubridate.tidyverse.org/reference/round_date.html).
2.  `svd_method` can be either “docs” or “topics”. The “doc” method
    means the Truncated SVD will be applied on the reconstructed
    documents to get the index directly; whereas “topics” means TSVD
    will be applied to the topics matrix before the construction of the
    index. The latter one is the one originally proposed in Xie (2020).
3.  `standardize`: bool, whether or not to standardize the result index
    as mean 100 and standard deviation 1. This is default to be true,
    following Baker et al. (2016), Xie (2020).

## `wdl_control`

This is the options supplied to the WDL modelling, and is used for both
[`wdl_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md)
and
[`wig_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md).

1.  `num_topics`: number of topics for the topic modeling
2.  `batch_size`: batch size for the training purpose
3.  `epochs`: epochs (i.e. number of passes) for the training data
4.  `shuffle`: bool, whether to shuffle the input data randomly
5.  `verbose`: bool, whether to print out useful diagnostic information

## `tokenizer_control`

Arguments for
[`tokenizers::tokenize_words()`](https://docs.ropensci.org/tokenizers/reference/basic-tokenizers.html).

## `word2vec_control`

Arguments for
[`word2vec::word2vec()`](https://rdrr.io/pkg/word2vec/man/word2vec.html),
but with the following default parameters:

``` r
type = "cbow"
dim = 10
min_count = 1
```

## `barycenter_control`

Identical to `barycenter_control` in
[`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md)
function, but with default

``` r
with_grad = TRUE
max_iter = 20
```

## `optimizer_control`

Parameters to control the optimizer (SGD, Adam, AdamW).

``` r
optimizer_control = list(
  optimizer = "adamw",
  lr = .005,
  decay = .01,
  beta1 = .9,
  beta2 = .999,
  eps = 1e-8
)
```

The default optimizer is AdamW (“adamw”), but you can also choose
vanilla SGD (“sgd”) or the vanilla (“adam”). You can also set the
learning rate `lr` in your hyper-parameter search.

The other default parameters should mostly be untouched for most people,
unless you know exactly what you are doing. For a reference, you can see
Section 7.1 in Xie (2025), and the references within.

## See Also

See also
[`vignette("wdl-model")`](https://fangzhou-xie.github.io/rwig/articles/wdl-model.md),
[`vignette("wig-model")`](https://fangzhou-xie.github.io/rwig/articles/wig-model.md).

## References

Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic
policy uncertainty. *The Quarterly Journal of Economics*, 131(4),
1593–1636. <https://doi.org/10.1093/qje/qjw024>

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Schmitz, M. A., Heitz, M., Bonneel, N., Ngolè, F., Coeurjolly, D.,
Cuturi, M., Peyré, G., & Starck, J.-L. (2018). Wasserstein dictionary
learning: Optimal transport-based unsupervised nonlinear dictionary
learning. *SIAM Journal on Imaging Sciences*, 11(1), 643–678.
<https://doi.org/10.1137/17M1140431>

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with application to economic policy uncertainty.
*Economics Letters*, 186, 108874.
<https://doi.org/10.1016/j.econlet.2019.108874>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
