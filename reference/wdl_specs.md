# Model Specs for WDL and WIG models

Control the parameters of WDL and WIG models

## Usage

``` r
wdl_specs(
  wdl_control = list(num_topics = 4, batch_size = 64, epochs = 2, shuffle = TRUE, verbose
    = FALSE),
  tokenizer_control = list(),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 1),
  barycenter_control = list(reg = 0.1, with_grad = TRUE, n_threads = 0, method = "auto",
    threshold = 0.1, max_iter = 20, zero_tol = 1e-06),
  optimizer_control = list(optimizer = "adamw", lr = 0.005, decay = 0.01, beta1 = 0.9,
    beta2 = 0.999, eps = 1e-08)
)

wig_specs(
  wig_control = list(group_unit = "month", svd_method = "docs", standardize = TRUE),
  wdl_control = list(num_topics = 4, batch_size = 64, epochs = 2, rng_seed = 123, verbose
    = FALSE),
  tokenizer_control = list(),
  word2vec_control = list(type = "cbow", dim = 10, min_count = 1),
  barycenter_control = list(reg = 0.1, with_grad = TRUE, method = "auto", threshold =
    0.1, max_iter = 20, zero_tol = 1e-06),
  optimizer_control = list(optimizer = "adamw", lr = 0.005, decay = 0.01, beta1 = 0.9,
    beta2 = 0.999, eps = 1e-08)
)
```

## Arguments

- wdl_control, :

  list, parameters for WDL

- tokenizer_control, :

  list, parameters for
  [`tokenizers::tokenize_words()`](https://docs.ropensci.org/tokenizers/reference/basic-tokenizers.html)

- word2vec_control, :

  list, parameters for
  [`word2vec::word2vec()`](https://rdrr.io/pkg/word2vec/man/word2vec.html)

- barycenter_control, :

  list, parameters for
  [`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md)

- optimizer_control, :

  list, parameters for the optimizer (SGD, Adam, AdamW)

- wig_control, :

  list, parameters for WIG model

## Value

list of the control lists

## Details

See
[`vignette("specs")`](https://fangzhou-xie.github.io/rwig/articles/specs.md)
for details on the parameters.

## References

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. https://doi.org/10.1561/2200000073

Schmitz, M. A., Heitz, M., Bonneel, N., Ngolè, F., Coeurjolly, D.,
Cuturi, M., Peyré, G., & Starck, J.-L. (2018). Wasserstein dictionary
learning: Optimal transport-based unsupervised nonlinear dictionary
learning. *SIAM Journal on Imaging Sciences*, 11(1), 643–678.
https://doi.org/10.1137/17M1140431

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
optimization. *International Conference on Learning Representations
(ICLR)*.

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay
Regularization (No. arXiv:1711.05101). *arXiv*.
https://doi.org/10.48550/arXiv.1711.05101

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with application to economic policy uncertainty.
*Economics Letters*, 186, 108874.
https://doi.org/10.1016/j.econlet.2019.108874

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
https://doi.org/10.48550/arXiv.2504.08722

## See also

`wig_specs()`,
[`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md),
[`word2vec::word2vec()`](https://rdrr.io/pkg/word2vec/man/word2vec.html),
[`tokenizers::tokenize_words()`](https://docs.ropensci.org/tokenizers/reference/basic-tokenizers.html),
[`vignette("specs")`](https://fangzhou-xie.github.io/rwig/articles/specs.md)
