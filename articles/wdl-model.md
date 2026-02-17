# WDL Model

``` r
library(rwig) |> suppressPackageStartupMessages()
```

Let’s say we have some documents as character vectors, and we want to
discover the underlying topics. This is called “topic modeling”, and
Latent Dirichlet Allocation (LDA) is probably the most famous among all
of topic models. Here, we consider the Wasserstein Dicionary Learning
(WDL) model.

``` r
# a very simple example
sentences <- c("this is a sentence", "this is another one")
wdl_fit <- wdl(sentences)
#> `method` is automatically switched to "log"

wdl_fit
#> WDL model topics:
#> 
#> Topic 1:
#>     one   anoth    </s> sentenc 
#>   0.427   0.363   0.131   0.079 
#> 
#> Topic 2:
#>   anoth     one sentenc    </s> 
#>   0.452   0.291   0.211   0.046 
#> 
#> Topic 3:
#>    </s>     one sentenc   anoth 
#>   0.660   0.163   0.139   0.039 
#> 
#> Topic 4:
#>     one    </s> sentenc   anoth 
#>    0.52    0.22    0.15    0.11
```

We can see from the topics that they are vectors of the tokens (words)
with associated probabilities. If you want to access the topics, you can
do this:

``` r
wdl_fit$topics
#>             topic1     topic2     topic3    topic4
#> anoth   0.36337276 0.45247625 0.03918154 0.1077225
#> one     0.42653605 0.29081991 0.16272523 0.5167396
#> sentenc 0.07892702 0.21085570 0.13856620 0.1533639
#> </s>    0.13116417 0.04584815 0.65952702 0.2221740
```

Alternatively, you can also obtain the weights of the topics used to
re-construct the input data:

``` r
wdl_fit$weights
#>              [,1]       [,2]
#> topic1 0.42835430 0.31270288
#> topic2 0.11177570 0.05975748
#> topic3 0.42987922 0.40321258
#> topic4 0.02999078 0.22432706
```

## See Also

See also
[`vignette("specs")`](https://fangzhou-xie.github.io/rwig/articles/specs.md).

## References

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet
allocation. *Journal of Machine Learning Research*, 3(Jan), 993–1022.

Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport: With
Applications to Data Science. *Foundations and Trends® in Machine
Learning*, 11(5–6), 355–607. <https://doi.org/10.1561/2200000073>

Schmitz, M. A., Heitz, M., Bonneel, N., Ngolè, F., Coeurjolly, D.,
Cuturi, M., Peyré, G., & Starck, J.-L. (2018). Wasserstein dictionary
learning: Optimal transport-based unsupervised nonlinear dictionary
learning. *SIAM Journal on Imaging Sciences*, 11(1), 643–678.
<https://doi.org/10.1137/17M1140431>

Xie, F. (2025). Deriving the Gradients of Some Popular Optimal Transport
Algorithms (No. arXiv:2504.08722). *arXiv*.
<https://doi.org/10.48550/arXiv.2504.08722>
