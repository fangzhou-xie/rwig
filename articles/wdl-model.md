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
sentences <- c("this is a sentence", "this is another one", "yet another sentence")
wdl_fit <- wdl(sentences, specs = wdl_specs(
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
#> Initializing WDL model with 5 vocabs, 3 docs, and 2 topics...
#> Training WDL model with 2 epochs, 1 batches
#> Epoch 1 of 2, batch 1 of 1:
#> avg speed: 0.00 sec, last speed: 0.00 sec
#> Epoch 2 of 2, batch 1 of 1:
#> avg speed: 0.00 sec, last speed: 0.00 sec
#> Inference on the dataset

wdl_fit
#> WDL model topics:
#> 
#> Topic 1:
#>     yet   anoth     one sentenc    </s> 
#>   0.346   0.281   0.173   0.112   0.088 
#> 
#> Topic 2:
#>   anoth    </s>     one sentenc     yet 
#>    0.32    0.21    0.19    0.16    0.12
```

We can see from the topics that they are vectors of the tokens (words)
with associated probabilities. If you want to access the topics, you can
do this:

``` r

wdl_fit$topics
#>             topic1    topic2
#> one     0.17323975 0.1902754
#> yet     0.34558828 0.1191494
#> anoth   0.28110783 0.3235259
#> sentenc 0.11233815 0.1616852
#> </s>    0.08772599 0.2053640
```

Alternatively, you can also obtain the weights of the topics used to
re-construct the input data:

``` r

wdl_fit$weights
#>             [,1]     [,2]      [,3]
#> topic1 0.8751024 0.604339 0.6874352
#> topic2 0.1248976 0.395661 0.3125648
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
