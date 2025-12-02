# WDL Model

``` r
library(wig) |> suppressPackageStartupMessages()
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
#>     </s>     this      one  another sentence 
#>    0.311    0.304    0.108    0.104    0.088 
#> 
#> Topic 2:
#> another    </s>     one    this       a 
#>   0.620   0.130   0.111   0.055   0.041 
#> 
#> Topic 3:
#>     this     </s>  another       is sentence 
#>    0.358    0.239    0.124    0.114    0.069 
#> 
#> Topic 4:
#> another    this    </s>     one      is 
#>   0.343   0.168   0.167   0.121   0.098
```

We can see from the topics that they are vectors of the tokens (words)
with associated probabilities. If you want to access the topics, you can
do this:

``` r
wdl_fit$topics
#>              topic1     topic2     topic3     topic4
#> a        0.07021960 0.04082612 0.06733181 0.03903212
#> one      0.10845734 0.11074871 0.02812378 0.12068278
#> another  0.10403697 0.62005355 0.12403799 0.34305199
#> sentence 0.08809656 0.02024121 0.06926678 0.06338548
#> is       0.01420343 0.02331677 0.11404292 0.09835958
#> this     0.30392984 0.05453664 0.35799228 0.16825826
#> </s>     0.31105626 0.13027700 0.23920443 0.16722978
```

Alternatively, you can also obtain the weights of the topics used to
re-construct the input data:

``` r
wdl_fit$weights
#>              [,1]       [,2]
#> topic1 0.08664043 0.07950508
#> topic2 0.03458663 0.59189857
#> topic3 0.05828124 0.07508092
#> topic4 0.82049170 0.25351543
```

## See Also

See also
[`vignette("specs")`](https://fangzhou-xie.github.io/wig/articles/specs.md).

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
