# Truncated SVD

Truncated Singular Value Decomposition algorithm

## Usage

``` r
tsvd(M, k = 1, flip_sign = c("auto", "sklearn", "none"))
```

## Arguments

- M:

  matrix, data to be analyzed

- k:

  int, number of columns/features to be kept

- flip_sign:

  character, one of the following: "auto", "sklearn", "none"

## Value

matrix after dimension reduction

## Details

Compute the truncated SVD for dimension reduction. Note that SVD suffers
from "sign indeterminacy," which means that the signs of the output
vectors could depend on the algorithm and random state. Two variants of
"sign flipping methods" are implemented here, one following the sklearn
implementation on Truncated SVD, another by Bro et al. (2008).

## References

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

Bro, R., Acar, E., & Kolda, T. G. (2008). Resolving the sign ambiguity
in the singular value decomposition. Journal of Chemometrics, 22(2),
135–140. https://doi.org/10.1002/cem.1122

## See also

[Truncated SVD
(sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
