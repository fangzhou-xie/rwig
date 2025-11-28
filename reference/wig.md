# Wasserstein Index Generation model

Wasserstein Index Generation (WIG) model for time-series sentiment index
autogeneration

## Usage

``` r
wig(.data, date_col, docs_col, ...)

# S3 method for class 'wig'
print(object, topic = 1, token_per_topic = 5, ...)
```

## Arguments

- date_col:

  name of the column for dates

- docs_col:

  name of the column for the texts

- spec:

  list, model specification for WIG see `wig_spec()` for reference

- verbose:

  bool, whether to print useful info

## Details

This is the re-implementation of WIG model from scratch in R.

## References

Xie, F. (2020). Wasserstein index generation model: Automatic generation
of time-series index with application to economic policy uncertainty.
*Economics Letters*, 186, 108874.
https://doi.org/10.1016/j.econlet.2019.108874

## Examples

``` r
# create a small dataset
wigdf <- data.frame(
  ref_date = as.Date(c("2012-01-01", "2012-02-01")),
  docs = c("this is a sentence", "this is another sentence")
)

wigfit <- wig(wigdf, ref_date, docs)
#> `method` is automatically switched to "log"


```
