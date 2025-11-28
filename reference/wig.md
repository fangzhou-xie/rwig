# Wasserstein Index Generation model

Wasserstein Index Generation (WIG) model for time-series sentiment index
autogeneration

## Usage

``` r
wig(.data, date_col, docs_col, ...)

# S3 method for class 'data.frame'
wig(.data, date_col, docs_col, specs = wig_specs(), ...)

# S3 method for class 'wig'
print(x, topic = 1, token_per_topic = 5, ...)

# S3 method for class 'wig'
summary(object, topic = 1, token_per_topic = 5, ...)
```

## Arguments

- .data:

  a dataframe containing the dates/datetimes and documents

- date_col:

  name of the column for dates/datetimes

- docs_col:

  name of the column for the texts/documents

- ...:

  only for compatibility

- specs:

  list, model specification for WIG see
  [`wig_specs`](https://fangzhou-xie.github.io/wig/reference/wdl_specs.md)
  for reference

- x:

  WIG model

- topic:

  int, number of topic to be printed

- token_per_topic:

  int, number of tokens to be printed

- object:

  WIG model

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
