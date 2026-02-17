# Check if CUDA is available

Check if CUDA is available for GPU computations.

## Usage

``` r
check_cuda()
```

## Value

logical, TRUE if CUDA is available, FALSE otherwise

## Examples

``` r
if (check_cuda()) {
  cat("CUDA is available for GPU computations.\n")
} else {
  cat("CUDA is not available.\n")
}
#> CUDA is not available.
```
