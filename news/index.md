# Changelog

## rwig 0.2.0

- Dropped the dependencies on Rcpp and RcppArmadillo. The C++ code talks
  to R through the native C API (`.Call` with registered routines) and
  calls the BLAS/LAPACK libraries shipped with R directly through a
  small internal matrix layer, which also makes the installed package
  much smaller. The package now has no compiled-code dependencies at
  all.
- Interrupting a long computation (Ctrl-C) now releases worker threads
  and buffers before returning to R, and surfaces as an R error.
- Faster log-stabilized
  [`sinkhorn()`](https://fangzhou-xie.github.io/rwig/reference/sinkhorn.md)
  and
  [`barycenter()`](https://fangzhou-xie.github.io/rwig/reference/barycenter.md):
  the soft-min kernels no longer materialize the M x N matrix
  `C - f 1' - 1 g'` on every iteration, and worker threads (`n_threads`)
  are created once per call instead of once per iteration.
- Faster [`wdl()`](https://fangzhou-xie.github.io/rwig/reference/wdl.md)
  on CPU: the softmax Jacobian is applied in O(N) instead of O(N^2) per
  column, an unused matrix product per backward step was removed, and
  the per-batch history buffers use about half the memory.
- [`wdl()`](https://fangzhou-xie.github.io/rwig/reference/wdl.md) now
  draws its random initialization from R’s RNG
  ([`rnorm()`](https://rdrr.io/r/stats/Normal.html)), so
  [`set.seed()`](https://rdrr.io/r/base/Random.html) reproduces the
  documented R sequence. Fits with the same seed therefore differ from
  version 0.1.0, which used Armadillo’s own generator.
- Fixed: [`wdl()`](https://fangzhou-xie.github.io/rwig/reference/wdl.md)
  returned `weights`, `docs_pred` and `docs_dist` in the shuffled
  training order (the default `shuffle = TRUE`) next to `docs` in the
  input order, so
  [`wig()`](https://fangzhou-xie.github.io/rwig/reference/wig.md) summed
  document scores into the wrong periods. Per-document outputs are now
  put back into the input order.
- Fixed: [`wig()`](https://fangzhou-xie.github.io/rwig/reference/wig.md)
  failed when `wig_control` was given without `group_unit`.
- [`wdl_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md)/[`wig_specs()`](https://fangzhou-xie.github.io/rwig/reference/wdl_specs.md)
  now honor the values passed in `barycenter_control` (`method`,
  `max_iter`, …) instead of silently overriding them; the `wdl_control`
  seed entry is named `seed` (it was documented as `rng_seed` but read
  as `seed`). `verbose` defaults to 0 when omitted from a partial
  `sinkhorn_control`/`barycenter_control`.
- Dropped the dependency on RhpcBLASctl: `rwig` no longer sets the BLAS
  thread count to 1 for the whole session when attached. The startup
  message now explains the recommendation and, if RhpcBLASctl is
  installed (it is now only suggested), reports the current thread count
  and the call to change it.
- Dropped the dependency on lubridate: `group_unit` is now passed to
  [`cut()`](https://rdrr.io/r/base/cut.html)
  ([`?cut.Date`](https://rdrr.io/r/base/cut.POSIXt.html)), so it accepts
  “day”, “week”, “month”, “quarter”, “year” and multiples such as “2
  months”. Weeks start on Monday.
- Fixed: threaded (`n_threads > 0`) log barycenter crashed when the cost
  matrix had more columns than rows.
- Fixed: the CUDA build passes an explicit GPU architecture to `nvcc`
  (recent CUDA toolkits default to `sm_52`, which lacks double-precision
  `atomicAdd`). Set `RWIG_CUDA_ARCH` to override the detected flag, or
  `RWIG_NO_CUDA=1` to build without GPU support even when CUDA is
  installed.

## rwig 0.1.0

CRAN release: 2026-04-21

- Initial CRAN submission.
- Efficient implementation of several Optimal Transport algorithms in  
  Fangzhou Xie (2025) <doi:10.48550/arXiv.2504.08722> and the
  Wasserstein Index Generation (WIG) model in Fangzhou Xie (2020)
  <doi:10.1016/j.econlet.2019.108874>.
