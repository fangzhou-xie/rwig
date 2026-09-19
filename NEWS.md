# rwig 0.2.0

- Dropped the dependencies on Rcpp and RcppArmadillo. The C++ code talks to
  R through the native C API (`.Call` with registered routines) and calls the
  BLAS/LAPACK libraries shipped with R directly through a small internal
  matrix layer, which also makes the installed package much smaller. The
  package now has no compiled-code dependencies at all.
- Interrupting a long computation (Ctrl-C) now releases worker threads and
  buffers before returning to R, and surfaces as an R error.
- Faster log-stabilized `sinkhorn()` and `barycenter()`: the soft-min
  kernels no longer materialize the M x N matrix `C - f 1' - 1 g'` on every
  iteration, and worker threads (`n_threads`) are created once per call
  instead of once per iteration.
- Faster `wdl()` on CPU: the softmax Jacobian is applied in O(N) instead of
  O(N^2) per column, an unused matrix product per backward step was removed,
  and the per-batch history buffers use about half the memory.
- `wdl()` now draws its random initialization from R's RNG (`rnorm()`), so
  `set.seed()` reproduces the documented R sequence. Fits with the same seed
  therefore differ from version 0.1.0, which used Armadillo's own generator.
- Fixed: threaded (`n_threads > 0`) log barycenter crashed when the cost
  matrix had more columns than rows.
- Fixed: the CUDA build passes an explicit GPU architecture to `nvcc`
  (recent CUDA toolkits default to `sm_52`, which lacks double-precision
  `atomicAdd`). Set `RWIG_CUDA_ARCH` to override the detected flag, or
  `RWIG_NO_CUDA=1` to build without GPU support even when CUDA is installed.

# rwig 0.1.0

- Initial CRAN submission.
- Efficient implementation of several Optimal Transport algorithms in  
   Fangzhou Xie (2025) <doi:10.48550/arXiv.2504.08722> and
  the Wasserstein Index Generation (WIG) model in
  Fangzhou Xie (2020) <doi:10.1016/j.econlet.2019.108874>.
