
# call the functions from barycenter.cpp
# for the algorithms mentioned in Section 5 & 6

# TODO: add functions to check input types and return error if not met

#' @export
barycenter <- function(A, C, w, reg, withjac = FALSE,
                       method = "auto", threshold = .1,
                       maxIter = 1000, zeroTol = 1e-6, verbose = FALSE) {
  # TODO: check the input types!

  # auto-switch the best sinkhorn algo (between vanilla and log)
  k1 <- exp(- min(C) / reg)
  k2 <- exp(- max(C) / reg)
  if (min(k1,k2) < threshold) {
    # use log
    barycenter_log(A, C, w, reg, withjac, maxIter, zeroTol, verbose)
  } else {
    # use vanilla (parallel)
    barycenter_parallel(A, C, w, reg, withjac, maxIter, zeroTol, verbose)
  }
}


#' @export
barycenter_parallel <- function(A, C, w, reg, withjac = FALSE,
                                maxiter = 1000, zerotol = 1e-6, verbose = FALSE) {

  # call the C++ routine
  barycenter_parallel_cpp(A, C, w, reg, withjac, maxiter, zerotol, verbose)
}


#' @export
barycenter_log <- function(A, C, w, reg, withjac = FALSE,
                           maxiter = 1000, zerotol = 1e-6, verbose = FALSE) {

  # call the C++ routine
  barycenter_log_cpp(A, C, w, reg, withjac, maxiter, zerotol, verbose)
}
