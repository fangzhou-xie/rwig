
# call the functions from sinkhorn.cpp
# for the algorithms mentioned in Section 3 & 4

#' @export
sinkhorn <- function(
    a, b, C, reg, withgrad = FALSE,
    method = "auto", threshold = .1, maxIter = 1000, zeroTol = 1e-6) {
  # auto-switch the best sinkhorn algo (between vanilla and log)
  k1 <- exp(- min(C) / reg)
  k2 <- exp(- max(C) / reg)
  if (min(k1,k2) < threshold) {
    # use log
    sinkhorn_log(a, b, C, reg, withgrad, maxIter, zeroTol)
  } else {
    # use vanilla
    sinkhorn_vanilla(a, b, C, reg, withgrad, maxIter, zeroTol)
  }
}


#' @export
sinkhorn_vanilla <- function(a, b, C, reg, withgrad = FALSE,
                             maxiter = 1000, zerotol = 1e-6) {
  if (!is.vector(a)) stop("a must be a numeric vector!")
  if (!is.vector(b)) stop("b must be a numeric vector!")
  if (!is.matrix(C)) stop("C must be a numeric matrix!")
  if ((length(a) != nrow(C)) | (length(b) != ncol(C)))
    stop("a, b, C don't match in dimension!")

  # call the c++ routine
  sinkhorn_vanilla_cpp(a, b, C, reg, withgrad, maxiter, zerotol)
}

#' @export
sinkhorn_log <- function(a, b, C, reg, withgrad = FALSE,
                         maxiter = 1000, zerotol = 1e-6) {
  if (!is.vector(a)) stop("a must be a numeric vector!")
  if (!is.vector(b)) stop("b must be a numeric vector!")
  if (!is.matrix(C)) stop("C must be a numeric matrix!")
  if ((length(a) != nrow(C)) | (length(b) != ncol(C)))
    stop("a, b, C don't match in dimension!")

  # call the c++ routine
  sinkhorn_log_cpp(a, b, C, reg, withgrad, maxiter, zerotol)
}

#' #' @export
#' sinkhorn_parallel <- function(A, B, C, reg, withgrad = FALSE,
#'                               maxIter = 1000, zeroTol = 1e-6) {
#'   if (!is.matrix(A)) stop("A must be a numeric vector!")
#'   if (!is.matrix(B)) stop("B must be a numeric vector!")
#'   if (!is.matrix(C)) stop("C must be a numeric matrix!")
#'
#'   if (withgrad) {
#'     sol <- sinkhorn_parallel_withjac_cpp(A, B, C, reg, maxIter, zeroTol)
#'   } else {
#'     sol <- sinkhorn_parallel_withoutjac_cpp(A, B, C, reg, maxIter, zeroTol)
#'   }
#'
#'   U <- sol$U
#'   V <- sol$V
#'   K <- sol$K
#'   append(
#'     list(P = Map(\(s) diag(U[,s]) %*% K %*% diag(V[,s]), 1:ncol(A))),
#'     sol
#'   )
#' }
