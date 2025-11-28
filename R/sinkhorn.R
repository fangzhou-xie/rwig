# call the functions from sinkhorn.cpp
# for the algorithms mentioned in Section 3 & 4

#' Sinkhorn algorithm
#'
#' @family Sinkhorn algorithms
#'
#' @description
#' Sinkhorn algorithm to solve entropy-regularized Optimal Transport
#' problems.
#'
#' @details
#' This is the general function to solve the OT problem,
#' and it will use either vanilla (`method = "vanilla"`) or
#' log-stabilized Sinkhorn algorithm (`method = "log"`)
#' for solving the problem.
#'
#' @references
#'
#' Peyré, G., & Cuturi, M. (2019). Computational Optimal Transport:
#' With Applications to Data Science.
#' *Foundations and Trends® in Machine Learning*, 11(5–6), 355–607.
#' https://doi.org/10.1561/2200000073
#'
#' Xie, F. (2025). Deriving the Gradients of Some Popular Optimal
#' Transport Algorithms (No. arXiv:2504.08722). *arXiv*.
#' https://doi.org/10.48550/arXiv.2504.08722
#'
#' @param a numeric vector, source discrete density (probability vector)
#' @param b numeric vector, target discrete density (probability vector)
#' @param C numeric matrix, cost matrix between source and target
#' @param sinkhorn_control list, control parameters for the computation
#' * reg double, regularization parameter (default = .1)
#' * with_grad: bool, whether to calculate the gradient w.r.t. a
#' * n_threads: int, number of threads (only used for `method = "log"`,
#' ignored by the `method = "vanilla"`, default = 0)
#' * method: character, which method to use: "auto", "vanilla", "log"
#' "auto" with try to calculate minimum value of the Gibbs kernel K
#' and switch to `method = "log"` if the minimum value is less than `threshold`
#' (default = "auto")
#' * threshold: double, threshold value below which "auto" method
#' will default to `method = "log"` for stablized computation in log-domain
#' (default = .1)
#' * max_iter: int, maximum iteration of \code{\link{sinkhorn}} algorithm
#' (default = 1000)
#' * zero_tol: double, determine covergence (default = 1e-6)
#' * verbose: int, print out debug info for the algorithm for every
#' `verbose` iteration (default to 0, i.e. not printing anything)
#'
#' @returns list of results
#' * P: optimal coupling matrix
#' * grad_a: gradient of loss w.r.t. `a` (only with `with_grad = TRUE`)
#' * u, v: scaling vectors
#' * loss: regularized loss
#' * iter: iterations of the algorithm
#' * err: condition for convergence
#' * return_status: 0 (convergence), 1 (max iteration reached), 2 (other)
#'
#' @examples
#' # simple sinkhorn example
#' a <- c(.3, .4, .1, .1, .1)
#' b <- c(.4, .5, .1)
#' C <- rbind(
#'   c(.1, .2, .3),
#'   c(.2, .3, .4),
#'   c(.4, .3, .2),
#'   c(.3, .2, .1),
#'   c(.5, .5, .4)
#' )
#' reg <- .1
#' sol <- sinkhorn(a, b, C, sinkhorn_control = list(reg = reg, verbose = 0))
#'
#' # you can also supply arguments to control the computation
#' # for example, calculate the gradient w.r.t. a
#' sol <- sinkhorn(a, b, C,
#' sinkhorn_control = list(reg = reg, with_grad = TRUE, verbose = 0))
#'
#' @importFrom Rcpp evalCpp
#' @export
sinkhorn <- function(
  a,
  b,
  C,
  sinkhorn_control = list(
    reg = .1,
    with_grad = FALSE,
    method = "auto",
    threshold = .1,
    max_iter = 1000L,
    zero_tol = 1e-6,
    verbose = 0L
  )
) {
  if (!is.vector(a)) {
    stop("a must be a numeric vector!")
  }
  if (!is.vector(b)) {
    stop("b must be a numeric vector!")
  }
  if (!is.matrix(C)) {
    stop("C must be a numeric matrix!")
  }
  if ((length(a) != nrow(C)) | (length(b) != ncol(C))) {
    stop("a, b, C don't match in dimension!")
  }

  # auto complete the arguments
  sinkhorn_control <- check_sinkhorn_args(sinkhorn_control)
  reg <- sinkhorn_control$reg

  if (!is.numeric(reg) || (length(reg) != 1)) {
    stop("reg must be a single value")
  }

  # calculating the minimum value of the Gibbs kernel K
  k1 <- exp(-min(C) / reg)
  k2 <- exp(-max(C) / reg)

  # auto-switch the best sinkhorn algo (between vanilla and log)
  # alert user for the method chosen if verbose
  if (sinkhorn_control$method == "auto") {
    if (min(k1, k2) < sinkhorn_control$threshold) {
      sinkhorn_control$method <- "log"

      if (sinkhorn_control$verbose) {
        message("`method` is automatically switched to \"log\"")
      }
    } else {
      sinkhorn_control$method <- "vanilla"

      if (sinkhorn_control$verbose) {
        message("`method` is automatically switch to \"vanilla\"")
      }
    }
  }

  if (sinkhorn_control$method == "log") {
    # use log sinkhorn
    sol <- sinkhorn_log_cpp(
      a,
      b,
      C,
      reg,
      sinkhorn_control$with_grad,
      sinkhorn_control$n_threads,
      sinkhorn_control$max_iter,
      sinkhorn_control$zero_tol,
      sinkhorn_control$verbose
    )
  } else if (sinkhorn_control$method == "vanilla") {
    # use vanilla sinkhorn
    sol <- sinkhorn_vanilla_cpp(
      a,
      b,
      C,
      reg,
      sinkhorn_control$with_grad,
      sinkhorn_control$max_iter,
      sinkhorn_control$zero_tol,
      sinkhorn_control$verbose
    )
  } else {
    stop("method is not supported!")
  }
  sol
}

# sinkhorn_vanilla <- function(
#   a,
#   b,
#   C,
#   reg,
#   withgrad = FALSE,
#   maxiter = 1000,
#   zerotol = 1e-6,
#   verbose = 0L
# ) {
#   if (!is.vector(a)) {
#     stop("a must be a numeric vector!")
#   }
#   if (!is.vector(b)) {
#     stop("b must be a numeric vector!")
#   }
#   if (!is.matrix(C)) {
#     stop("C must be a numeric matrix!")
#   }
#   if ((length(a) != nrow(C)) | (length(b) != ncol(C))) {
#     stop("a, b, C don't match in dimension!")
#   }
#
#   # call the c++ routine
#   sinkhorn_vanilla_cpp(a, b, C, reg, withgrad, maxiter, zerotol)
# }

# sinkhorn_log <- function(
#   a,
#   b,
#   C,
#   reg,
#   withgrad = FALSE,
#   n_threads = 0,
#   maxiter = 1000,
#   zerotol = 1e-6,
#   verbose = 0L
# ) {
#   if (!is.vector(a)) {
#     stop("a must be a numeric vector!")
#   }
#   if (!is.vector(b)) {
#     stop("b must be a numeric vector!")
#   }
#   if (!is.matrix(C)) {
#     stop("C must be a numeric matrix!")
#   }
#   if ((length(a) != nrow(C)) | (length(b) != ncol(C))) {
#     stop("a, b, C don't match in dimension!")
#   }
#
#   # call the c++ routine
#   sinkhorn_log_cpp(a, b, C, reg, withgrad, n_threads, maxiter, zerotol)
# }

# sinkhorn_parallel <- function(A, B, C, reg, withgrad = FALSE,
#                               maxIter = 1000, zeroTol = 1e-6) {
#   if (!is.matrix(A)) stop("A must be a numeric vector!")
#   if (!is.matrix(B)) stop("B must be a numeric vector!")
#   if (!is.matrix(C)) stop("C must be a numeric matrix!")
#
#   if (withgrad) {
#     sol <- sinkhorn_parallel_withjac_cpp(A, B, C, reg, maxIter, zeroTol)
#   } else {
#     sol <- sinkhorn_parallel_withoutjac_cpp(A, B, C, reg, maxIter, zeroTol)
#   }
#
#   U <- sol$U
#   V <- sol$V
#   K <- sol$K
#   append(
#     list(P = Map(\(s) diag(U[,s]) %*% K %*% diag(V[,s]), 1:ncol(A))),
#     sol
#   )
# }
