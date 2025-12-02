# call the functions from barycenter.cpp
# for the algorithms mentioned in Section 5 & 6

#' Barycenter algorithm
#'
#' @family Barycenter algorithms
#'
#' @description
#' Barycenter algorithm to solve for entropy-regularized Optimal Transport
#' Barycenter problems.
#' For a more detailed explaination, please refer to
#' \code{vignette("barycenter")}.
#'
#' @details
#' This is the general function to solve OT Barycenter problem,
#' and it will use either parallel (`method = "parallel"`) or
#' log-stablized Barycenter algorithm (`method = "log"`)
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
#' @param A numeric matrix, source discrete densities (M * S)
#' @param C numeric matrix, cost matrix between source and target (M * N)
#' @param w numeric vector, weight vector (S)
#' @param b_ext numeric vector, only used to calculate quadratic loss
#' against the computed barycenter (default = NULL)
#' @param barycenter_control list, control parameters for the computation
#' * reg: double, regularization parameter (default = .1)
#' * with_grad: bool, whether to calculate the gradient w.r.t. A
#' * n_threads: int, number of threads (only used for `method = "log"`,
#' ignored by `method = "parallel"`, default = 0)
#' * threshold: double, threshold value below which "auto" method
#' will default to `method = "log"` for stablized computation in log-domain
#' (default = .1)
#' * max_iter: int, maximum iteration of \code{\link{barycenter}} algorithm
#' (default = 1000)
#' * zero_tol: double, determine covergence (default = 1e-6)
#' * verbose: int, print out debug info for the algorithm for every
#' `verbose` iteration (default to 0, i.e. not printing anything)
#'
#' @returns list of results
#' * b: numeric vector, computed barycenter
#' * grad_A: gradient w.r.t. A (only with `with_grad = TRUE`)
#' * grad_w: gradient w.r.t. w (only with `with_grad = TRUE`)
#' * loss: double, quadratic loss between `b` and `b_ext`
#' (only with `with_grad = TRUE`)
#' * U, V: scaling variables for the Sinkhorn algorithm
#' (only with `method = "parallel"`)
#' * F, G: scaling variables for the Sinkhorn algorithm
#' (only with `method = "log"`)
#' * iter: iterations of the algorithm
#' * err: condition for convergence
#' * return_status: 0 (convergence), 1 (max iteration reached), 2 (other)
#'
#' @examples
#' A <- rbind(c(.3, .2), c(.2, .1), c(.1, .2), c(.1, .1), c(.3, .4))
#' C <- rbind(
#'   c(.1, .2, .3, .4, .5),
#'   c(.2, .3, .4, .3, .2),
#'   c(.4, .3, .2, .1, .2),
#'   c(.3, .2, .1, .2, .5),
#'   c(.5, .5, .4, .0, .2)
#' )
#' w <- c(.4, .6)
#' b <- c(.2, .2, .2, .2, .2)
#' reg <- .1
#'
#' # simple barycenter example
#' sol <- barycenter(A, C, w, barycenter_control = list(reg = reg))
#'
#' # you can also supply arguments to control the computation
#' # for example, including the loss and gradient w.r.t. `A`
#' sol <- barycenter(A, C, w, b, barycenter_control = list(reg = reg, with_grad = TRUE))
#'
#' @seealso
#' \code{vignette("gradient")},
#' \code{vignette("threading")}
#'
#' @importFrom Rcpp evalCpp
#' @export
barycenter <- function(
  A,
  C,
  w,
  b_ext = NULL,
  barycenter_control = list(
    reg = .1,
    with_grad = FALSE,
    n_threads = 0,
    method = "auto",
    threshold = .1,
    max_iter = 1000,
    zero_tol = 1e-6,
    verbose = 0
  )
) {
  # check the input types
  if (!is.matrix(A)) {
    stop("A must be a numeric matrix!")
  }
  if (!is.matrix(C)) {
    stop("C must be a numeric matrix!")
  }
  if (!is.vector(w)) {
    stop("w must be a numeric vector!")
  }
  if (nrow(A) != nrow(C)) {
    stop("number of rows of A and C must equal!")
  }

  # fill the args if not set
  barycenter_control <- check_barycenter_args(barycenter_control)
  reg <- barycenter_control$reg

  if (!is.numeric(reg) || (length(reg) != 1)) {
    stop("reg must be a single value")
  }

  # auto-switch the best barycenter algo (between vanilla and log)
  k1 <- exp(-min(C) / reg)
  k2 <- exp(-max(C) / reg)

  # auto-switch the best sinkhorn algo (between vanilla and log)
  # alert user for the method chosen
  if (barycenter_control$verbose) {
    if (barycenter_control$method == "auto") {
      if (min(k1, k2) < barycenter_control$threshold) {
        message("`method` is automatically switched to \"log\"")
        barycenter_control$method <- "log"
      } else {
        message("`method` is automatically switch to \"parallel\"")
        barycenter_control$method <- "parallel"
      }
    }
  }

  # check `b_ext` and add it if not supplied
  if (barycenter_control$with_grad && is.null(b_ext)) {
    stop("you must supply the `b_ext` with `with_grad = TRUE`!")
  } else if (!(barycenter_control$with_grad) && !is.null(b_ext)) {
    warning(
      paste0(
        "you have supplied `b_ext`,",
        "but it will not be used as `with_grad = FALSE`!",
        collapse = " "
      )
    )
  } else if (!(barycenter_control$with_grad) && is.null(b_ext)) {
    b_ext <- rep(0, 2) # only to pass to cpp function without using it
  }

  # calculating the minimum value of the Gibbs kernel K
  k1 <- exp(-min(C) / reg)
  k2 <- exp(-max(C) / reg)

  # auto-switch the best barycenter algo (between parallel and log)
  # alert user for the method chosen if verbose
  if (barycenter_control$method == "auto") {
    if (min(k1, k2) < barycenter_control$threshold) {
      barycenter_control$method <- "log"

      if (barycenter_control$verbose) {
        message("`method` is automatically switched to \"log\"")
      }
    } else {
      barycenter_control$method <- "parallel"

      if (barycenter_control$verbose) {
        message("`method` is automatically switch to \"parallel\"")
      }
    }
  }

  if (barycenter_control$method == "log") {
    # use log barycenter
    sol <- barycenter_log_cpp(
      A,
      C,
      w,
      reg,
      b_ext,
      barycenter_control$with_grad,
      barycenter_control$n_threads,
      barycenter_control$max_iter,
      barycenter_control$zero_tol,
      barycenter_control$verbose
    )
  } else if (barycenter_control$method == "parallel") {
    sol <- barycenter_parallel_cpp(
      A,
      C,
      w,
      reg,
      b_ext,
      barycenter_control$with_grad,
      barycenter_control$max_iter,
      barycenter_control$zero_tol,
      barycenter_control$verbose
    )
  } else {
    stop("method is not supported!")
  }

  # add method into the results
  sol$method <- barycenter_control$method
  sol
}

# barycenter_parallel <- function(
#   A,
#   C,
#   w,
#   reg,
#   b_ext = NULL,
#   withgrad = FALSE,
#   maxiter = 1000,
#   zerotol = 1e-6,
#   verbose = 0
# ) {
#   # check the input types
#   if (!is.matrix(A)) {
#     stop("A must be a numeric matrix!")
#   }
#   if (!is.matrix(C)) {
#     stop("C must be a numeric matrix!")
#   }
#   if (!is.vector(w)) {
#     stop("w must be a numeric vector!")
#   }
#   if (nrow(A) != nrow(C)) {
#     stop("number of rows of A and C must equal!")
#   }
#
#   if (withgrad && is.null(b_ext)) {
#     stop("you must supply the `b_ext` with `withgrad = TRUE`!")
#   } else if (!withgrad && !is.null(b_ext)) {
#     warning(
#       "you have supplied `b_ext`, but it will not be used as `withgrad = FALSE`!"
#     )
#   } else if (!withgrad && is.null(b_ext)) {
#     b_ext <- rep(0, 2)
#   }
#
#   # call the C++ routine
#   barycenter_parallel_cpp(
#     A,
#     C,
#     w,
#     reg,
#     b_ext,
#     withgrad,
#     maxiter,
#     zerotol,
#     verbose
#   )
# }

# barycenter_log <- function(
#   A,
#   C,
#   w,
#   reg,
#   b_ext = NULL,
#   withgrad = FALSE,
#   n_threads = 0,
#   maxiter = 1000,
#   zerotol = 1e-6,
#   verbose = 0
# ) {
#   # check the input types
#   if (!is.matrix(A)) {
#     stop("A must be a numeric matrix!")
#   }
#   if (!is.matrix(C)) {
#     stop("C must be a numeric matrix!")
#   }
#   if (!is.vector(w)) {
#     stop("w must be a numeric vector!")
#   }
#   if (nrow(A) != nrow(C)) {
#     stop("number of rows of A and C must equal!")
#   }
#
#   if (withgrad && is.null(b_ext)) {
#     stop("you must supply the `b_ext` with `withgrad = TRUE`!")
#   } else if (!withgrad && !is.null(b_ext)) {
#     warning(
#       "you have supplied `b_ext`, but it will not be used as `withgrad = FALSE`!"
#     )
#   } else if (!withgrad && is.null(b_ext)) {
#     b_ext <- rep(0, 2)
#   }
#
#   # call the C++ routine
#   barycenter_log_cpp(
#     A,
#     C,
#     w,
#     reg,
#     b_ext,
#     withgrad,
#     n_threads,
#     maxiter,
#     zerotol,
#     verbose
#   )
# }
