#' Truncated SVD
#'
#' @description
#' Truncated Singular Value Decomposition algorithm
#'
#' @details
#' Compute the truncated SVD for dimension reduction.
#' Note that SVD suffers from "sign indeterminacy," which means
#' that the signs of the output vectors could depend on the algorithm
#' and random state.
#' Two variants of "sign flipping methods" are implemented here,
#' one following the sklearn implementation on Truncated SVD,
#' another by Bro et al. (2008).
#'
#' @param M matrix, data to be analyzed
#' @param k int, number of columns/features to be kept
#' @param flip_sign character, one of the following: "auto", "sklearn", "none"
#'
#' @return matrix after dimension reduction
#'
#' @examples
#' A <- rbind(c(1,3), c(2,4))
#' tsvd(A)
#'
#' @references
#'
#' https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
#'
#' Bro, R., Acar, E., & Kolda, T. G. (2008).
#' Resolving the sign ambiguity in the singular value decomposition.
#' Journal of Chemometrics, 22(2), 135–140. https://doi.org/10.1002/cem.1122
#'
#' @seealso
#' [Truncated SVD (sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
#' \code{vignette("tsvd")}
#'
#' @export
tsvd <- function(M, k = 1, flip_sign = c("auto", "sklearn", "none")) {
  flip_sign <- match.arg(flip_sign)

  # ensure min features: 2
  if (ncol(M) < 2) {
    stop("Features less than 2! Check number of columns of 'M'")
  }
  if (ncol(M) <= k) {
    stop("Dimension 'k' should be less than number of columns of 'M'")
  }

  flip_sign_int <- switch(flip_sign, "auto" = 0L, "sklearn" = 1L, "none" = 2L)

  if (!(flip_sign_int %in% c(0L, 1L, 2L))) {
    stop_msg <- paste0(
      "Truncated SVD method not supported!",
      "Must be one of the following: \"auto\", \"sklearn\", or \"none\"",
      collapse = " "
    )
    stop(stop_msg)
  }

  tsvd_cpp(M, k, flip_sign_int)
}
