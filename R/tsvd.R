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

  # M_svd <- svd(M)
  # U <- M_svd$u
  # S <- M_svd$d
  # V <- M_svd$v
  #
  # # flip sign: auto or sklearn
  # # auto: flip sign for sign ambiguity
  # # Reference:
  #   # Bro, R., Acar, E., & Kolda, T. G. (2008).
  #   # Resolving the sign ambiguity in the singular value decomposition.
  #   # Journal of Chemometrics, 22(2), 135–140. https://doi.org/10.1002/cem.1122
  # # sklearn: max entry per column should be positive
  # # Reference:
  # # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_truncated_svd.py#L133
  # # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py#L433
  #
  # if (flip_sign == "auto") {
  #
  #   Y <- M - U[,1:k] %*% diag(S[1:k], k, k) %*% t(V[,1:k])
  #   for (K in 1:k) {
  #     sk_left <- 0
  #     for (j in 1:ncol(Y)) {
  #       uTy <- sum(U[,K] * Y[,j])
  #       sk_left <- sk_left + sign(uTy) * uTy^2
  #     }
  #
  #     sk_right <- 0
  #     for (i in 1:nrow(Y)) {
  #       vTy <- sum(V[,K] * Y[i,])
  #       sk_right <- sk_right + sign(vTy) * vTy^2
  #     }
  #
  #     if (sk_left * sk_right < 0) {
  #       if (abs(sk_left) < abs(sk_right)) {
  #         sk_left <- -sk_left
  #       } else {
  #         sk_right <- -sk_right
  #       }
  #     }
  #
  #     U[,K] <- sign(sk_left) * U[,K]
  #     V[,K] <- sign(sk_right) * V[,K]
  #
  #   }
  #   M_tsvd <- U[,1:k] %*% diag(S[1:k], k, k)
  #
  # } else if (flip_sign == "sklearn") {
  #
  #   M_tsvd <- U[,1:k] %*% diag(S[1:k], k, k)
  #
  #   for (i in 1:k) {
  #     x <- M_tsvd[,i]
  #     if (x[which.max(abs(x))] < 0) {
  #       M_tsvd[,i] <- -M_tsvd[,i]
  #     }
  #   }
  # } else if (flip_sign == "none") {
  #   M_tsvd <- U[,1:k] %*% diag(S[1:k], k, k)
  # }
  #
  # M_tsvd
  flip_sign_int <- switch(flip_sign, "auto" = 0L, "sklearn" = 1L, "non" = 2L)
  if (!(flip_sign_int %in% c(0L, 1L, 2L))) {
    stop_msg <- paste0(
      "Truncated SVD method not supported!",
      "Must be one of the following: \"auto\", \"sklearn\", or \"non\"",
      collapse = " "
    )
    stop(stop_msg)
  }

  tsvd_cpp(M, k, flip_sign_int)
}
