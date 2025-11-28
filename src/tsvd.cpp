
// implement the tsvd algorithm in c++

// #ifndef ARMA_DONT_USE_OPENMP
// #define ARMA_DONT_USE_OPENMP
// #endif

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>
// using namespace arma;

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat tsvd_cpp(const arma::mat& M,
                   const int k, const int flip_sign) {
  // flip_sign: how to determine the sign of the vectors
  // flip_sign = 0, auto
  // flip_sign = 1, sklearn
  // flip_sign = 2, none

  // Reference:
  //
  // auto mode:
  // Bro, R., Acar, E., & Kolda, T. G. (2008).
  // Resolving the sign ambiguity in the singular value decomposition.
  // Journal of Chemometrics, 22(2), 135–140. https://doi.org/10.1002/cem.1122
  //
  // sklearn: max entry per column should be positive
  // https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_truncated_svd.py#L133
  // https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/extmath.py#L433

  // convert input
  // mat M = as_Mat(MR);

  // output
  arma::mat Msvd{arma::mat(arma::size(M), arma::fill::zeros)};

  // first do the full SVD
  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, M);

  if (flip_sign == 0) {
    // auto mode
    arma::mat Y = M - U.cols(0,k-1) * diagmat(s.rows(0,k-1)) * (V.cols(0,k-1)).t();
    double sk_left, sk_right;
    double uTy = 0.;
    double vTy = 0.;

    for (int K = 0; K < k; ++K) {

      sk_left = 0.;
      for (size_t j = 0; j < Y.n_cols; ++j) {
        uTy = arma::accu(U.col(K) % Y.col(j));
        sk_left += arma::sign(uTy) * pow(uTy, 2);
      }

      sk_right = 0.;
      for (size_t i = 0; i < Y.n_rows; ++i) {
        vTy = arma::accu(V.col(K) % Y.row(i).t());
        sk_right += arma::sign(vTy) * pow(vTy, 2);
      }

      if (sk_left * sk_right < 0) {
        if (fabs(sk_left) < fabs(sk_right)) {
          sk_left *= -1;
        } else {
          sk_right *= -1;
        }
      }

      U.col(K) *= arma::sign(sk_left);
      V.col(K) *= arma::sign(sk_right);
    }
    Msvd = U.cols(0,k-1) * diagmat(s.rows(0,k-1));

  } else if (flip_sign == 1) {
    Msvd = U.cols(0,k-1) * diagmat(s.rows(0,k-1));

    for (size_t i = 0; i < Msvd.n_cols; ++i) {
      if (Msvd.col(i)(abs(Msvd.col(i)).index_max()) < 0) {
        Msvd.col(i) *= -1;
      }
    }
  } else {
    Msvd = U.cols(0,k-1) * diagmat(s.rows(0,k-1));
  }
  // return as_doubles_matrix(Msvd);
  return Msvd;
}
