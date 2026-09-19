
// implement the tsvd algorithm in c++ (thin SVD via LAPACK dgesdd)

#include <cmath>

#include "rcpp_glue.hpp"

static inline double sgn(double x) { return (x > 0) - (x < 0); }

// [[Rcpp::export]]
Rcpp::NumericMatrix tsvd_cpp(const SEXP &MR, const int k, const int flip_sign) {
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

  la::Mat M = la::mat_from_R(MR);
  const int m = (int)M.nrow(), n = (int)M.ncol();

  // thin SVD: M = U diag(s) Vt
  la::Mat U, Vt;
  la::Vec s;
  const int info = la::gesdd_thin(M, U, s, Vt);
  if (info != 0) {
    Rcpp::stop("SVD failed to converge (LAPACK dgesdd info = %d)", info);
  }

  // clamp k to available singular values
  const int kk = std::min(k, (int)s.size());

  // Msvd = U[:, 1:kk] diag(s[1:kk])   (m x kk), signs fixed below
  la::Mat Msvd(m, kk);
  auto fill_Msvd = [&]() {
    for (int j = 0; j < kk; ++j) {
      const double *Uj = U.col(j);
      double *out = Msvd.col(j);
      for (int i = 0; i < m; ++i) out[i] = Uj[i] * s[j];
    }
  };

  if (flip_sign == 0) {
    // auto mode: residual Y = M - U_k diag(s_k) V_k^T
    fill_Msvd();
    la::Mat Y(M);
    la::gemm(false, false, m, n, kk, -1.0, Msvd.data(), m, Vt.data(),
             (int)Vt.nrow(), 1.0, Y.data(), m);
    // G = U_k^T Y (kk x n): G(K, j) = u_K . Y[:, j]
    la::Mat G(kk, n);
    la::gemm(true, false, kk, n, m, 1.0, U.data(), m, Y.data(), m, 0.0,
             G.data(), kk);
    // H = Y V_k (m x kk): H(i, K) = v_K . Y[i, :]
    la::Mat H(m, kk);
    la::gemm(false, true, m, kk, n, 1.0, Y.data(), m, Vt.data(), (int)Vt.nrow(),
             0.0, H.data(), m);

    for (int K = 0; K < kk; ++K) {
      double sk_left = 0.;
      for (int j = 0; j < n; ++j) {
        const double uTy = G(K, j);
        sk_left += sgn(uTy) * uTy * uTy;
      }
      double sk_right = 0.;
      for (int i = 0; i < m; ++i) {
        const double vTy = H(i, K);
        sk_right += sgn(vTy) * vTy * vTy;
      }
      if (sk_left * sk_right < 0) {
        if (std::fabs(sk_left) < std::fabs(sk_right)) {
          sk_left *= -1;
        } else {
          sk_right *= -1;
        }
      }
      // U.col(K) *= sign(sk_left)  (V.col(K) *= sign(sk_right) is unused)
      const double f = sgn(sk_left);
      double *UK = U.col(K);
      for (int i = 0; i < m; ++i) UK[i] *= f;
    }
    fill_Msvd();

  } else if (flip_sign == 1) {
    fill_Msvd();
    for (int j = 0; j < kk; ++j) {
      double *col = Msvd.col(j);
      int imax = 0;
      double amax = std::fabs(col[0]);
      for (int i = 1; i < m; ++i) {
        const double a = std::fabs(col[i]);
        if (a > amax) {
          amax = a;
          imax = i;
        }
      }
      if (col[imax] < 0) {
        for (int i = 0; i < m; ++i) col[i] *= -1;
      }
    }
  } else {
    fill_Msvd();
  }
  return la::to_R(Msvd);
}
