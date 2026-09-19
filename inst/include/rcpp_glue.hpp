// Conversions between R objects and la::Vec / la::Mat.
// Only the interface files (sinkhorn.cpp, barycenter.cpp, ...) include this;
// the algorithm files stay free of R headers apart from Rcpp::message and
// Rcpp::checkUserInterrupt.

#ifndef RWIG_RCPP_GLUE_H
#define RWIG_RCPP_GLUE_H

#include <Rcpp.h>

#include "linalg.hpp"

namespace la {

// both coerce integer/logical input to double, like Rcpp::as<arma::vec> did
inline Vec vec_from_R(SEXP x) {
  Rcpp::NumericVector nv(x);
  return Vec(nv.begin(), (idx)nv.size());
}

inline Mat mat_from_R(SEXP x) {
  Rcpp::NumericMatrix nm(x);
  return Mat(nm.begin(), (idx)nm.nrow(), (idx)nm.ncol());
}

inline Rcpp::NumericVector to_R(const Vec &v) {
  Rcpp::NumericVector out((R_xlen_t)v.size());
  std::copy(v.data(), v.data() + v.size(), out.begin());
  return out;
}

inline Rcpp::NumericMatrix to_R(const Mat &m) {
  Rcpp::NumericMatrix out((int)m.nrow(), (int)m.ncol());
  std::copy(m.data(), m.data() + m.size(), out.begin());
  return out;
}

} // namespace la

#endif // RWIG_RCPP_GLUE_H
