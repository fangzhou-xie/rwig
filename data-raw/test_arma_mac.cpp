
// #include <RcppArmadillo/Lighter>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat test_func(const int m, const int n) {
  auto out = arma::mat(m,n);
  return out;
}
