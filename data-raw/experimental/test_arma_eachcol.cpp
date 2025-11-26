
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

using namespace arma;

// [[Rcpp::export]]
arma::mat test1(arma::mat& m, arma::vec& v) {
  mat out = m.each_col() % v;
  return out;
}

// [[Rcpp::export]]
arma::mat test2(arma::mat& m, arma::vec& v) {
  mat out = diagmat(v) * m;
  return out;
}
