// common headers (armadillo)

#ifndef WIG_COMMON_H
#define WIG_COMMON_H

#ifndef ARMA_DONT_USE_OPENMP
#define ARMA_DONT_USE_OPENMP
#endif

#define RCPP_NO_SUGAR
#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

#include <RcppArmadillo.h>
// #include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]


#endif // WIG_COMMON_H
