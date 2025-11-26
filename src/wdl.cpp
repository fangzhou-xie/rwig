
// this is the file defining the functions exporting to R side
// WDL algos

#define ARMA_DONT_USE_OPENMP

// #include <iostream> // std::cout
// #include "R_ext/Print.h"    // for REprintf

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "wdl_impl.hpp" // header for the WDL definition
// #include "ctrack.hpp"

// using namespace arma;
// using namespace cpp11;
// using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable;

/////////////////////////////////////////////////////////////////////
// R interfaces for WDL
/////////////////////////////////////////////////////////////////////

// TODO: add warning for the non-converging Barycenter?

// [[Rcpp::export]]
Rcpp::List wdl_cpp(
    const arma::mat& Y,    // document matrix N * M
    const arma::mat& C,    // cost matrix N * N
    const double reg,
    const int S,
    const int n_threads,
    const int batch_size,
    const int epochs,
    int sinkhorn_mode = 1,
    const int max_iter = 1000, const double zero_tol = 1e-6,
    const int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    const double beta1 = .9, const double beta2 = .999,
    const double eps = 1e-8,
    const bool verbose = false
) {
  // const int rng_seed = 123,
  // const double sinkhorn_mode_threshold = .1,

  // check sinkhorn mode
  if ((sinkhorn_mode != 1) && (sinkhorn_mode != 2)) {
    // cpp11::stop("Sinkhorn mode not supported");
    Rcpp::stop("Sinkhorn mode not supported");
  }

  // check optimizer mode
  if ((optimizer != 0) && (optimizer != 1) && (optimizer != 2)) {
    // cpp11::stop("optimizer must be: 0, 1, 2!");
    Rcpp::stop("optimizer must be: 0, 1, 2!");
  }

  // init the WDL class
  WassersteinDictionaryLearning wdl(
    batch_size,epochs,n_threads,
    sinkhorn_mode,max_iter,zero_tol,
    optimizer,eta,gamma,beta1,beta2,eps,verbose
  );

  //  with data
  wdl.init_data(Y, C, reg, S);

  // start the actual WDL computation
  wdl.compute();

  // ctrack::result_print();

  return Rcpp::List::create(
    Rcpp::Named("A") = wdl.A,
    Rcpp::Named("W") = wdl.W,
    Rcpp::Named("Yhat") = wdl.Yhat
  );
}

// [[cpp11::register]]
// writable::list wdl_cpp(
//     const doubles_matrix<>& Y,  // document matrix N * M
//     const doubles_matrix<>& C,  // cost matrix N * N
//     const double reg,
//     const int S,
//     const int threads,
//     const int batch_size,
//     const int epochs,
//     int sinkhorn_mode = 1,
//     // const double sinkhorn_mode_threshold = .1,
//     const int max_iter = 1000, const double zero_tol = 1e-6,
//     const int optimizer = 2,
//     const double eta = .001, const double gamma = .01,
//     const double beta1 = .9, const double beta2 = .999,
//     const double eps = 1e-8,
//     const int rng_seed = 123, const int verbose = 0
// ) {
//   // convert R matrices into arma ones
//   mat Y_ = as_Mat(Y);
//   mat C_ = as_Mat(C);
//
//   // check sinkhorn mode
//   if ((sinkhorn_mode != 1) && (sinkhorn_mode != 2)) {
//     cpp11::stop("Sinkhorn mode not supported");
//   }
//
//   // check optimizer mode
//   if ((optimizer != 0) && (optimizer != 1) && (optimizer != 2)) {
//     cpp11::stop("optimizer must be: 0, 1, 2!");
//   }
//
//   // init the WDL class
//   WassersteinDictionaryLearning wdl(
//       batch_size,epochs,sinkhorn_mode,max_iter,zero_tol,
//       optimizer,eta,gamma,beta1,beta2,eps,rng_seed,verbose
//   );
//
//   //  with data
//   wdl.init_data(Y_, C_, reg, S);
//
//   // start the actual WDL computation
//   wdl.compute(threads);
//
//   // ctrack::result_print();
//
//   // output the optimized topics A and weights W
//   writable::list res;
//   res.push_back({"A"_nm = as_doubles_matrix(wdl.A)});
//   res.push_back({"W"_nm = as_doubles_matrix(wdl.W)});
//   res.push_back({"Yhat"_nm = as_doubles_matrix(wdl.Yhat)});
//   return res;
// }
