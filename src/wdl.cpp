
// this is the file defining the functions exporting to R side
// WDL algos

// #include <iostream> // std::cout
// #include "R_ext/Print.h"    // for REprintf

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

Rcpp::List wdl_cpp_cpu(const SEXP &Y, // document matrix N * M
                       const SEXP &C, // cost matrix N * N
                       const double reg, const int S, const int n_threads,
                       const int batch_size, const int epochs,
                       int sinkhorn_mode = 1, const int max_iter = 1000,
                       const double zero_tol = 1e-6, const int optimizer = 2,
                       const double eta = .001, const double gamma = .01,
                       const double beta1 = .9, const double beta2 = .999,
                       const double eps = 1e-8, const bool verbose = false) {
  // convert R vectors/matrices into arma ones
  arma::mat Y_{Rcpp::as<arma::mat>(Y)};
  arma::mat C_{Rcpp::as<arma::mat>(C)};

  // init the WDL class
  WassersteinDictionaryLearning wdl(
      batch_size, epochs, n_threads, sinkhorn_mode, max_iter, zero_tol,
      optimizer, eta, gamma, beta1, beta2, eps, verbose);

  //  with data
  wdl.init_data(Y_, C_, reg, S);

  // start the actual WDL computation
  wdl.compute();

  // ctrack::result_print();

  return Rcpp::List::create(Rcpp::Named("A") = wdl.A, Rcpp::Named("W") = wdl.W,
                            Rcpp::Named("Yhat") = wdl.Yhat);
}

// only have the CUDA version when they are detected
#ifdef HAVE_CUBLAS
#ifdef HAVE_CUDA_RUNTIME

#include "cuda_interface.cuh"

// TODO: implement wdl_cpp_cuda
Rcpp::List wdl_cpp_cuda(const SEXP &Y, // document matrix N * M
                        const SEXP &C, // cost matrix N * N
                        const double reg, const int S, const int n_threads,
                        const int batch_size, const int epochs,
                        int sinkhorn_mode = 1, const int max_iter = 1000,
                        const double zero_tol = 1e-6, const int optimizer = 2,
                        const double eta = .001, const double gamma = .01,
                        const double beta1 = .9, const double beta2 = .999,
                        const double eps = 1e-8, const bool verbose = false) {
  Rcpp::stop("CUDA version of WDL not implemented yet!");
  Rcpp::List res;
  return res;
}

#endif
#endif

/*
Interfaces for the R side
*/

// [[Rcpp::export]]
Rcpp::List wdl_cpp(const SEXP &Y, // document matrix N * M
                   const SEXP &C, // cost matrix N * N
                   const double reg, const int S, const int n_threads,
                   const int batch_size, const int epochs,
                   int sinkhorn_mode = 1, bool usecuda = true,
                   const int max_iter = 1000, const double zero_tol = 1e-6,
                   const int optimizer = 2, const double eta = .001,
                   const double gamma = .01, const double beta1 = .9,
                   const double beta2 = .999, const double eps = 1e-8,
                   const bool verbose = false) {

  // check sinkhorn mode
  if ((sinkhorn_mode != 1) && (sinkhorn_mode != 2)) {
    Rcpp::stop("Sinkhorn mode not supported");
  }

  // check optimizer mode
  if ((optimizer != 0) && (optimizer != 1) && (optimizer != 2)) {
    Rcpp::stop("optimizer must be: 0, 1, 2!");
  }

  Rcpp::List res;

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  if (usecuda) {
    res = wdl_cpp_cuda(Y, C, reg, S, n_threads, batch_size, epochs,
                       sinkhorn_mode, max_iter, zero_tol, optimizer, eta, gamma,
                       beta1, beta2, eps, verbose);
  } else {
    res = wdl_cpp_cpu(Y, C, reg, S, n_threads, batch_size, epochs,
                      sinkhorn_mode, max_iter, zero_tol, optimizer, eta, gamma,
                      beta1, beta2, eps, verbose);
  }
#else
  res = wdl_cpp_cpu(Y, C, reg, S, n_threads, batch_size, epochs, sinkhorn_mode,
                    max_iter, zero_tol, optimizer, eta, gamma, beta1, beta2,
                    eps, verbose);
#endif

  return res;
}
