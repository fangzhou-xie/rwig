
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
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

#include "cuda_interface.cuh"

Rcpp::List wdl_cpp_cuda(const SEXP &Y, // document matrix N * M
                        const SEXP &C, // cost matrix N * N
                        const double reg, const int S, const int batch_size,
                        const int epochs, int sinkhorn_mode = 1,
                        const int max_iter = 1000, const double zero_tol = 1e-6,
                        const int optimizer = 2, const double eta = .001,
                        const double gamma = .01, const double beta1 = .9,
                        const double beta2 = .999, const double eps = 1e-8,
                        const bool verbose = false, const int seed = 42) {
  double *Y_ptr = REAL(Y);
  double *C_ptr = REAL(C);
  int N = Rf_nrows(Y);
  int M = Rf_ncols(Y);

  // allocate output matrices
  SEXP A_ = PROTECT(Rf_allocVector(REALSXP, N * S));
  SEXP W_ = PROTECT(Rf_allocVector(REALSXP, S * M));
  SEXP Yhat_ = PROTECT(Rf_allocVector(REALSXP, N * M));

  cuda_wdl(REAL(A_), REAL(W_), REAL(Yhat_), Y_ptr, C_ptr, N, M, S, reg,
           max_iter, zero_tol, batch_size, epochs, optimizer, eta, gamma, beta1,
           beta2, eps, seed, verbose);

  // set dims
  SEXP dims_NS = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims_NS)[0] = N;
  INTEGER(dims_NS)[1] = S;
  SEXP dims_SM = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims_SM)[0] = S;
  INTEGER(dims_SM)[1] = M;
  SEXP dims_NM = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims_NM)[0] = N;
  INTEGER(dims_NM)[1] = M;

  Rf_setAttrib(A_, R_DimSymbol, dims_NS);
  Rf_setAttrib(W_, R_DimSymbol, dims_SM);
  Rf_setAttrib(Yhat_, R_DimSymbol, dims_NM);

  UNPROTECT(6);

  return Rcpp::List::create(Rcpp::Named("A") = A_, Rcpp::Named("W") = W_,
                            Rcpp::Named("Yhat") = Yhat_);
}

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
                   const bool verbose = false, const int seed = 42) {

  // check sinkhorn mode
  if ((sinkhorn_mode != 1) && (sinkhorn_mode != 2)) {
    Rcpp::stop("Sinkhorn mode not supported");
  }

  // check optimizer mode
  if ((optimizer != 0) && (optimizer != 1) && (optimizer != 2)) {
    Rcpp::stop("optimizer must be: 0, 1, 2!");
  }

  // stop if N <= S
  if (Rf_nrows(Y) <= S) {
    Rcpp::stop("Number of topics S must be smaller than the vocab size N!");
  }

  Rcpp::List res;

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  if (usecuda) {
    Rcpp::message(Rf_mkString("Running WDL in CUDA mode..."));
    Rcpp::message(Rf_mkString(
        "This might take a while depending on the problem size..."));
    res = wdl_cpp_cuda(Y, C, reg, S, batch_size, epochs, sinkhorn_mode,
                       max_iter, zero_tol, optimizer, eta, gamma, beta1, beta2,
                       eps, verbose, seed);
  } else {
    Rcpp::message(Rf_mkString("Running WDL in CPU mode..."));
    Rcpp::message(Rf_mkString(
        "This might take a while depending on the problem size..."));
    res = wdl_cpp_cpu(Y, C, reg, S, n_threads, batch_size, epochs,
                      sinkhorn_mode, max_iter, zero_tol, optimizer, eta, gamma,
                      beta1, beta2, eps, verbose);
  }
#else
  Rcpp::message(Rf_mkString("Running WDL in CPU mode..."));
  Rcpp::message(
      Rf_mkString("This might take a while depending on the problem size..."));
  res = wdl_cpp_cpu(Y, C, reg, S, n_threads, batch_size, epochs, sinkhorn_mode,
                    max_iter, zero_tol, optimizer, eta, gamma, beta1, beta2,
                    eps, verbose);
#endif

  return res;
}
