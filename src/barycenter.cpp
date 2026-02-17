
// this is the file defining the functions exporting to R side
// barycenter algos

#include "check_cuda.hpp"

#include "barycenter_impl.hpp"
// #include "ctrack.hpp"

// using namespace arma;
// using namespace cpp11;
// using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable; // writable list from cpp11

Rcpp::List barycenter_parallel_cpu(const SEXP &A, const SEXP &C, const SEXP &w,
                                   double reg, const SEXP &b_ext,
                                   bool withgrad = false, int maxiter = 1000,
                                   double zerotol = 1e-6, int verbose = 0) {

  // convert R vectors/matrices into arma ones
  arma::mat A_{Rcpp::as<arma::mat>(A)};
  arma::mat C_{Rcpp::as<arma::mat>(C)};
  arma::vec w_{Rcpp::as<arma::vec>(w)};
  arma::vec b_ext_;
  if (withgrad) {
    b_ext_ = Rcpp::as<arma::vec>(b_ext);
  }

  // init the class
  Barycenter bc(A_.n_cols, withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  if (withgrad) {
    bc.update_b_ext(b_ext_);
  }

  // start the computation
  bc.compute_parallel();

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
        Rcpp::Named("b") = bc.b, Rcpp::Named("grad_A") = bc.grad_A,
        Rcpp::Named("grad_w") = bc.grad_w, Rcpp::Named("loss") = bc.loss,
        Rcpp::Named("U") = bc.U, Rcpp::Named("V") = bc.V,
        Rcpp::Named("iter") = bc.iter, Rcpp::Named("err") = bc.err,
        Rcpp::Named("return_status") = bc.return_code);
  } else {
    return Rcpp::List::create(Rcpp::Named("b") = bc.b,
                              // Rcpp::Named("grad_A") = bc.grad_A,
                              // Rcpp::Named("grad_w") = bc.grad_w,
                              // Rcpp::Named("loss") = bc.loss,
                              Rcpp::Named("U") = bc.U, Rcpp::Named("V") = bc.V,
                              Rcpp::Named("iter") = bc.iter,
                              Rcpp::Named("err") = bc.err,
                              Rcpp::Named("return_status") = bc.return_code);
  }
}

// only have the CUDA version when they are detected
#ifdef HAVE_CUBLAS
#ifdef HAVE_CUDA_RUNTIME

#include "cuda_interface.cuh"

Rcpp::List barycenter_parallel_cuda(const SEXP &A, const SEXP &C, const SEXP &w,
                                    double reg, const SEXP &b_ext,
                                    bool withgrad = false, int maxiter = 1000,
                                    double zerotol = 1e-6, int verbose = 0) {
  double *A_ptr = REAL(A);
  double *C_ptr = REAL(C);
  double *w_ptr = REAL(w);
  double *b_ext_ptr = REAL(b_ext);
  int M = Rf_nrows(C);
  int N = Rf_ncols(C);
  int S = Rf_ncols(A);

  SEXP b_ = PROTECT(Rf_allocVector(REALSXP, N));
  SEXP U_ = PROTECT(Rf_allocVector(REALSXP, M * S));
  SEXP V_ = PROTECT(Rf_allocVector(REALSXP, N * S));
  SEXP grad_A_ = PROTECT(Rf_allocVector(REALSXP, M * S));
  SEXP grad_w_ = PROTECT(Rf_allocVector(REALSXP, S));

  double *b_ptr = REAL(b_);
  double *U_ptr = REAL(U_);
  double *V_ptr = REAL(V_);
  double *grad_A_ptr = REAL(grad_A_);
  double *grad_w_ptr = REAL(grad_w_);

  double loss = 0.;
  int iter = 0;
  double err = 0.;

  cuda_barycenter_parallel(U_ptr, V_ptr, b_ptr, grad_A_ptr, grad_w_ptr, &loss,
                           &iter, &err, A_ptr, w_ptr, C_ptr, b_ext_ptr, M, N, S,
                           reg, withgrad, maxiter, zerotol);

  // set dims
  SEXP dims_MS = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims_MS)[0] = M;
  INTEGER(dims_MS)[1] = S;
  SEXP dims_NS = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims_NS)[0] = N;
  INTEGER(dims_NS)[1] = S;

  Rf_setAttrib(U_, R_DimSymbol, dims_MS);
  Rf_setAttrib(V_, R_DimSymbol, dims_NS);
  Rf_setAttrib(grad_A_, R_DimSymbol, dims_MS);

  UNPROTECT(7);

  int return_code;
  if (err <= zerotol) {
    return_code = 0;
  } else if (iter >= maxiter) {
    return_code = 1;
  } else {
    return_code = 2;
  }

  // Rcpp::List res;
  if (withgrad) {
    return Rcpp::List::create(
        Rcpp::Named("b") = b_, Rcpp::Named("grad_A") = grad_A_,
        Rcpp::Named("grad_w") = grad_w_, Rcpp::Named("loss") = loss,
        Rcpp::Named("U") = U_, Rcpp::Named("V") = V_,
        Rcpp::Named("iter") = iter, Rcpp::Named("err") = err,
        Rcpp::Named("return_status") = return_code);
  } else {
    return Rcpp::List::create(Rcpp::Named("b") = b_, Rcpp::Named("U") = U_,
                              Rcpp::Named("V") = V_, Rcpp::Named("iter") = iter,
                              Rcpp::Named("err") = err,
                              Rcpp::Named("return_status") = return_code);
  }
}

#endif
#endif

/*
Interfaces for the R side
*/

// [[Rcpp::export]]
Rcpp::List barycenter_parallel_cpp(const SEXP &A, const SEXP &C, const SEXP &w,
                                   double reg, const SEXP &b_ext,
                                   bool withgrad = false, bool usecuda = true,
                                   int maxiter = 1000, double zerotol = 1e-6,
                                   int verbose = 0) {
  Rcpp::List res;

  if constexpr (check_cuda::has_cuda) {
    if (usecuda) {
      res = barycenter_parallel_cuda(A, C, w, reg, b_ext, withgrad, maxiter,
                                     zerotol, verbose);
    } else {
      res = barycenter_parallel_cpu(A, C, w, reg, b_ext, withgrad, maxiter,
                                    zerotol, verbose);
    }
  } else {
    res = barycenter_parallel_cpu(A, C, w, reg, b_ext, withgrad, maxiter,
                                  zerotol, verbose);
  }
  return res;
}

// [[Rcpp::export]]
Rcpp::List barycenter_log_cpp(const arma::mat &A, const arma::mat &C,
                              const arma::vec &w, double reg,
                              const arma::vec &b_ext, bool withgrad = false,
                              const int &n_threads = 0, int maxiter = 1000,
                              double zerotol = 1e-6, int verbose = 0) {

  // init the class
  Barycenter bc(A.n_cols, withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C);
  bc.update_reg(reg);
  bc.update_A(A);
  bc.update_w(w);
  if (withgrad) {
    bc.update_b_ext(b_ext);
  }

  // start the computation
  bc.compute_log(n_threads);

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
        Rcpp::Named("b") = bc.b, Rcpp::Named("grad_A") = bc.grad_A,
        Rcpp::Named("grad_w") = bc.grad_w, Rcpp::Named("loss") = bc.loss,
        Rcpp::Named("F") = bc.U, Rcpp::Named("G") = bc.V,
        Rcpp::Named("iter") = bc.iter, Rcpp::Named("err") = bc.err,
        Rcpp::Named("return_status") = bc.return_code);
  } else {
    return Rcpp::List::create(Rcpp::Named("b") = bc.b,
                              // Rcpp::Named("grad_A") = bc.grad_A,
                              // Rcpp::Named("grad_w") = bc.grad_w,
                              // Rcpp::Named("loss") = bc.loss,
                              Rcpp::Named("F") = bc.U, Rcpp::Named("G") = bc.V,
                              Rcpp::Named("iter") = bc.iter,
                              Rcpp::Named("err") = bc.err,
                              Rcpp::Named("return_status") = bc.return_code);
  }
}
