
// this is the file defining the functions exporting to R side
// barycenter algos

#include "barycenter_impl.hpp"
#include "rcpp_glue.hpp"

static Rcpp::List barycenter_result(const Barycenter &bc, bool withgrad,
                                    const char *Uname, const char *Vname) {
  if (withgrad) {
    return Rcpp::List::create(
        Rcpp::Named("b") = la::to_R(bc.b), Rcpp::Named("grad_A") = la::to_R(bc.grad_A),
        Rcpp::Named("grad_w") = la::to_R(bc.grad_w), Rcpp::Named("loss") = bc.loss,
        Rcpp::Named(Uname) = la::to_R(bc.U), Rcpp::Named(Vname) = la::to_R(bc.V),
        Rcpp::Named("iter") = bc.iter, Rcpp::Named("err") = bc.err,
        Rcpp::Named("return_status") = bc.return_code);
  } else {
    return Rcpp::List::create(
        Rcpp::Named("b") = la::to_R(bc.b), Rcpp::Named(Uname) = la::to_R(bc.U),
        Rcpp::Named(Vname) = la::to_R(bc.V), Rcpp::Named("iter") = bc.iter,
        Rcpp::Named("err") = bc.err,
        Rcpp::Named("return_status") = bc.return_code);
  }
}

Rcpp::List barycenter_parallel_cpu(const SEXP &A, const SEXP &C, const SEXP &w,
                                   double reg, const SEXP &b_ext,
                                   bool withgrad = false, int maxiter = 1000,
                                   double zerotol = 1e-6, int verbose = 0) {
  la::Mat A_ = la::mat_from_R(A);
  la::Mat C_ = la::mat_from_R(C);
  la::Vec w_ = la::vec_from_R(w);

  // init the class
  Barycenter bc((int)A_.ncol(), withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  if (withgrad) {
    bc.update_b_ext(la::vec_from_R(b_ext));
  }

  // start the computation
  bc.compute_parallel();

  return barycenter_result(bc, withgrad, "U", "V");
}

// only have the CUDA version when they are detected
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

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

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  if (usecuda) {
    res = barycenter_parallel_cuda(A, C, w, reg, b_ext, withgrad, maxiter,
                                   zerotol, verbose);
  } else {
    res = barycenter_parallel_cpu(A, C, w, reg, b_ext, withgrad, maxiter,
                                  zerotol, verbose);
  }
#else
  res = barycenter_parallel_cpu(A, C, w, reg, b_ext, withgrad, maxiter, zerotol,
                                verbose);
#endif
  return res;
}

// [[Rcpp::export]]
Rcpp::List barycenter_log_cpp(const SEXP &A, const SEXP &C, const SEXP &w,
                              double reg, const SEXP &b_ext,
                              bool withgrad = false, const int &n_threads = 0,
                              int maxiter = 1000, double zerotol = 1e-6,
                              int verbose = 0) {
  la::Mat A_ = la::mat_from_R(A);
  la::Mat C_ = la::mat_from_R(C);
  la::Vec w_ = la::vec_from_R(w);

  // init the class
  Barycenter bc((int)A_.ncol(), withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  if (withgrad) {
    bc.update_b_ext(la::vec_from_R(b_ext));
  }

  // start the computation
  bc.compute_log(n_threads);

  return barycenter_result(bc, withgrad, "F", "G");
}
