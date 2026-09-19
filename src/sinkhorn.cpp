
// this is the file defining the functions exporting to R side
// sinkhorn algos

#include "rcpp_glue.hpp"
#include "sinkhorn_impl.hpp"

static Rcpp::List sinkhorn_result(const Sinkhorn &s, bool withgrad,
                                  const char *uname, const char *vname) {
  if (withgrad) {
    return Rcpp::List::create(
        Rcpp::Named("P") = la::to_R(s.P), Rcpp::Named("grad_a") = la::to_R(s.grad_a),
        Rcpp::Named(uname) = la::to_R(s.u), Rcpp::Named(vname) = la::to_R(s.v),
        Rcpp::Named("loss") = s.loss, Rcpp::Named("iter") = s.iter,
        Rcpp::Named("err") = s.err,
        Rcpp::Named("return_status") = s.return_code);
  } else {
    return Rcpp::List::create(
        Rcpp::Named("P") = la::to_R(s.P), Rcpp::Named(uname) = la::to_R(s.u),
        Rcpp::Named(vname) = la::to_R(s.v), Rcpp::Named("loss") = s.loss,
        Rcpp::Named("iter") = s.iter, Rcpp::Named("err") = s.err,
        Rcpp::Named("return_status") = s.return_code);
  }
}

Rcpp::List sinkhorn_vanilla_cpu(const SEXP &a, const SEXP &b, const SEXP &C,
                                double reg, bool withgrad = false,
                                int maxiter = 1000, double zerotol = 1e-6,
                                int verbose = 0) {
  la::Vec a_ = la::vec_from_R(a);
  la::Vec b_ = la::vec_from_R(b);
  la::Mat C_ = la::mat_from_R(C);

  // init the class and start the computation
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_vanilla(a_, b_, C_, reg);

  return sinkhorn_result(s, withgrad, "u", "v");
}

// only have the CUDA version when they are detected
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

#include "cuda_interface.cuh"

Rcpp::List sinkhorn_vanilla_cuda(const SEXP &a, const SEXP &b, const SEXP &C,
                                 double reg, bool withgrad = false,
                                 int maxiter = 1000, double zerotol = 1e-6,
                                 int verbose = 0) {

  double *a_ptr = REAL(a);
  double *b_ptr = REAL(b);
  double *C_ptr = REAL(C);
  int m = Rf_nrows(C);
  int n = Rf_ncols(C);

  SEXP u_ = PROTECT(Rf_allocVector(REALSXP, m));
  SEXP v_ = PROTECT(Rf_allocVector(REALSXP, n));
  SEXP P_ = PROTECT(Rf_allocVector(REALSXP, m * n));
  SEXP grad_a_ = PROTECT(Rf_allocVector(REALSXP, m));

  double *u_ptr = REAL(u_);
  double *v_ptr = REAL(v_);
  double *P_ptr = REAL(P_);
  double *grad_a_ptr = REAL(grad_a_);

  double loss = 0.;
  int iter = 0;
  double err = 0.;

  cuda_sinkhorn_vanilla(P_ptr, grad_a_ptr, u_ptr, v_ptr, &loss, &iter, &err,
                        a_ptr, b_ptr, C_ptr, m, n, reg, withgrad, maxiter,
                        zerotol);

  // Set matrix dimensions for P
  SEXP dims = PROTECT(Rf_allocVector(INTSXP, 2));
  INTEGER(dims)[0] = m;
  INTEGER(dims)[1] = n;
  Rf_setAttrib(P_, R_DimSymbol, dims);

  UNPROTECT(5);

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
        Rcpp::Named("P") = P_, Rcpp::Named("grad_a") = grad_a_,
        Rcpp::Named("u") = u_, Rcpp::Named("v") = v_,
        Rcpp::Named("loss") = loss, Rcpp::Named("iter") = iter,
        Rcpp::Named("err") = err, Rcpp::Named("return_status") = return_code);
  } else {
    return Rcpp::List::create(
        Rcpp::Named("P") = P_, Rcpp::Named("u") = u_, Rcpp::Named("v") = v_,
        Rcpp::Named("loss") = loss, Rcpp::Named("iter") = iter,
        Rcpp::Named("err") = err, Rcpp::Named("return_status") = return_code);
  }
}

#endif

/*
Interfaces for the R side
*/

// [[Rcpp::export]]
Rcpp::List sinkhorn_vanilla_cpp(const SEXP &a, const SEXP &b, const SEXP &C,
                                double reg, bool withgrad = false,
                                bool usecuda = true, int maxiter = 1000,
                                double zerotol = 1e-6, int verbose = 0) {
  // NOTE: currently only vanilla algo supports CUDA

  Rcpp::List res;

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  if (usecuda) {
    res = sinkhorn_vanilla_cuda(a, b, C, reg, withgrad, maxiter, zerotol,
                                verbose);
  } else {
    res =
        sinkhorn_vanilla_cpu(a, b, C, reg, withgrad, maxiter, zerotol, verbose);
  }
#else
  res = sinkhorn_vanilla_cpu(a, b, C, reg, withgrad, maxiter, zerotol, verbose);
#endif

  return res;
}

// [[Rcpp::export]]
Rcpp::List sinkhorn_log_cpp(const SEXP &a, const SEXP &b, const SEXP &C,
                            double reg, bool withgrad = false,
                            const int &n_threads = 0, int maxiter = 1000,
                            double zerotol = 1e-6, int verbose = 0) {
  la::Vec a_ = la::vec_from_R(a);
  la::Vec b_ = la::vec_from_R(b);
  la::Mat C_ = la::mat_from_R(C);

  // init the class and start the computation
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_log(a_, b_, C_, reg, n_threads);

  return sinkhorn_result(s, withgrad, "f", "g");
}
