
// .Call entry points for the barycenter algorithms

#include "barycenter_impl.hpp"

static SEXP barycenter_result(const Barycenter &bc, bool withgrad,
                              const char *Uname, const char *Vname) {
  rr::ListBuilder out;
  out.add("b", rr::to_R(bc.b));
  if (withgrad) {
    out.add("grad_A", rr::to_R(bc.grad_A))
        .add("grad_w", rr::to_R(bc.grad_w))
        .add("loss", bc.loss);
  }
  out.add(Uname, rr::to_R(bc.U))
      .add(Vname, rr::to_R(bc.V))
      .add("iter", bc.iter)
      .add("err", bc.err)
      .add("return_status", bc.return_code);
  return out.build();
}

static SEXP barycenter_parallel_cpu(SEXP A, SEXP C, SEXP w, double reg,
                                    SEXP b_ext, bool withgrad, int maxiter,
                                    double zerotol, int verbose) {
  la::Mat A_ = rr::mat_from_R(A);
  la::Mat C_ = rr::mat_from_R(C);
  la::Vec w_ = rr::vec_from_R(w);

  // init the class
  Barycenter bc((int)A_.ncol(), withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  if (withgrad) {
    bc.update_b_ext(rr::vec_from_R(b_ext));
  }

  // start the computation
  bc.compute_parallel();

  return barycenter_result(bc, withgrad, "U", "V");
}

// only have the CUDA version when they are detected
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

#include "cuda_interface.cuh"

static SEXP barycenter_parallel_cuda(SEXP A, SEXP C, SEXP w, double reg,
                                     SEXP b_ext, bool withgrad, int maxiter,
                                     double zerotol, int verbose) {
  rr::Protector p;
  A = rr::as_real(A, p);
  C = rr::as_real(C, p);
  w = rr::as_real(w, p);
  b_ext = rr::as_real(b_ext, p);
  const int M = Rf_nrows(C);
  const int N = Rf_ncols(C);
  const int S = Rf_ncols(A);

  SEXP b_ = rr::alloc_vector(N, p);
  SEXP U_ = rr::alloc_matrix(M, S, p);
  SEXP V_ = rr::alloc_matrix(N, S, p);
  SEXP grad_A_ = rr::alloc_matrix(M, S, p);
  SEXP grad_w_ = rr::alloc_vector(S, p);

  double loss = 0.;
  int iter = 0;
  double err = 0.;

  cuda_barycenter_parallel(REAL(U_), REAL(V_), REAL(b_), REAL(grad_A_),
                           REAL(grad_w_), &loss, &iter, &err, REAL(A), REAL(w),
                           REAL(C), REAL(b_ext), M, N, S, reg, withgrad,
                           maxiter, zerotol);

  int return_code;
  if (err <= zerotol) {
    return_code = 0;
  } else if (iter >= maxiter) {
    return_code = 1;
  } else {
    return_code = 2;
  }

  rr::ListBuilder out;
  out.add("b", b_);
  if (withgrad) out.add("grad_A", grad_A_).add("grad_w", grad_w_).add("loss", loss);
  out.add("U", U_).add("V", V_).add("iter", iter).add("err", err);
  out.add("return_status", return_code);
  return out.build();
}

#endif

/*
Interfaces for the R side
*/

extern "C" SEXP rwig_barycenter_parallel_cpp(SEXP A, SEXP C, SEXP w, SEXP reg,
                                             SEXP b_ext, SEXP withgrad,
                                             SEXP usecuda, SEXP maxiter,
                                             SEXP zerotol, SEXP verbose) {
  return rr::call_guard([&]() -> SEXP {
    const double reg_ = rr::as_double(reg);
    const bool withgrad_ = rr::as_bool(withgrad);
    const int maxiter_ = rr::as_int(maxiter);
    const double zerotol_ = rr::as_double(zerotol);
    const int verbose_ = rr::as_int(verbose);
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
    if (rr::as_bool(usecuda)) {
      return barycenter_parallel_cuda(A, C, w, reg_, b_ext, withgrad_, maxiter_,
                                      zerotol_, verbose_);
    }
#else
    (void)usecuda;
#endif
    return barycenter_parallel_cpu(A, C, w, reg_, b_ext, withgrad_, maxiter_,
                                   zerotol_, verbose_);
  });
}

extern "C" SEXP rwig_barycenter_log_cpp(SEXP A, SEXP C, SEXP w, SEXP reg,
                                        SEXP b_ext, SEXP withgrad,
                                        SEXP n_threads, SEXP maxiter,
                                        SEXP zerotol, SEXP verbose) {
  return rr::call_guard([&]() -> SEXP {
    la::Mat A_ = rr::mat_from_R(A);
    la::Mat C_ = rr::mat_from_R(C);
    la::Vec w_ = rr::vec_from_R(w);
    const bool withgrad_ = rr::as_bool(withgrad);

    // init the class
    Barycenter bc((int)A_.ncol(), withgrad_, rr::as_int(maxiter),
                  rr::as_double(zerotol), rr::as_int(verbose));
    // update/load all the data
    bc.update_C(C_);
    bc.update_reg(rr::as_double(reg));
    bc.update_A(A_);
    bc.update_w(w_);
    if (withgrad_) {
      bc.update_b_ext(rr::vec_from_R(b_ext));
    }

    // start the computation
    bc.compute_log(rr::as_int(n_threads));

    return barycenter_result(bc, withgrad_, "F", "G");
  });
}
