
// .Call entry points for the Sinkhorn algorithms

#include "sinkhorn_impl.hpp"

static SEXP sinkhorn_result(const Sinkhorn &s, bool withgrad, const char *uname,
                            const char *vname) {
  rr::ListBuilder out;
  out.add("P", rr::to_R(s.P));
  if (withgrad) out.add("grad_a", rr::to_R(s.grad_a));
  out.add(uname, rr::to_R(s.u))
      .add(vname, rr::to_R(s.v))
      .add("loss", s.loss)
      .add("iter", s.iter)
      .add("err", s.err)
      .add("return_status", s.return_code);
  return out.build();
}

static SEXP sinkhorn_vanilla_cpu(SEXP a, SEXP b, SEXP C, double reg,
                                 bool withgrad, int maxiter, double zerotol,
                                 int verbose) {
  la::Vec a_ = rr::vec_from_R(a);
  la::Vec b_ = rr::vec_from_R(b);
  la::Mat C_ = rr::mat_from_R(C);

  // init the class and start the computation
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_vanilla(a_, b_, C_, reg);

  return sinkhorn_result(s, withgrad, "u", "v");
}

// only have the CUDA version when they are detected
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

#include "cuda_interface.cuh"

static SEXP sinkhorn_vanilla_cuda(SEXP a, SEXP b, SEXP C, double reg,
                                  bool withgrad, int maxiter, double zerotol,
                                  int verbose) {
  rr::Protector p;
  a = rr::as_real(a, p);
  b = rr::as_real(b, p);
  C = rr::as_real(C, p);
  const int m = Rf_nrows(C);
  const int n = Rf_ncols(C);

  SEXP u_ = rr::alloc_vector(m, p);
  SEXP v_ = rr::alloc_vector(n, p);
  SEXP P_ = rr::alloc_matrix(m, n, p);
  SEXP grad_a_ = rr::alloc_vector(m, p);

  double loss = 0.;
  int iter = 0;
  double err = 0.;

  cuda_sinkhorn_vanilla(REAL(P_), REAL(grad_a_), REAL(u_), REAL(v_), &loss,
                        &iter, &err, REAL(a), REAL(b), REAL(C), m, n, reg,
                        withgrad, maxiter, zerotol);

  int return_code;
  if (err <= zerotol) {
    return_code = 0;
  } else if (iter >= maxiter) {
    return_code = 1;
  } else {
    return_code = 2;
  }

  rr::ListBuilder out;
  out.add("P", P_);
  if (withgrad) out.add("grad_a", grad_a_);
  out.add("u", u_).add("v", v_).add("loss", loss).add("iter", iter).add(
      "err", err);
  out.add("return_status", return_code);
  return out.build();
}

#endif

/*
Interfaces for the R side
*/

extern "C" SEXP rwig_sinkhorn_vanilla_cpp(SEXP a, SEXP b, SEXP C, SEXP reg,
                                          SEXP withgrad, SEXP usecuda,
                                          SEXP maxiter, SEXP zerotol,
                                          SEXP verbose) {
  return rr::call_guard([&]() -> SEXP {
    // NOTE: currently only vanilla algo supports CUDA
    const double reg_ = rr::as_double(reg);
    const bool withgrad_ = rr::as_bool(withgrad);
    const int maxiter_ = rr::as_int(maxiter);
    const double zerotol_ = rr::as_double(zerotol);
    const int verbose_ = rr::as_int(verbose);
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
    if (rr::as_bool(usecuda)) {
      return sinkhorn_vanilla_cuda(a, b, C, reg_, withgrad_, maxiter_, zerotol_,
                                   verbose_);
    }
#else
    (void)usecuda;
#endif
    return sinkhorn_vanilla_cpu(a, b, C, reg_, withgrad_, maxiter_, zerotol_,
                                verbose_);
  });
}

extern "C" SEXP rwig_sinkhorn_log_cpp(SEXP a, SEXP b, SEXP C, SEXP reg,
                                      SEXP withgrad, SEXP n_threads,
                                      SEXP maxiter, SEXP zerotol, SEXP verbose) {
  return rr::call_guard([&]() -> SEXP {
    la::Vec a_ = rr::vec_from_R(a);
    la::Vec b_ = rr::vec_from_R(b);
    la::Mat C_ = rr::mat_from_R(C);
    const bool withgrad_ = rr::as_bool(withgrad);

    // init the class and start the computation
    Sinkhorn s(withgrad_, rr::as_int(maxiter), rr::as_double(zerotol),
               rr::as_int(verbose));
    s.compute_log(a_, b_, C_, rr::as_double(reg), rr::as_int(n_threads));

    return sinkhorn_result(s, withgrad_, "f", "g");
  });
}
