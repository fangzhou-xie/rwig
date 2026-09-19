
// .Call entry point for the Wasserstein Dictionary Learning model

#include <stdexcept>

#include "wdl_impl.hpp" // header for the WDL definition

/////////////////////////////////////////////////////////////////////
// R interfaces for WDL
/////////////////////////////////////////////////////////////////////

// TODO: add warning for the non-converging Barycenter?

struct WdlArgs {
  double reg;
  int S, n_threads, batch_size, epochs, sinkhorn_mode, max_iter, optimizer;
  double zero_tol, eta, gamma, beta1, beta2, eps;
  bool verbose;
  int seed;
};

static SEXP wdl_cpp_cpu(SEXP Y, SEXP C, const WdlArgs &a) {
  la::Mat Y_ = rr::mat_from_R(Y);
  la::Mat C_ = rr::mat_from_R(C);

  // init the WDL class
  WassersteinDictionaryLearning wdl(a.batch_size, a.epochs, a.n_threads,
                                    a.sinkhorn_mode, a.max_iter, a.zero_tol,
                                    a.optimizer, a.eta, a.gamma, a.beta1,
                                    a.beta2, a.eps, a.verbose);

  //  with data
  wdl.init_data(Y_, C_, a.reg, a.S);

  // start the actual WDL computation (random init uses R's RNG)
  {
    rr::RNGScope rng;
    wdl.compute();
  }

  rr::ListBuilder out;
  out.add("A", rr::to_R(wdl.A)).add("W", rr::to_R(wdl.W)).add("Yhat", rr::to_R(wdl.Yhat));
  return out.build();
}

// only have the CUDA version when they are detected
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)

#include "cuda_interface.cuh"

static SEXP wdl_cpp_cuda(SEXP Y, SEXP C, const WdlArgs &a) {
  rr::Protector p;
  Y = rr::as_real(Y, p);
  C = rr::as_real(C, p);
  const int N = Rf_nrows(Y);
  const int M = Rf_ncols(Y);

  // allocate output matrices
  SEXP A_ = rr::alloc_matrix(N, a.S, p);
  SEXP W_ = rr::alloc_matrix(a.S, M, p);
  SEXP Yhat_ = rr::alloc_matrix(N, M, p);

  cuda_wdl(REAL(A_), REAL(W_), REAL(Yhat_), REAL(Y), REAL(C), N, M, a.S, a.reg,
           a.max_iter, a.zero_tol, a.batch_size, a.epochs, a.optimizer, a.eta,
           a.gamma, a.beta1, a.beta2, a.eps, a.seed, a.verbose);

  rr::ListBuilder out;
  out.add("A", A_).add("W", W_).add("Yhat", Yhat_);
  return out.build();
}

#endif

/*
Interfaces for the R side
*/

extern "C" SEXP rwig_wdl_cpp(SEXP Y, SEXP C, SEXP reg, SEXP S, SEXP n_threads,
                             SEXP batch_size, SEXP epochs, SEXP sinkhorn_mode,
                             SEXP usecuda, SEXP max_iter, SEXP zero_tol,
                             SEXP optimizer, SEXP eta, SEXP gamma, SEXP beta1,
                             SEXP beta2, SEXP eps, SEXP verbose, SEXP seed) {
  return rr::call_guard([&]() -> SEXP {
    WdlArgs a;
    a.reg = rr::as_double(reg);
    a.S = rr::as_int(S);
    a.n_threads = rr::as_int(n_threads);
    a.batch_size = rr::as_int(batch_size);
    a.epochs = rr::as_int(epochs);
    a.sinkhorn_mode = rr::as_int(sinkhorn_mode);
    a.max_iter = rr::as_int(max_iter);
    a.zero_tol = rr::as_double(zero_tol);
    a.optimizer = rr::as_int(optimizer);
    a.eta = rr::as_double(eta);
    a.gamma = rr::as_double(gamma);
    a.beta1 = rr::as_double(beta1);
    a.beta2 = rr::as_double(beta2);
    a.eps = rr::as_double(eps);
    a.verbose = rr::as_bool(verbose);
    a.seed = rr::as_int(seed);

    // check sinkhorn mode
    if ((a.sinkhorn_mode != 1) && (a.sinkhorn_mode != 2)) {
      throw std::runtime_error("Sinkhorn mode not supported");
    }

    // check optimizer mode
    if ((a.optimizer != 0) && (a.optimizer != 1) && (a.optimizer != 2)) {
      throw std::runtime_error("optimizer must be: 0, 1, 2!");
    }

    // stop if N <= S
    if (Rf_nrows(Y) <= a.S) {
      throw std::runtime_error(
          "Number of topics S must be smaller than the vocab size N!");
    }

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
    if (rr::as_bool(usecuda)) {
      if (a.verbose) {
        rr::message("Running WDL in CUDA mode...");
        rr::message("This might take a while depending on the problem size...");
      }
      return wdl_cpp_cuda(Y, C, a);
    }
#else
    (void)usecuda;
#endif
    if (a.verbose) {
      rr::message("Running WDL in CPU mode...");
      rr::message("This might take a while depending on the problem size...");
    }
    return wdl_cpp_cpu(Y, C, a);
  });
}
