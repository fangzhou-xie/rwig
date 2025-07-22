
// implement the WDL algorithm
// https://arxiv.org/abs/2504.08722
// Section 7.2

// #include <iostream> // std::cout
#include "R_ext/Print.h"    // for REprintf

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "utils.h"
#include "utils_legacy.h"
#include "barycenter_legacy.h"

using namespace arma;
using namespace cpp11;
namespace writable = cpp11::writable;
using namespace cpp11::literals; // so we can use ""_nm syntax


/////////////////////////////////////////////////////////////////////
// implementation of WDL in arma arguments
/////////////////////////////////////////////////////////////////////

writable::list wdl_impl(
    arma::mat &Y, arma::mat &C,
    const double reg,
    const int S,
    const int batch_size,
    const int epochs,
    int sinkhorn_mode = 0,
    const double sinkhorn_mode_threshold = .1,
    const int maxIter = 1000, const double zeroTol = 1e-6,
    int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    double beta1 = .9, double beta2 = .999,
    double eps = 1e-8,
    const int rng_seed = 123, const bool verbose = false
) {
  // sinkhorn_mode: which Sinkhorn algorithm to be used for Barycenter
  // i.e. regular Parallel or Log-Stabilized
  // sinkhorn_mode = 0, Auto (default)
  // sinkhorn_mode = 1, Parallel
  // sinkhorn_mode = 2, Log-Stabilized

  // optimizer: which optimizer to use
  // optimizer = 0, SGD
  // optimizer = 1, Adam
  // optimizer = 2, AdamW (default)

  // sinkhorn_mode_threshold: the minimum value in the K matrix allowed
  // for using parallel algorithm

  // first set up the RNG state
  arma::arma_rng rng_state;
  rng_state.set_seed(rng_seed);

  // dims
  const int N = C.n_rows;
  const int M = Y.n_cols;

  if (verbose) {
    REprintf("Running WDL model with %i vocabs and %i docs...\n", N, M);
  }
  cpp11::check_user_interrupt();

  // TODO: maybe don't use random init? fix the init alpha and lambda to be flat
  // latent vars: alpha, lambda: randomly initiated
  mat alpha = mat(N, S, fill::randn);
  mat lambda = mat(S, M, fill::randn);
  // mat alpha = mat(N, S, fill::ones);
  // mat lambda = mat(S, M, fill::ones);
  // softmax on alpha and lambda to create A and W
  mat A = mat(arma::size(alpha), fill::zeros);
  mat W = mat(arma::size(lambda), fill::zeros);
  mat Jsoftmaxalpha = mat(N*S, N*S, fill::zeros);
  softmaxjac(Jsoftmaxalpha, A, alpha);
  softmax(W, lambda);

  // calculate K and decide the mode
  mat K = exp(-C/reg);
  if (sinkhorn_mode == 0) {
    if (K.min() < sinkhorn_mode_threshold) {
      // switch to log mode
      sinkhorn_mode = 2;
    } else {
      // switch to parallel mode
      sinkhorn_mode = 1;
    }
  }

  // init grad vars
  mat g_alpha = mat(N, S, fill::zeros);
  vec g_lambda_vec = vec(S, fill::zeros);
  mat g_lambda = mat(S, M, fill::zeros);
  // optimizer aux vars
  mat m_alpha = mat(arma::size(g_alpha), fill::zeros);
  mat v_alpha = mat(arma::size(g_alpha), fill::zeros);
  mat m_lambda = mat(arma::size(g_lambda), fill::zeros);
  mat v_lambda = mat(arma::size(g_lambda), fill::zeros);
  int step = 1; // optimizer step counter
  // temp vars
  vec b = vec(N, fill::zeros);
  mat JbA = mat(N, N*S, fill::zeros);
  mat Jbw = mat(N, S, fill::zeros);
  mat Jwl = mat(S, S, fill::zeros);
  vec Glossb = vec(N, fill::zeros);

  // batch indicator
  int batches = M / batch_size + 1; // how many batches

  if (verbose) REprintf("Optimizing WDL model\n");
  cpp11::check_user_interrupt();

  for (int e = 0; e < epochs; ++e) {

    for (int bi = 0; bi < batches; ++bi) {

      ////////////////////////////////////////////////////////////
      // 0. reset the gradients before each batch
      ////////////////////////////////////////////////////////////

      g_alpha.zeros();
      g_lambda_vec.zeros();

      ////////////////////////////////////////////////////////////
      // 1. accumulate the gradient for each entry
      ////////////////////////////////////////////////////////////

      for (int bm = 0; bm < batch_size; ++bm) {
        // check user interrupt if computation is stuck
        cpp11::check_user_interrupt();

        int m = bi * batch_size + bm; // index of the entire corpus 0,...,M-1

        if (verbose) {
          REprintf("Epoch %i, batch %i, doc %i\n", e, bi, m);
        }

        if (m == M) break; // this only happens in the last batch

        // get the doc/data and corresponding weight l
        vec yl = Y.col(m);
        vec lbdm = lambda.col(m);
        vec w = W.col(m);

        // run the barycenter algo
        barycenter_wdl(
          b, JbA, Jbw, A, C, w, reg,
          sinkhorn_mode, sinkhorn_mode_threshold, maxIter, zeroTol, verbose
        );
        // compute the gradient
        Glossb = 2 * (b - yl);
        softmax_jac_only(Jwl, lbdm);

        // std::cout << size(Glossb.t()) << std::endl;
        // std::cout << size(JbA) << std::endl;
        // std::cout << size(Jsoftmaxalpha) << std::endl;

        // accumulate the gradients
        g_alpha += reshape((Glossb.t() * JbA * Jsoftmaxalpha).t(), N, S);
        g_lambda_vec += (Glossb.t() * Jbw * Jwl).t();
      } // END OF ONE BATCH

      ////////////////////////////////////////////////////////////
      // 2. average gradients across the batch
      ////////////////////////////////////////////////////////////

      // after each batch, average gradient
      g_alpha /= batch_size;
      g_lambda_vec /= batch_size;
      g_lambda = g_lambda_vec * ones(M).t();

      ////////////////////////////////////////////////////////////
      // 3. update the parameters with the mini-batch gradients
      ////////////////////////////////////////////////////////////

      // update by the optimizer
      if (optimizer == 0) {
        // SGD
        optimizer_sgd(alpha, g_alpha, m_alpha, v_alpha,
                      step, eta, gamma, beta1, beta2, eps);
        optimizer_sgd(lambda, g_lambda, m_lambda, v_lambda,
                      step, eta, gamma, beta1, beta2, eps);
      } else if (optimizer == 1) {
        // Adam
        optimizer_adam(alpha, g_alpha, m_alpha, v_alpha,
                       step, eta, gamma, beta1, beta2, eps);
        optimizer_adam(lambda, g_lambda, m_lambda, v_lambda,
                       step, eta, gamma, beta1, beta2, eps);
      } else if (optimizer == 2) {
        // AdamW
        optimizer_adamw(alpha, g_alpha, m_alpha, v_alpha,
                        step, eta, gamma, beta1, beta2, eps);
        optimizer_adamw(lambda, g_lambda, m_lambda, v_lambda,
                        step, eta, gamma, beta1, beta2, eps);
      } else {
        cpp11::stop("optimizer must be: 0, 1, 2!");
      }
      // update the optimizer step counter
      step++;

      ////////////////////////////////////////////////////////////
      // 4. update the latent variables A and W
      ////////////////////////////////////////////////////////////

      softmaxjac(Jsoftmaxalpha, A, alpha);
      softmax(W, lambda);

    } // END OF ALL BATCHES
  } // END OF ALL EPOCHS

  if (verbose) {
    REprintf("Inference on the dataset\n");
  }

  // TODO: re-run the barycenter based on current A and W
  mat Yhat = mat(size(Y), fill::zeros);
  vec what = vec(S, fill::zeros);
  for (int m = 0; m < M; ++m) {
    what = W.col(m);
    barycenter_wdl(
      b, JbA, Jbw, A, C, what, reg,
      sinkhorn_mode, sinkhorn_mode_threshold, maxIter, zeroTol
    );
    // std::cout << b << std::endl;
    Yhat.col(m) = b;
  }

  // output the optimized topics A and weights W
  writable::list res;
  res.push_back({"A"_nm = as_doubles_matrix(A)});
  res.push_back({"W"_nm = as_doubles_matrix(W)});
  res.push_back({"Yhat"_nm = as_doubles_matrix(Yhat)});
  return res;
}



/////////////////////////////////////////////////////////////////////
// Functions for the R interface for direct calling
/////////////////////////////////////////////////////////////////////

[[cpp11::register]]
writable::list wdl_legacy_cpp(
    const doubles_matrix<>& YR, const doubles_matrix<>& CR,
    const double reg,
    const int S,
    const int batch_size,
    const int epochs,
    int sinkhorn_mode = 0,
    const double sinkhorn_mode_threshold = .1,
    const int maxIter = 1000, const double zeroTol = 1e-6,
    int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    double beta1 = .9, double beta2 = .999,
    double eps = 1e-8,
    const int rng_seed = 123, const bool verbose = false
) {
  // convert R matrices into arma ones
  mat Y = as_Mat(YR);
  mat C = as_Mat(CR);
  return wdl_impl(
    Y,C,reg,S,batch_size,epochs,
    sinkhorn_mode,sinkhorn_mode_threshold,maxIter,zeroTol,
    optimizer,eta,gamma,beta1,beta2,eps,rng_seed,verbose
  );
}
