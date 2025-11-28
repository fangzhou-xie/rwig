
// implement the WDL algorithm in this header file
// https://arxiv.org/abs/2504.08722
// Section 7.2 and 7.3


#ifndef WIG_WDL_H
#define WIG_WDL_H

#ifndef ARMA_DONT_USE_OPENMP
#define ARMA_DONT_USE_OPENMP
#endif

// #include <iostream>               // std::cout
// #include "R_ext/Print.h"          // for REprintf used for verbose logging

#define RCPP_NO_SUGAR
#define RCPP_ARMADILLO_RETURN_ANYVEC_AS_VECTOR

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

// WIG-related headers
#include "timer.hpp"              // TicToc timer class
#include "barycenter_impl.hpp"    // using the barycenter algorithms
#include "optimizer.hpp"          // using the optimizers: SGD, Adam, AdamW

// #include "ctrack.hpp"             // performance profiler (only for testing)


using namespace arma;

class WassersteinDictionaryLearning {

private:

  // data
  mat _Y;           // document matrix
  mat _C;           // distance matrix
  double _reg;      // regularization

  // model dimensions
  int _N;           // number of tokens
  int _M;           // number of docs
  int _S;           // number of topics

  // model related
  int _B;           // number of batchs
  int _E;           // number of epochs
  int _n_threads;   // number of threads, 0: serial

  // sinkhorn related
  int _sinkmode;    // sinkhorn mode: vanilla (parallel) or log
  int _maxiter;     // max iteration of sinkhorn/barycenter algo
  double _zerotol;  // convergence tolerance

  // optimizer related
  int _opt;         // optimizer mode (SGD, Adam, AdamW)
  double _eta, _gamma, _beta1, _beta2, _eps; // AdamW params

  // logging
  bool _verbose;     // logging frequency

  // RNG state for randomness
  // int _rng_seed;

  // latent vars
  mat _Alpha;       // N * S
  mat _Lambda;      // S * M
  vec _onesM;       // vector of ones

  // TicToc timer class
  TicToc _timer;    // used for logging each iteration time

  // gradients for latent vars: Alpha, Lambda
  mat _g_Alpha;     // gradient of Alpha (for A)
  mat _g_Lambda;    // gradient of Lambda (for W)
  vec _g_lambda;    // used for each doc processing

  // optimizers for latent vars
  Optimizer _opt_Alpha, _opt_Lambda;

  // softmax for both latent vars (A and W)
  void _softmax() {
    // CTRACK;

    // requires init of alpha, lambda

    // A = softmax(Alpha)
    rowvec col_maxs_alpha = arma::max(_Alpha, 0);
    this->A = exp(_Alpha.each_row() - col_maxs_alpha);
    this->A = this->A.each_row() / sum(this->A, 0);

    // W = softmax(Lambda)
    rowvec col_maxs_lambda = arma::max(_Lambda, 0);
    this->W = exp(_Lambda.each_row() - col_maxs_lambda);
    this->W = this->W.each_row() / sum(this->W, 0);

    // update this->A, this->W
  }

  ///////////////////////////////////////////
  // methods for serial
  ///////////////////////////////////////////

  void _compute_serial();
  void _train_batch_serial(Barycenter& bc, int batch_id);
  // void _train_batch_compute(Barycenter& bc, vec& y, vec& w);
  // optimizer step
  void _optimize() {
    // CTRACK;

    // update the params by the optimizer
    if (_opt == 0) {                  // SGD
      _opt_Alpha.sgd(_Alpha, _g_Alpha);
      _opt_Lambda.sgd(_Lambda, _g_Lambda);
    } else if (_opt == 1) {           // Adam
      _opt_Alpha.adam(_Alpha, _g_Alpha);
      _opt_Lambda.adam(_Lambda, _g_Lambda);
    } else if (_opt == 2) {           // AdamW
      _opt_Alpha.adamw(_Alpha, _g_Alpha);
      _opt_Lambda.adamw(_Lambda, _g_Lambda);
    }

    // update _Alpha, _Lambda in-place
  };

  // accumulate gradient
  void _accu_grad(mat& g_A, vec& g_w, vec& w) {
    // CTRACK;

    mat g_Alpha = g_A;
    vec g_lambda = g_w;

    // multiply the jacobian of softmax
    for (int s = 0; s < _S; ++s) {
      g_Alpha.col(s) = (
        diagmat(A.col(s)) - (A.col(s) * A.col(s).t())
      ).t() * g_Alpha.col(s);
    }
    g_lambda = (diagmat(w) - (w * w.t())) * g_lambda;

    // accumulate the gradients
    _g_Alpha += g_Alpha;
    _g_lambda += g_lambda;
  }


  ///////////////////////////////////////////
  // methods for thread
  ///////////////////////////////////////////

  // void _compute_thread();
  // void _train_batch_thread(int batch_id);
  // void _infer_thread();


  // TODO: separate the batch call from the main compute loop
  // then to have serial/thread calls separately
  // for each batch => update gradients
  // void _proc_batch(const int threads);
  // void _proc_batch_serial();

public:

  // output vars
  mat A;    // topics, N * S
  mat W;    // weight, S * M
  mat Yhat; // predicted barycenters, N * M

  // init the class
  WassersteinDictionaryLearning(
    const int batch_size,
    const int epochs,
    const int n_threads,
    const int sinkhorn_mode = 2,
    const int max_iter = 1000, const double zero_tol = 1e-6,
    const int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    const double beta1 = .9, const double beta2 = .999,
    const double eps = 1e-8,
    const bool verbose = false
  ) {
    // const int rng_seed = 123,
    _B = batch_size;
    _E = epochs;
    _n_threads = n_threads;
    _sinkmode = sinkhorn_mode;
    _maxiter = max_iter;
    _zerotol = zero_tol;
    _opt = optimizer;
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
    // _rng_seed = rng_seed;
    _verbose = verbose;
  }
  // destructor
  ~WassersteinDictionaryLearning() {}

  // init the data
  void init_data(const mat& Y, const mat& C, double reg, int S) {
    // load the data
    _Y = Y;
    _C = C;
    _reg = reg;

    // dimension
    _M = _Y.n_cols;
    _N = _Y.n_rows;
    _S = S;

    // helper vecs
    _onesM = vec(_M, fill::ones);

  }

  // actual compute method for WDL
  void compute() {
    _compute_serial();
    // if (_n_threads == 0) {
    //   _compute_serial();
    // } else {
    //   _compute_thread();
    // }
  }


};


#endif // WIG_WDL_H
