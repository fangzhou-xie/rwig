
// implement the WDL algorithm in this header file
// https://arxiv.org/abs/2504.08722
// Section 7.2

#ifndef WIG_WDL_H
#define WIG_WDL_H

// #include <iostream>               // std::cout
// #include "R_ext/Print.h"          // for REprintf used for verbose logging

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "timer.h"                // TicToc timer class
#include "utils.h"                // using the utility functions
#include "barycenter.h"           // using the barycenter algorithms
#include "optimizer.h"            // using the optimizers: SGD, Adam, AdamW

using namespace arma;

class WassersteinDictionaryLearning {

private:

  int _S, _M, _N; // dims
  int _B, _E, _sinkmode, _opt, _maxiter, _rng_seed;
  double _zerotol, _eta, _gamma, _beta1, _beta2, _eps;
  bool _verbose;

  // latent vars
  mat _alpha, _lambda; // N*S, M*N
  vec _onesM;

  // TicToc timer class
  TicToc _timer;

  // init the data
  void init_data(mat& Y, mat& C, double reg, int S) {
    // load the data
    _Y = Y;
    _C = C;
    _reg = reg;

    // dimensions
    _M = _Y.n_cols;  // number of docs
    _N = _Y.n_rows;  // size of dictionary/words
    _S = S;          // nubmer of topics

    // helper vecs
    _onesM = vec(_M, fill::ones);
  }

  // softmax function for updating the A and W
  void softmax() {

    // update A with alpha
    vec exp_alphas = vec(_N, fill::zeros);
    for (int s = 0; s < _S; ++s) {
      exp_alphas = exp(_alpha.col(s) - _alpha.col(s).max());
      A.col(s) = exp_alphas / accu(exp_alphas);
    }
    // std::cout << "softmaxed A" << std::endl;

    // update W with lambda
    vec exp_lambdam = vec(_S, fill::zeros);
    for (int m = 0; m < _M; ++m) {
      exp_lambdam = exp(_lambda.col(m) - _lambda.col(m).max());
      W.col(m) = exp_lambdam / accu(exp_lambdam);
    }
    // std::cout << "softmaxed W" << std::endl;

  }

  // update the gradient of alpha
  void update_g_alpha(mat& g_alpha, vec& bhat, vec& y, mat& JbA) {

    mat JA_alphas(_N*_S, _N, fill::zeros);
    for (int s = 0; s < _S; ++s) {
      // reset the current block Jacobian
      JA_alphas.zeros();

      // update the current block of softmax Jacobian
      JA_alphas(_N,0,size(_N,_N)) =
        diagmat(A.col(s)) - A.col(s) * A.col(s).t();

      // update the s-th column of the gradient
      g_alpha.col(s) += (
        2 * (bhat - y).t() * JbA * JA_alphas
      ).t();
    }
  }

  // update the gradient of lambda
  void update_g_lambda(mat& g_lambda, vec& bhat, vec& y, mat& Jbw, vec& w) {
    // mat Jw_lambdas(_S, _S, fill::zeros);
    mat Jw_lambdas = diagmat(w) - w * w.t();
    g_lambda += (2 * (bhat - y).t() * Jbw * Jw_lambdas).t();
  }

  // actual implementation of WDL algorithm
  void compute_impl() {

    // first set up the RNG state
    // TODO: use static init instead of random init
    arma::arma_rng rng_state;
    rng_state.set_seed(_rng_seed);

    if (_verbose) {
      cpp11::message(
        "Initializing WDL model with %i vocabs, %i docs, and %i topics...\n",
        _N, _M, _S);
    }

    // init the latent vars
    _alpha = mat(_N, _S, fill::randn);
    _lambda = mat(_S, _M, fill::randn);
    // _alpha = mat(_N, _S, fill::ones);
    // _lambda = mat(_S, _M, fill::ones);
    // init A and W
    A = mat(_N, _S, fill::zeros);
    W = mat(_S, _M, fill::zeros);
    // softmax on alpha/lambda to have A/W
    softmax();

    // std::cout << "A: \n" << std::endl;
    // std::cout << A << std::endl;
    // std::cout << "W: \n" << std::endl;
    // std::cout << W << std::endl;

    // init the intermediate vars inside wdl algo
    mat g_alpha = mat(_N, _S, fill::zeros);
    vec g_lambda_vec = vec(_S, fill::zeros);
    mat g_lambda = mat(_S, _M, fill::zeros);

    // tmp vars for Jacobians
    // mat Jwl = mat(_S, _S, fill::zeros);
    // vec Glossb = vec(_N, fill::zeros);

    // barycenter class init
    Barycenter bc(_N, _N, _S, true, _maxiter, _zerotol);
    bc.update_C(_C);
    bc.update_reg(_reg);
    // bc.update_A(A);
    // std::cout << "maxiter: " << _maxiter << ", zerotol: " << _zerotol << std::endl;
    // init the optimizers
    Optimizer opt_alpha(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
    Optimizer opt_lambda(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);


    // batch indicator: how many batches are there
    int batches = _M / _B + 1;

    if (_verbose) {
      cpp11::message("Optimizing WDL model with %i epochs, %i batches\n",
               _E, batches);
    }
    cpp11::check_user_interrupt();

    for (int e = 0; e < _E; ++e) { // START: epochs
      cpp11::check_user_interrupt();

      for (int bi = 0; bi < batches; ++bi) { // START: one batch
        cpp11::check_user_interrupt();

        // if (_verbose) cpp11::message("Epoch %i, batch %i\n", e+1, bi+1);

        ////////////////////////////////////////////////////////////
        // 0. reset the gradients before each batch
        ////////////////////////////////////////////////////////////

        g_alpha.zeros();
        g_lambda_vec.zeros();

        // before each batch: update A
        bc.update_A(A);

        ////////////////////////////////////////////////////////////
        // 1. accumulate the gradient for each entry
        ////////////////////////////////////////////////////////////

        for (int bm = 0; bm < _B; ++bm) { // START: one doc in batch
          cpp11::check_user_interrupt();

          int m = bi * _B + bm; // index w.r.t. the entire corpus 0,...M-1
          if (m == _M) break; // this only happens in the last batch

          // get the doc/data and corresponding weight l
          vec yl = _Y.col(m);
          // vec lbdm = _lambda.col(m);
          vec w = W.col(m);

          // before each doc: update w
          bc.update_w(w);

          // std::cout << "before barycenter, sinkmode: " << _sinkmode << std::endl;

          // start the computation: must be 1 or 2, should be checked already
          _timer.tic();
          if (_sinkmode == 1) {
            bc.compute_parallel();
          } else if (_sinkmode == 2) {
            bc.compute_log();
          }
          _timer.toc();

          if (_verbose) {
            cpp11::message(
              "Epoch: %i/%i, batch: %i/%i, doc: %i/%i, iter: %.4i, err: %.2f\n, last speed: %.2f, avg speed: %.2f",
              e+1, _E,
              bi+1, _B,
              m+1, _M,
              bc._iter, bc._err,
              _timer.speed_last(),
              _timer.speed_avg()
            );
          }

          // access the Jacobians inside
          // TODO: fix the softmax
          // Glossb = 2 * (bc.b - yl);
          // g_alpha += (Glossb.t() * bc.JbA * ).t();
          // g_lambda_vec += (Glossb.t() * bc.Jbw * ).t();
          update_g_alpha(g_alpha, bc.b, yl, bc._JbA);
          update_g_lambda(g_lambda_vec, bc.b, yl, bc._Jbw, w);

          // std::cout << "g_alpha: \n" << g_alpha << std::endl;
          // std::cout << "g_lambda: \n" << g_lambda << std::endl;

        } // END: one doc in batch

        ////////////////////////////////////////////////////////////
        // 2. average gradients across the batch
        ////////////////////////////////////////////////////////////

        // after each batch, average gradient
        g_alpha /= _B;
        g_lambda_vec /= _B;
        g_lambda = g_lambda_vec * ones(_M).t();

        ////////////////////////////////////////////////////////////
        // 3. update the parameters with the mini-batch gradients
        ////////////////////////////////////////////////////////////

        // TODO: optimizer class in separate file (not in utils!)
        // FIXME: also easier to call

        // update the params by the optimizer
        if (_opt == 0) {                  // SGD
          opt_alpha.sgd(g_alpha);
          opt_lambda.sgd(g_lambda);
        } else if (_opt == 1) {           // Adam
          opt_alpha.adam(g_alpha);
          opt_lambda.adam(g_lambda);
        } else if (_opt == 2) {           // AdamW
          opt_alpha.adamw(g_alpha);
          opt_lambda.adamw(g_lambda);
        }

        ////////////////////////////////////////////////////////////
        // 4. update the latent variables A and W
        ////////////////////////////////////////////////////////////

        softmax();    // update A and W in-place

        // print at the end: first entry needs lots of inits (memory usage?)
        if (_verbose) {
          cpp11::message("Epoch %i, batch %i\n", e+1, bi+1);
        }

      } // END: one batch
    } // END: epochs

    if (_verbose) cpp11::message("Inference on the dataset\n");

    // TODO: re-run the barycenter based on the current A and W

    // first update A and turnoff Jacobian computation
    bc.update_A(A); // update the final topics matrix
    bc.update_withjac(false);

    Yhat = mat(_N, _M, fill::zeros);
    vec what = vec(_S, fill::zeros);
    // loop all the docs
    for (int m = 0; m < _M; ++m) {
      what = W.col(m);

      bc.update_w(what); // update the current weight w

      if (_sinkmode == 1) {
        bc.compute_parallel();
      } else if (_sinkmode == 2) {
        bc.compute_log();
      }

      // std::cout << bc.b << std::endl;

      // write the barycenter into the column
      Yhat.col(m) = bc.b;
    }
  }

public:
  mat _Y, _C; // data
  double _reg;
  mat A, W, Yhat; // output vars: topics, weights, predicted barycenters

  // init the class with parameters
  WassersteinDictionaryLearning(
    const int batch_size,
    const int epochs,
    const int sinkhorn_mode = 2,
    const int max_iter = 1000, const double zero_tol = 1e-6,
    const int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    const double beta1 = .9, const double beta2 = .999,
    const double eps = 1e-8,
    const int rng_seed = 123, const bool verbose = false
  ) {
    _B = batch_size;
    _E = epochs;
    _sinkmode = sinkhorn_mode;
    _maxiter = max_iter;
    _zerotol = zero_tol;
    _opt = optimizer;
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
    _rng_seed = rng_seed;
    _verbose = verbose;
  }

  ////////////////////////////////////////////////////////////////
  // Algo 7.1: Wasserstein Dictionary Learning algorithm
  ////////////////////////////////////////////////////////////////

  void compute(mat& Y, mat& C, double reg, int S) {
    // init the data
    init_data(Y, C, reg, S);

    // actually start the computation
    compute_impl();
  }

};


#endif // WIG_WDL_H
