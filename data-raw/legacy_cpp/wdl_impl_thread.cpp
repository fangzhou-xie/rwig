
// actual implementation of the WDL class in threading
// ERROR: too close to the limit

#define ARMA_DONT_USE_OPENMP

#include <thread>             // std::thread
#include <vector>             // std::vector

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "barycenter_impl.hpp" // barycenter class
#include "wdl_impl.hpp"        // wdl class
#include "optimizer.hpp"       // optimizers
#include "vformat.hpp"         // vformat formatting the logging message
#include "timer.hpp"           // TicToc timer

#include "ctrack.hpp"          // performance profiler (only for testing)

using namespace arma;


// threading process each batch
void WassersteinDictionaryLearning::_train_batch_thread(int batch_id) {
  // CTRACK;

  // Rcpp::checkUserInterrupt(); // check interruption

  // get the total number of docs within this current batch
  int n_docs = (batch_id == (_M / _B)) ? (_M % _B) : _B;

  // create the thread pool
  std::vector<std::thread> threads;
  // collect the gradients: length is batch size
  std::vector<arma::mat> res_g_Alpha(n_docs);
  std::vector<arma::vec> res_g_lambda(n_docs);

  // calculate the chunk size (number of docs) for each thread to handle
  int chunk_size = n_docs / _n_threads;

  // threading
  for (int t = 0; t < _n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (_n_threads - 1)) ? n_docs : (t + 1) * chunk_size;

    // set up worker for each thread
    threads.emplace_back([
      &res_g_Alpha, &res_g_lambda,
      &N = this->_N, &M = this->_M, &S = this->_S, &B = this->_B,
      &C = this->_C, &r = this->_reg, &Y = this->_Y,
      &A = this->A, &W = this->W,
      &sm = this->_sinkmode, &mi = this->_maxiter, &zt = this->_zerotol,
      batch_id, start, end
    ]() {

      // setup the barycenter class within each thread
      Barycenter bc(S, true, mi, zt, 0);
      bc.update_C(C);
      bc.update_reg(r);
      bc.update_A(A);

      // temp vecs
      vec y(N, fill::zeros);
      vec w(S, fill::zeros);

      // gradients to be accumulated within the thread (batch)
      mat g_Alpha(N, S, fill::zeros);
      vec g_lambda(S, fill::zeros);

      // compute gradient
      for (int i = start; i < end; ++i) {
        // find the index of the current doc in the whole dataset
        int m = batch_id * B + i; // this should never be above total M

        // retrieve the doc and weight
        y = Y.col(m);
        w = W.col(m);

        // update y (b) and w in the barycenter class
        bc.update_b_ext(y);
        bc.update_w(w);

        if (sm == 1) {
          bc.compute_parallel();
        } else if (sm == 2) {
          bc.compute_log(0);
        }

        // accumulate the gradients
        g_Alpha = bc.grad_A;  // N * S
        g_lambda = bc.grad_w; // S * 1
        // multiply the jacobian of softmax
        for (int s = 0; s < S; ++s) {
          g_Alpha.col(s) = (
            diagmat(A.col(s)) - (A.col(s) * (A.col(s)).t())
          ).t() * g_Alpha.col(s);
        }
        g_lambda *= (diagmat(w) - (w * w.t()));

        // write the result into the vector
        res_g_Alpha[i] = g_Alpha;
        res_g_lambda[i] = g_lambda;
      }
    });
  }

  // join the threads to collect all the gradients within batch
  for (auto& t : threads) { t.join(); }

  // calculate the average gradient within this batch
  for (int i = 0; i < n_docs; ++i) {
    _g_Alpha += res_g_Alpha[i];
    _g_lambda += res_g_lambda[i];
  }
  _g_Alpha /= n_docs;
  _g_lambda /= n_docs;
  // make sure the propagate g_Lambda
  _g_Lambda = _g_lambda * _onesM.t();
}

// inference in threading (even faster, no batching)
void WassersteinDictionaryLearning::_infer_thread() {
  CTRACK;

  // Rcpp::checkUserInterrupt(); // check interruption

  // create the thread pool
  std::vector<std::thread> threads;
  // collect the barycenters
  // std::vector<arma::vec> res_yhat(_M);
  mat yhat(_N, _M, fill::zeros);

  // calculate chunk size for each thread
  int chunk_size = _M / _n_threads;

  // threading
  for (int t = 0; t < _n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (_n_threads - 1)) ? _M : (t + 1) * chunk_size;

    // setup workers
    threads.emplace_back([
      &yhat,
      &N = this->_N, &M = this->_M, &S = this->_S, &B = this->_B,
      &C = this->_C, &r = this->_reg, &Y = this->_Y,
      &A = this->A, &W = this->W,
      &sm = this->_sinkmode, &mi = this->_maxiter, &zt = this->_zerotol,
      start, end
    ]() {
      // setup the barycenter class within each thread
      Barycenter bc(S, false, mi, zt, 0); // no need for gradient and b_ext (y)
      bc.update_C(C);
      bc.update_reg(r);
      bc.update_A(A);

      // temp vecs
      vec what{vec(S, fill::zeros)};

      // compute the barycenter
      for (int i = 0; i < end; ++i) {
        // retrieve the what from W, and update into barycenter class
        what = W.col(i);
        bc.update_w(what);

        if (sm == 1) {
          bc.compute_parallel();
        } else if (sm == 2) {
          bc.compute_log(0);
        }

        // write results into the vector
        yhat.col(i) = bc.b;
      }
    });
  }

  // join the threads to collect all the barycenters
  for (auto& t : threads) { t.join(); }

  // write the output into the Yhat
  this->Yhat = yhat;
}


///////////////////////////////////////////////////////////////////
// Algo 7.2 (b): Wasserstein Dictionary Learning algorithm (thread)
///////////////////////////////////////////////////////////////////


// threading computation
void WassersteinDictionaryLearning::_compute_thread() {
  CTRACK;

  // logging at the beginning
  if (_verbose) {
    Rcpp::message(Rf_mkString("Running in threading mode..."));

    Rcpp::message(Rf_mkString(vformat(
      "Initializing WDL model with %i vocabs, %i docs, and %i topics...",
      _N, _M, _S
    ).c_str()));
  }

  // first set up the RNG state
  // TODO: use static init instead of random init
  // arma::arma_rng rng_state;
  // rng_state.set_seed(_rng_seed);

  // init latent vars
  // 1. random init
  _Alpha = mat(_N, _S, fill::randn);
  _Lambda = mat(_S, _M, fill::randn);
  // 2. unit init
  // _Alpha = mat(_N, _S, fill::ones);
  // _Lambda = mat(_S, _M, fill::ones);

  // init A and W as softmax of alpha and lambda
  _softmax();

  // init the optimizers
  _opt_Alpha.init(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
  _opt_Lambda.init(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);

  // batch indicator: how many batches are there
  int batches = (_M % _B) ? (_M / _B + 1) : (_M / _B);

  // logging before the main loop
  if (_verbose) {
    Rcpp::message(Rf_mkString(vformat(
      "Training WDL model with %i epochs, %i batches", _E, batches
    ).c_str()));
  }

  // start training
  for (int e = 0; e < _E; ++e) {

    // Rcpp::Rcout << _timer.elapsed() << std::endl;

    for (int batch_id = 0; batch_id < batches; ++batch_id) {
      Rcpp::checkUserInterrupt();

      _timer.tic();

      // clear gradients
      _g_Alpha.zeros();
      _g_lambda.zeros();

      // use threading to compute the batch-avg gradients
      _train_batch_thread(batch_id); // _g_Alpha, _g_Lambda

      // run optimizer step
      _optimize(); // _Alpha, _Lambda

      // update A and W after Alpha and Lambda
      _softmax(); // A, W

      _timer.toc();
      // logging for each batch
      if (_verbose) {
        Rcpp::message(Rf_mkString(vformat(
          "Epoch: %i/%i, batch: %i/%i, avg speed: %.2f sec, gnorm A: %.3f, gnorm w: %.3f",
          e+1, _E, batch_id+1, batches, _timer.speed_avg(),
          norm(_g_Alpha, 2), norm(_g_lambda, 2)
          // norm(bc.grad_A, 2), norm(bc.grad_w, 2)
        ).c_str()));
      }
    }
  }

  // logging before inference
  if (_verbose) {
    Rcpp::message(Rf_mkString("Inference on the dataset"));
  }

  // run the parallel inference
  _infer_thread();

}


