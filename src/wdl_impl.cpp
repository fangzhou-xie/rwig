// implementation of the WDL class (serial)

#define ARMA_DONT_USE_OPENMP

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "barycenter_impl.hpp" // barycenter class
#include "wdl_impl.hpp"        // wdl class
#include "optimizer.hpp"       // optimizers
#include "vformat.hpp"         // vformat formatting the logging message
#include "timer.hpp"           // TicToc timer

// #include "ctrack.hpp"          // performance profiler (only for testing)

using namespace arma;

// void WassersteinDictionaryLearning::_train_batch_compute(
//   Barycenter& bc, vec& y, vec& w
// ) {
//   // CTRACK;
//
//   if (_sinkmode == 1) {
//     bc.compute_parallel();
//   } else if (_sinkmode == 2) {
//     bc.compute_log();
//   }
// }

// serially process each batch
void WassersteinDictionaryLearning::_train_batch_serial(
  Barycenter& bc,
  int batch_id
) {
  // CTRACK;

  Rcpp::checkUserInterrupt();

  // get the total number of docs within this current batch
  int n_docs = (batch_id == (_M / _B)) ? (_M % _B) : _B;

  // temp vecs
  vec y{vec(_N, fill::zeros)};
  vec w{vec(_S, fill::zeros)};

  // gradients to be accumulated within the batch
  mat g_Alpha{mat(_N, _S, fill::zeros)};
  vec g_lambda{vec(_S, fill::zeros)};

  for (int i = 0; i < n_docs; ++i) {

    // find the index
    int m = batch_id * _B + i;

    // retrieve the doc and weight
    y = _Y.col(m);
    w = this->W.col(m);

    // update y (b) and w in the barycenter class
    bc.update_b_ext(y);
    bc.update_w(w);

    if (_sinkmode == 1) {
      bc.compute_parallel();
    } else if (_sinkmode == 2) {
      bc.compute_log(_n_threads);
    } else {
      Rcpp::stop("barycenter method not supported");
    }
    // _train_batch_compute(bc, y, w);

    // // calculate the gradients
    // g_Alpha = bc.grad_A;  // N * S
    // g_lambda = bc.grad_w; // S * 1
    //
    // // multiply the jacobian of softmax
    // for (int s = 0; s < _S; ++s) {
    //   g_Alpha.col(s) = (
    //     diagmat(A.col(s)) - (A.col(s) * A.col(s).t())
    //   ).t() * g_Alpha.col(s);
    // }
    // g_lambda = (diagmat(w) - (w * w.t())) * g_lambda;
    //
    // // accumulate the gradients
    // _g_Alpha += g_Alpha;
    // _g_lambda += g_lambda;

    _accu_grad(bc.grad_A, bc.grad_w, w);

    // Rcpp::Rcout << "grad A: \n" << bc.grad_A << std::endl;
    // Rcpp::Rcout << "grad w: \n" << bc.grad_w << std::endl;
  }

  // calculate batch-average
  _g_Alpha /= n_docs;
  _g_lambda /= n_docs;
  // make sure the propagate g_Lambda
  _g_Lambda = _g_lambda * _onesM.t();
}


///////////////////////////////////////////////////////////////////
// Algo 7.2 (a): Wasserstein Dictionary Learning algorithm (serial)
///////////////////////////////////////////////////////////////////


// serial computation
void WassersteinDictionaryLearning::_compute_serial() {
  // CTRACK;

  // logging at the beginning
  if (_verbose) {
    Rcpp::message(Rf_mkString("Running in serial mode..."));

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
  // _Alpha = abs(mat(_N, _S, fill::randn));
  // _Lambda = abs(mat(_S, _M, fill::randn));
  _Alpha = mat(_N, _S, fill::randn);
  _Lambda = mat(_S, _M, fill::randn);
  // 2. unit init
  // _Alpha = mat(_N, _S, fill::ones);
  // _Lambda = mat(_S, _M, fill::ones);

  // init A and W as softmax of alpha and lambda
  _softmax();

  // Rcpp::Rcout << "A: \n" << this->A.rows(0, 5) << std::endl;
  // Rcpp::Rcout << "W: \n" << this->W.cols(0, 5) << std::endl;
  // Rcpp::Rcout << "A sum: \n" << sum(this->A, 0) << std::endl;
  // Rcpp::Rcout << "W sum: \n" << sum(this->W.cols(0,5), 0) << std::endl;

  // init the gradient into proper dimensions
  _g_Alpha = mat(_N, _S, fill::zeros);
  _g_lambda = vec(_S, fill::zeros);
  _g_Lambda = mat(_S, _M, fill::zeros);

  // init the barycenter class
  Barycenter bc(_S, true, _maxiter, _zerotol, 0);
  bc.update_C(_C);
  bc.update_reg(_reg);
  // bc.update_A(this->A);
  // bc.update_w();

  // init the optimizers
  // Optimizer opt_alpha(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
  // Optimizer opt_lambda(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);
  _opt_Alpha.init(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
  _opt_Lambda.init(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);

  // batch indicator: how many batches are there
  int batches = (_M % _B) ? (_M / _B + 1) : (_M / _B);

  // setup a counter for logging purpose
  // int cnt = 0;

  // logging before the main loop
  if (_verbose) {
    Rcpp::message(Rf_mkString(vformat(
      "Training WDL model with %i epochs, %i batches", _E, batches
    ).c_str()));
  }
  // Rcpp::checkUserInterrupt();

  // temp vars
  vec y{vec(_N)};
  vec w{vec(_S)};

  for (int e = 0; e < _E; ++e) { // START: one epoch

    for (int batch_id = 0; batch_id < batches; ++batch_id) { // START: one batch
      // cpp11::check_user_interrupt();
      Rcpp::checkUserInterrupt();

      _timer.tic();
      // if (_verbose) { _timer.tic(); }

      // clear gradients
      _g_Alpha.zeros();
      _g_lambda.zeros();

      // softmax and update A and W
      bc.update_A(this->A);

      // use serial method to compute the batch-avg gradients
      _train_batch_serial(bc, batch_id); // update _g_Alpha, _g_Lambda

      // run optimizer step
      _optimize(); // update _Alpha, _Lambda

      // update A and W after Alpha and Lambda
      _softmax(); // update this->A, this->W

      // Rcpp::Rcout << "A: \n" << this->A.rows(0, 5) << std::endl;
      // Rcpp::Rcout << "W: \n" << this->W.cols(0, 5) << std::endl;
      // Rcpp::Rcout << "A sum: \n" << sum(this->A, 0) << std::endl;
      // Rcpp::Rcout << "W sum: \n" << sum(this->W.cols(0,5), 0) << std::endl;

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
    } // END: one batch
  } // END: one epoch

  // after done with the training, now move on to the inference
  // if (_verbose) cpp11::message("Inference on the dataset\n");
  if (_verbose) {
    Rcpp::message(Rf_mkString("Inference on the dataset"));
  }

  // first update A and turnoff gradient computation
  bc.update_A(this->A); // update the final topics matrix
  bc.update_withgrad(false);

  // output (predicted barycenters)
  Yhat = mat(_N, _M, fill::zeros);
  vec what = vec(_S, fill::zeros); // tmp var for the weight
  for (int m = 0; m < _M; ++m) {
    what = this->W.col(m);

    bc.update_w(what); // update current weight w

    if (_sinkmode == 1) {
      bc.compute_parallel();
    } else if (_sinkmode == 2) {
      bc.compute_log(_n_threads);
    } else {
      Rcpp::stop("barycenter method not supported");
    }

    // write the barycenter into the column
    Yhat.col(m) = bc.b;
  }
}
