// implementation of the WDL class (serial)

#include "wdl_impl.hpp"        // wdl class
#include "barycenter_impl.hpp" // barycenter class
#include "optimizer.hpp"       // optimizers
#include "timer.hpp"           // TicToc timer
#include "vformat.hpp"         // vformat formatting the logging message

// #include "ctrack.hpp"          // performance profiler (only for testing)

using namespace arma;

// serially process each batch
void WassersteinDictionaryLearning::_train_batch_serial(Barycenter &bc,
                                                        int batch_id) {
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
// Batched: process all D docs in a batch simultaneously
// Mirrors cuda wdl_batch — bypasses Barycenter class
///////////////////////////////////////////////////////////////////

void WassersteinDictionaryLearning::_train_batch_batched(int batch_id) {
  Rcpp::checkUserInterrupt();

  // number of docs in this batch
  int D = (batch_id == (_M / _B)) ? (_M % _B) : _B;

  // views into Y and W for this batch (contiguous columns)
  mat bB_ext = _Y.cols(batch_id * _B, batch_id * _B + D - 1); // N x D
  mat wB = this->W.cols(batch_id * _B, batch_id * _B + D - 1); // S x D

  int SD = _S * D;

  // helper: build tiled A division — UB[:,d*S+s] = A[:,s] / KVB[:,d*S+s]
  // and tiled b division — VB[:,d*S+s] = bB[:,d] / KTUB[:,d*S+s]
  vec onesS(_S, fill::ones);

  // scaling vectors
  mat UB(_N, SD, fill::ones);
  mat VB(_N, SD, fill::ones);
  mat bB(_N, D, fill::zeros);
  mat KVB(_N, SD);
  mat KTUB(_N, SD);

  // history buffers (L+1 entries for U,V,b,KTUB; L entries for KVB)
  int L = _maxiter;
  std::vector<mat> UB_hist(L + 1);
  std::vector<mat> VB_hist(L + 1);
  std::vector<mat> bB_hist(L + 1);
  std::vector<mat> KVB_hist(L);
  std::vector<mat> KTUB_hist(L + 1);

  // save initial state
  UB_hist[0] = UB;
  VB_hist[0] = VB;
  bB_hist[0] = bB;

  // ---- FORWARD: fixed _maxiter iterations ----
  for (int l = 0; l < L; ++l) {
    Rcpp::checkUserInterrupt();

    // KVB = K * VB  (one big GEMM: N x N * N x SD)
    KVB = _K * VB;
    KVB_hist[l] = KVB;

    // UB[:,d*S+s] = A[:,s] / KVB[:,d*S+s]
    for (int d = 0; d < D; ++d) {
      UB.cols(d * _S, d * _S + _S - 1) =
          this->A / KVB.cols(d * _S, d * _S + _S - 1);
    }
    UB_hist[l + 1] = UB;

    // KTUB = K^T * UB  (one big GEMM)
    KTUB = _K.t() * UB;
    KTUB_hist[l + 1] = KTUB;

    // bB[:,d] = prod_s( KTUB[:,d*S+s] ^ wB[s,d] )
    for (int d = 0; d < D; ++d) {
      mat KTU_d = KTUB.cols(d * _S, d * _S + _S - 1); // N x S
      // element-wise power: each col s raised to wB(s,d)
      mat powered(_N, _S);
      for (int s = 0; s < _S; ++s) {
        powered.col(s) = pow(KTU_d.col(s), wB(s, d));
      }
      bB.col(d) = prod(powered, 1); // row-wise product → N x 1
    }
    bB_hist[l + 1] = bB;

    // VB[:,d*S+s] = bB[:,d] / KTUB[:,d*S+s]  (use pre-power KTUB)
    for (int d = 0; d < D; ++d) {
      VB.cols(d * _S, d * _S + _S - 1) =
          (bB.col(d) * onesS.t()) / KTUB_hist[l + 1].cols(d * _S, d * _S + _S - 1);
    }
    VB_hist[l + 1] = VB;
  }

  // ---- BACKWARD: l = L down to 1 ----
  mat UBbar(_N, SD, fill::zeros);
  mat VBbar(_N, SD, fill::zeros);
  mat bBbar(_N, D, fill::zeros);
  mat ABbar(_N, SD, fill::zeros);  // will be reduced to N x S
  mat wBbar(_S, D, fill::zeros);

  for (int l = L; l > 0; --l) {
    Rcpp::checkUserInterrupt();

    mat KTU_l = _K.t() * UB_hist[l]; // N x SD

    if (l == L) {
      // bBbar = 2 * (bB_hist[L] - bB_ext)
      bBbar = 2 * (bB_hist[l] - bB_ext);

      // UBbar = K * ((bBbar * wB^T tiled) % VB_hist[L])
      // build the tiled outer product: tmp[:,d*S+s] = bBbar[:,d] * wB[s,d]
      mat tmp(_N, SD);
      for (int d = 0; d < D; ++d) {
        tmp.cols(d * _S, d * _S + _S - 1) = bBbar.col(d) * wB.col(d).t();
      }
      UBbar = _K * (tmp % VB_hist[l]);

    } else {
      // VBbar = -K^T * (UBbar % UB_hist[l+1] / KVB_hist[l])
      VBbar = -_K.t() * ((UBbar % UB_hist[l + 1]) / KVB_hist[l]);

      // bBbar[:,d] = sum_s( VBbar[:,d*S+s] / KTUB_hist[l][:,d*S+s] )
      for (int d = 0; d < D; ++d) {
        bBbar.col(d) = sum(
            VBbar.cols(d * _S, d * _S + _S - 1) /
                KTUB_hist[l].cols(d * _S, d * _S + _S - 1),
            1);
      }

      // UBbar = K * ((bBbar*wB^T - VBbar/KTUB_hist[l]) % VB_hist[l])
      mat tmp(_N, SD);
      for (int d = 0; d < D; ++d) {
        tmp.cols(d * _S, d * _S + _S - 1) =
            bBbar.col(d) * wB.col(d).t() -
            VBbar.cols(d * _S, d * _S + _S - 1) /
                KTUB_hist[l].cols(d * _S, d * _S + _S - 1);
      }
      UBbar = _K * (tmp % VB_hist[l]);
    }

    // ABbar += UBbar / KVB_hist[l-1]
    ABbar += UBbar / KVB_hist[l - 1];

    // wBbar_d += log(KTUB_hist[l])^T * (bBbar_d % bB_hist[l]_d)
    for (int d = 0; d < D; ++d) {
      wBbar.col(d) +=
          log(KTUB_hist[l].cols(d * _S, d * _S + _S - 1)).t() *
          (bBbar.col(d) % bB_hist[l].col(d));
    }
  }

  // ---- GRADIENT FINALIZATION ----

  // softmax Jacobian for A: for each column of ABbar
  for (int j = 0; j < SD; ++j) {
    int s = j % _S;
    vec a_s = this->A.col(s);
    ABbar.col(j) = (diagmat(a_s) - a_s * a_s.t()) * ABbar.col(j);
  }

  // softmax Jacobian for w: for each doc
  for (int d = 0; d < D; ++d) {
    vec w_d = wB.col(d);
    wBbar.col(d) = (diagmat(w_d) - w_d * w_d.t()) * wBbar.col(d);
  }

  // reduce ABbar: sum D blocks of N x S into _g_Alpha
  _g_Alpha.zeros();
  for (int d = 0; d < D; ++d) {
    _g_Alpha += ABbar.cols(d * _S, d * _S + _S - 1);
  }
  _g_Alpha /= D;

  // reduce wBbar: sum D columns into _g_lambda
  _g_lambda = sum(wBbar, 1) / D;

  // broadcast g_Lambda
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

    Rcpp::message(Rf_mkString(
        vformat(
            "Initializing WDL model with %i vocabs, %i docs, and %i topics...",
            _N, _M, _S)
            .c_str()));
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

  // precompute Gibbs kernel K = exp(-C / reg)
  _K = exp(-_C / _reg);
  if (_C.is_symmetric()) {
    _K = symmatu(_K);
  }

  // init the optimizers
  _opt_Alpha.init(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
  _opt_Lambda.init(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);

  // batch indicator: how many batches are there
  int batches = (_M % _B) ? (_M / _B + 1) : (_M / _B);

  // logging before the main loop
  if (_verbose) {
    Rcpp::message(Rf_mkString(
        vformat("Training WDL model with %i epochs, %i batches", _E, batches)
            .c_str()));
  }

  for (int e = 0; e < _E; ++e) { // START: one epoch

    for (int batch_id = 0; batch_id < batches; ++batch_id) { // START: one batch
      Rcpp::checkUserInterrupt();

      if (_verbose) {
        Rcpp::message(Rf_mkString(vformat("Epoch %i of %i, batch %i of %i:",
                                          e + 1, _E, batch_id + 1, batches)
                                      .c_str()));
      }

      _timer.tic();

      // batched forward + backward for all docs in this batch
      _train_batch_batched(batch_id); // update _g_Alpha, _g_Lambda

      // run optimizer step
      _optimize(); // update _Alpha, _Lambda

      // update A and W after Alpha and Lambda
      _softmax(); // update this->A, this->W

      _timer.toc();
      // logging for each batch
      if (_verbose) {
        Rcpp::message(
            Rf_mkString(vformat("avg speed: %.2f sec, last speed: %.2f sec",
                                _timer.speed_avg(), _timer.speed_last())
                            .c_str()));
      }
    } // END: one batch
  } // END: one epoch

  // after done with the training, now move on to the inference
  if (_verbose) {
    Rcpp::message(Rf_mkString("Inference on the dataset"));
  }

  // init a barycenter class for inference (no gradients)
  Barycenter bc(_S, false, _maxiter, _zerotol, 0);
  bc.update_C(_C);
  bc.update_reg(_reg);
  bc.update_A(this->A);

  // output (predicted barycenters)
  Yhat = mat(_N, _M, fill::zeros);
  vec what = vec(_S, fill::zeros);
  for (int m = 0; m < _M; ++m) {
    what = this->W.col(m);

    bc.update_w(what);

    if (_sinkmode == 1) {
      bc.compute_parallel();
    } else if (_sinkmode == 2) {
      bc.compute_log(_n_threads);
    } else {
      Rcpp::stop("barycenter method not supported");
    }

    Yhat.col(m) = bc.b;
  }
}
