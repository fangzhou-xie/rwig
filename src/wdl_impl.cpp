// implementation of the WDL class (CPU)

#include <cmath>

#include "wdl_impl.hpp"        // wdl class
#include "barycenter_impl.hpp" // barycenter class
#include "optimizer.hpp"       // optimizers
#include "timer.hpp"           // TicToc timer
#include "vformat.hpp"         // vformat formatting the logging message

///////////////////////////////////////////////////////////////////
// Batched: process all D docs in a batch simultaneously
// Mirrors cuda wdl_batch — bypasses Barycenter class.
//
// Layout: the D docs of a batch are stacked column-wise, so every N x (S*D)
// matrix has column d*S+s for (doc d, topic s). Only three history buffers
// are kept: KVB (L slots), KTUB and bB (L+1 slots). The scaling matrices
// U and V of the forward pass are recovered elementwise from those:
//   UB[l+1] = A_tiled / KVB[l],   VB[l+1] = bB[l+1]_tiled / KTUB[l+1]
///////////////////////////////////////////////////////////////////

void WassersteinDictionaryLearning::_train_batch_batched(int batch_id) {
  Rcpp::checkUserInterrupt();

  // number of docs in this batch
  const int D = (batch_id == (int)(_M / _B)) ? (int)(_M % _B) : _B;
  const la::idx N = _N, S = _S;
  const la::idx SD = S * (la::idx)D;
  const la::idx NSD = N * SD, ND = N * (la::idx)D;
  const int L = _maxiter;

  // views into Y and W for this batch (contiguous columns)
  const double *bB_ext = _Y.col((la::idx)batch_id * _B); // N x D
  const double *wB = this->W.col((la::idx)batch_id * _B); // S x D

  // history slot pointers
  auto KVB_h = [&](int l) { return _KVB_hist.data() + (la::idx)l * NSD; };
  auto KTUB_h = [&](int l) { return _KTUB_hist.data() + (la::idx)l * NSD; };
  auto bB_h = [&](int l) { return _bB_hist.data() + (la::idx)l * ND; };

  // ---- FORWARD: fixed _maxiter iterations ----
  _VB.fill(1.0);
  for (int l = 0; l < L; ++l) {
    Rcpp::checkUserInterrupt();

    // KVB = K * VB  (one big GEMM: N x N * N x SD)
    _Kmul(false, _VB, _KVB, SD);
    std::copy(_KVB.data(), _KVB.data() + NSD, KVB_h(l));

    // UB[:,d*S+s] = A[:,s] / KVB[:,d*S+s]
    for (la::idx c = 0; c < SD; ++c) {
      const double *As = this->A.col(c % S);
      const double *KVc = _KVB.col(c);
      double *Uc = _UB.col(c);
      for (la::idx i = 0; i < N; ++i) Uc[i] = As[i] / KVc[i];
    }

    // KTUB = K^T * UB  (one big GEMM)
    _Kmul(true, _UB, _KTUB, SD);
    std::copy(_KTUB.data(), _KTUB.data() + NSD, KTUB_h(l + 1));

    // bB[:,d] = prod_s( KTUB[:,d*S+s] ^ wB[s,d] )
    for (int d = 0; d < D; ++d) {
      double *bd = _bB.col(d);
      for (la::idx i = 0; i < N; ++i) bd[i] = 1.0;
      for (la::idx s = 0; s < S; ++s) {
        const double *KTUc = _KTUB.col((la::idx)d * S + s);
        const double w = wB[s + (la::idx)d * S];
        for (la::idx i = 0; i < N; ++i) bd[i] *= std::pow(KTUc[i], w);
      }
    }
    std::copy(_bB.data(), _bB.data() + ND, bB_h(l + 1));

    // VB[:,d*S+s] = bB[:,d] / KTUB[:,d*S+s]
    for (la::idx c = 0; c < SD; ++c) {
      const double *bd = _bB.col(c / S);
      const double *KTUc = _KTUB.col(c);
      double *Vc = _VB.col(c);
      for (la::idx i = 0; i < N; ++i) Vc[i] = bd[i] / KTUc[i];
    }
  }

  // ---- BACKWARD: l = L down to 1 ----
  std::fill(_ABbar.data(), _ABbar.data() + NSD, 0.0);
  std::fill(_wBbar.data(), _wBbar.data() + SD, 0.0);

  for (int l = L; l > 0; --l) {
    Rcpp::checkUserInterrupt();

    const double *KTUBl = KTUB_h(l);
    const double *bBl = bB_h(l);

    if (l == L) {
      // bBbar = 2 * (bB_hist[L] - bB_ext)
      for (la::idx k = 0; k < ND; ++k) _bBbar[k] = 2 * (bBl[k] - bB_ext[k]);

      // UBbar = K * ((bBbar * wB^T tiled) % VB_hist[L]),
      // VB_hist[L] = bB_hist[L]_tiled / KTUB_hist[L]
      for (la::idx c = 0; c < SD; ++c) {
        const la::idx d = c / S, s = c % S;
        const double *bbd = _bBbar.col(d);
        const double *bd = bBl + d * N;
        const double *KTUc = KTUBl + c * N;
        const double w = wB[s + d * S];
        double *tc = _tmpB.col(c);
        for (la::idx i = 0; i < N; ++i) tc[i] = bbd[i] * w * (bd[i] / KTUc[i]);
      }
      _Kmul(false, _tmpB, _UBbar, SD);

    } else {
      // VBbar = -K^T * (UBbar % UB_hist[l+1] / KVB_hist[l]),
      // UB_hist[l+1] = A_tiled / KVB_hist[l]
      const double *KVBl = KVB_h(l);
      for (la::idx c = 0; c < SD; ++c) {
        const double *As = this->A.col(c % S);
        const double *KVc = KVBl + c * N;
        const double *Ubc = _UBbar.col(c);
        double *tc = _tmpB.col(c);
        for (la::idx i = 0; i < N; ++i)
          tc[i] = (Ubc[i] * (As[i] / KVc[i])) / KVc[i];
      }
      _Kmul(true, _tmpB, _VBbar, SD);
      for (la::idx k = 0; k < NSD; ++k) _VBbar[k] = -_VBbar[k];

      // bBbar[:,d] = sum_s( VBbar[:,d*S+s] / KTUB_hist[l][:,d*S+s] )
      for (int d = 0; d < D; ++d) {
        double *bbd = _bBbar.col(d);
        for (la::idx i = 0; i < N; ++i) bbd[i] = 0.0;
        for (la::idx s = 0; s < S; ++s) {
          const la::idx c = (la::idx)d * S + s;
          const double *Vbc = _VBbar.col(c);
          const double *KTUc = KTUBl + c * N;
          for (la::idx i = 0; i < N; ++i) bbd[i] += Vbc[i] / KTUc[i];
        }
      }

      // UBbar = K * ((bBbar*wB^T - VBbar/KTUB_hist[l]) % VB_hist[l]),
      // VB_hist[l] = bB_hist[l]_tiled / KTUB_hist[l]
      for (la::idx c = 0; c < SD; ++c) {
        const la::idx d = c / S, s = c % S;
        const double *bbd = _bBbar.col(d);
        const double *bd = bBl + d * N;
        const double *KTUc = KTUBl + c * N;
        const double *Vbc = _VBbar.col(c);
        const double w = wB[s + d * S];
        double *tc = _tmpB.col(c);
        for (la::idx i = 0; i < N; ++i)
          tc[i] = (bbd[i] * w - Vbc[i] / KTUc[i]) * (bd[i] / KTUc[i]);
      }
      _Kmul(false, _tmpB, _UBbar, SD);
    }

    // ABbar += UBbar / KVB_hist[l-1]
    {
      const double *KVBl1 = KVB_h(l - 1);
      for (la::idx k = 0; k < NSD; ++k) _ABbar[k] += _UBbar[k] / KVBl1[k];
    }

    // wBbar_d += log(KTUB_hist[l])^T * (bBbar_d % bB_hist[l]_d)
    for (int d = 0; d < D; ++d) {
      const double *bbd = _bBbar.col(d);
      const double *bd = bBl + (la::idx)d * N;
      double *wbd = _wBbar.col(d);
      for (la::idx s = 0; s < S; ++s) {
        const double *KTUc = KTUBl + ((la::idx)d * S + s) * N;
        double acc = 0.0;
        for (la::idx i = 0; i < N; ++i) acc += std::log(KTUc[i]) * (bbd[i] * bd[i]);
        wbd[s] += acc;
      }
    }
  }

  // ---- GRADIENT FINALIZATION ----

  // softmax Jacobian for A: for each column of ABbar
  for (la::idx c = 0; c < SD; ++c) _softmax_jac(this->A.col(c % S), _ABbar.col(c), N);

  // softmax Jacobian for w: for each doc
  for (int d = 0; d < D; ++d) _softmax_jac(wB + (la::idx)d * S, _wBbar.col(d), S);

  // reduce ABbar: sum D blocks of N x S into _g_Alpha
  _g_Alpha.zeros();
  for (la::idx c = 0; c < SD; ++c) {
    const double *src = _ABbar.col(c);
    double *dst = _g_Alpha.col(c % S);
    for (la::idx i = 0; i < N; ++i) dst[i] += src[i];
  }
  for (la::idx k = 0; k < _g_Alpha.size(); ++k) _g_Alpha[k] /= D;

  // reduce wBbar: sum D columns into _g_lambda
  _g_lambda.zeros();
  for (int d = 0; d < D; ++d) {
    const double *wbd = _wBbar.col(d);
    for (la::idx s = 0; s < S; ++s) _g_lambda[s] += wbd[s];
  }
  for (la::idx s = 0; s < S; ++s) _g_lambda[s] /= D;

  // broadcast g_Lambda = g_lambda * ones^T
  for (la::idx m = 0; m < _M; ++m) {
    double *col = _g_Lambda.col(m);
    for (la::idx s = 0; s < S; ++s) col[s] = _g_lambda[s];
  }
}

///////////////////////////////////////////////////////////////////
// Algo 7.2 (a): Wasserstein Dictionary Learning algorithm (serial)
///////////////////////////////////////////////////////////////////

// serial computation
void WassersteinDictionaryLearning::_compute_serial() {
  // logging at the beginning
  if (_verbose) {
    Rcpp::message(Rf_mkString("Running in serial mode..."));

    Rcpp::message(Rf_mkString(
        vformat(
            "Initializing WDL model with %i vocabs, %i docs, and %i topics...",
            (int)_N, (int)_M, (int)_S)
            .c_str()));
  }

  // init latent vars: random normal init from R's RNG (respects set.seed)
  _Alpha.resize(_N, _S);
  _Lambda.resize(_S, _M);
  for (la::idx k = 0; k < _Alpha.size(); ++k) _Alpha[k] = norm_rand();
  for (la::idx k = 0; k < _Lambda.size(); ++k) _Lambda[k] = norm_rand();

  // init A and W as softmax of alpha and lambda
  this->A.resize(_N, _S);
  this->W.resize(_S, _M);
  _softmax();

  // init the gradient into proper dimensions
  _g_Alpha.resize(_N, _S);
  _g_lambda.resize(_S);
  _g_Lambda.resize(_S, _M);

  // precompute Gibbs kernel K = exp(-C / reg)
  _K.resize(_N, _N);
  for (la::idx k = 0; k < _K.size(); ++k) _K[k] = std::exp(-_C[k] / _reg);
  _C_is_symm = _C.is_symmetric();
  if (_C_is_symm) {
    _K.symmetrize_upper();
  }

  // init the optimizers
  _opt_Alpha.init(_N, _S, _eta, _gamma, _beta1, _beta2, _eps);
  _opt_Lambda.init(_S, _M, _eta, _gamma, _beta1, _beta2, _eps);

  // batched scratch and history, allocated once for B docs
  {
    const la::idx B = (la::idx)std::min<int>(_B, (int)_M);
    const la::idx NSB = _N * _S * B, NB = _N * B;
    _UB.resize(_N, _S * B);
    _VB.resize(_N, _S * B);
    _KVB.resize(_N, _S * B);
    _KTUB.resize(_N, _S * B);
    _UBbar.resize(_N, _S * B);
    _VBbar.resize(_N, _S * B);
    _ABbar.resize(_N, _S * B);
    _tmpB.resize(_N, _S * B);
    _bB.resize(_N, B);
    _bBbar.resize(_N, B);
    _wBbar.resize(_S, B);
    _KVB_hist.resize((la::idx)_maxiter * NSB);
    _KTUB_hist.resize((la::idx)(_maxiter + 1) * NSB);
    _bB_hist.resize((la::idx)(_maxiter + 1) * NB);
  }

  // batch indicator: how many batches are there
  const int batches = (_M % _B) ? (int)(_M / _B + 1) : (int)(_M / _B);

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
  Barycenter bc((int)_S, false, _maxiter, _zerotol, 0);
  bc.update_C(_C);
  bc.update_reg(_reg);
  bc.update_A(this->A);

  // output (predicted barycenters)
  Yhat.resize(_N, _M);
  la::Vec what(_S);
  for (la::idx m = 0; m < _M; ++m) {
    const double *wm = this->W.col(m);
    for (la::idx s = 0; s < _S; ++s) what[s] = wm[s];

    bc.update_w(what);

    if (_sinkmode == 1) {
      bc.compute_parallel();
    } else if (_sinkmode == 2) {
      bc.compute_log(_n_threads);
    } else {
      Rcpp::stop("barycenter method not supported");
    }

    std::copy(bc.b.data(), bc.b.data() + _N, Yhat.col(m));
  }
}
