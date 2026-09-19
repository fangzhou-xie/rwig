// Barycenter: log-stabilized algorithm (serial and threaded share the code)

#include <cmath>

#include "barycenter_impl.hpp"

/////////////////////////////////////////////////////////////////////////
// Algo 6.2/5.2: Log Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

void Barycenter::compute_log(const int &n_threads) {
  _reset_counter();
  _scratch.resize((int)_M, (int)_N, _withgrad);

  ThreadPool pool(n_threads); // worker threads live for the whole computation
  _fwd_log(pool);
  if (_withgrad) _bwd_log(pool);
  _set_return_code();
}

// soft-mins per topic s of R(F[:,s], G[:,s])
void Barycenter::_minrow(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
                         la::Mat &out) {
  const logdom::Problem p = _prob();
  for (la::idx s = 0; s < _S; ++s)
    logdom::soft_min_rows(pool, p, F.col(s), G.col(s), out.col(s), _scratch);
}

void Barycenter::_mincol(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
                         la::Mat &out) {
  const logdom::Problem p = _prob();
  for (la::idx s = 0; s < _S; ++s)
    logdom::soft_min_cols(pool, p, F.col(s), G.col(s), out.col(s));
}

// forward pass for the log barycenter
void Barycenter::_fwd_log(ThreadPool &pool) {
  _logb.resize(_N);
  this->U.resize(_M, _S); // F
  this->V.resize(_N, _S); // G
  if (_withgrad) {
    _Uhist.clear();
    _Vhist.clear();
    _logbhist.clear();
    _Uhist.reserve(_maxiter + 1);
    _Vhist.reserve(_maxiter + 1);
    _logbhist.reserve(_maxiter + 1);
    _Uhist.push_back(this->U);  // F hist
    _Vhist.push_back(this->V);  // G hist
    _logbhist.push_back(_logb); // logb hist
  }
  _logA.resize(_M, _S);
  for (la::idx k = 0; k < _A.size(); ++k) _logA[k] = std::log(_A[k]);

  // _KV holds Rminrow (M x S), _KTU holds Rmincol (N x S)
  _KV.resize(_M, _S);
  _KTU.resize(_N, _S);
  la::Mat err_mat(_M, _S);
  _log_stage("Forward pass:");
  _minrow(pool, this->U, this->V, _KV);

  while (_keep_going()) {
    rr::check_interrupt();
    iter++;
    _tic();

    // update F
    for (la::idx k = 0; k < this->U.size(); ++k)
      this->U[k] += _reg * _logA[k] + _KV[k];
    if (_withgrad) _Uhist.push_back(this->U);

    // update logb = -(G + Rmincol) w / reg
    _mincol(pool, this->U, this->V, _KTU);
    for (la::idx j = 0; j < _N; ++j) _logb[j] = 0.0;
    for (la::idx s = 0; s < _S; ++s) {
      const double *Vs = this->V.col(s);
      const double *Rs = _KTU.col(s);
      for (la::idx j = 0; j < _N; ++j) _logb[j] += (Vs[j] + Rs[j]) * _w[s];
    }
    for (la::idx j = 0; j < _N; ++j) _logb[j] = -_logb[j] / _reg;
    if (_withgrad) _logbhist.push_back(_logb);

    // update G
    for (la::idx s = 0; s < _S; ++s) {
      double *Vs = this->V.col(s);
      const double *Rs = _KTU.col(s);
      for (la::idx j = 0; j < _N; ++j) Vs[j] += _reg * _logb[j] + Rs[j];
    }
    if (_withgrad) _Vhist.push_back(this->V);

    // err = || -Rminrow / reg - logA ||_2 (spectral norm, as arma::norm(mat, 2))
    _minrow(pool, this->U, this->V, _KV);
    for (la::idx k = 0; k < err_mat.size(); ++k)
      err_mat[k] = -_KV[k] / _reg - _logA[k];
    err = la::spectral_norm(err_mat);
    _toc_fwd();
  }

  // recover b from logb, rescale, loss
  this->b.resize(_N);
  for (la::idx j = 0; j < _N; ++j) this->b[j] = std::exp(_logb[j]);
  _normalize_b_and_loss();
}

// backward pass for the log barycenter
void Barycenter::_bwd_log(ThreadPool &pool) {
  const logdom::Problem p = _prob();
  this->grad_A.resize(_M, _S);
  this->grad_w.resize(_S);

  // adjoints of the intermediate vars
  la::Mat Fbar(_M, _S), Gbar(_N, _S);
  la::Vec logbbar(_N), y(_N);
  _log_stage("Backward pass:");

  for (int l = iter; l > 0; --l) {
    _tic();

    // adjoint of G (only for l < L): Gbar_s = -X^T Fbar_s, X from R(F^l_s, G^l_s)
    if (l != iter) {
      for (la::idx s = 0; s < _S; ++s) {
        logdom::apply_XT(pool, p, _Uhist[l].col(s), _Vhist[l].col(s), -1.0,
                         Fbar.col(s), Gbar.col(s), _scratch);
      }
    }

    // adjoint of logb
    if (l == iter) {
      for (la::idx j = 0; j < _N; ++j)
        logbbar[j] = 2 * (this->b[j] - _b_ext[j]) * this->b[j];
    } else {
      for (la::idx j = 0; j < _N; ++j) {
        double acc = 0.0;
        for (la::idx s = 0; s < _S; ++s) acc += Gbar(j, s);
        logbbar[j] = _reg * acc;
      }
    }

    // adjoint of F: Fbar_s = W y_s, W from R(F^l_s, G^{l-1}_s), with
    // y_s = (w_s / reg) logbbar - Gbar_s  (Gbar_s = 0 at l = L)
    for (la::idx s = 0; s < _S; ++s) {
      const double *g = _Vhist[l - 1].col(s);
      const double ws_reg = _w[s] / _reg;
      if (l == iter) {
        for (la::idx j = 0; j < _N; ++j) y[j] = logbbar[j];
      } else {
        const double *Gs = Gbar.col(s);
        for (la::idx j = 0; j < _N; ++j) y[j] = ws_reg * logbbar[j] - Gs[j];
      }
      logdom::apply_W(pool, p, _Uhist[l].col(s), g, 1.0, y.data(), Fbar.col(s),
                      _scratch);
      if (l == iter) {
        double *Fs = Fbar.col(s);
        for (la::idx i = 0; i < _M; ++i) Fs[i] *= ws_reg;
      }

      // adjoint of w (without reg scaling), reusing the column soft-min of
      // R(F^l, G^{l-1}) left in the scratch by apply_W:
      // grad_w[s] -= (G^{l-1}_s + Rmincol_s)^T logbbar
      const double *colmin = _scratch.colmin.data();
      const double *colsum = _scratch.colsum.data();
      double acc = 0.0;
      for (la::idx j = 0; j < _N; ++j) {
        const double rmin = colmin[j] - _reg * std::log(colsum[j]);
        acc += (g[j] + rmin) * logbbar[j];
      }
      this->grad_w[s] -= acc;
    }
    _toc_bwd(l);

    // adjoint of A (without reg scaling)
    for (la::idx k = 0; k < Fbar.size(); ++k) this->grad_A[k] += Fbar[k] / _A[k];
  }

  // scale per the formula
  for (la::idx k = 0; k < this->grad_A.size(); ++k) this->grad_A[k] *= _reg;
  for (la::idx s = 0; s < _S; ++s) this->grad_w[s] /= _reg;

  // revert F^L and G^L
  this->U = _Uhist[iter];
  this->V = _Vhist[iter];
}
