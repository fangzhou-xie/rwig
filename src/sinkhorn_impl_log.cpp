// Sinkhorn: log-stabilized algorithm (serial and threaded share the code;
// the pool is created once per call and every kernel works on a row or
// column range of the implicit matrix R = C - f 1^T - 1 g^T)

#include <cmath>
#include <limits>

#include "sinkhorn_impl.hpp"

////////////////////////////////////////////////////////////////
// Algo 4.2/3.2: Log Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_log(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                           double reg, const int &n_threads) {
  _reset_counter();

  std::vector<la::idx> a_ind, b_ind;
  _reduce(a, b, C, a_ind, b_ind);
  _reg = reg;
  _Rminrow.resize(_M);
  _Rmincol.resize(_N);
  _scratch.resize((int)_M, (int)_N, _withgrad);

  ThreadPool pool(n_threads); // worker threads live for the whole computation
  _fwd_log(pool);

  // optimal coupling P = exp(-R(f, g) / reg)
  _P.resize(_M, _N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Cj = _C.col(j);
    double *Pj = _P.col(j);
    for (la::idx i = 0; i < _M; ++i)
      Pj[i] = std::exp(-(Cj[i] - _u[i] - _v[j]) / _reg);
  }
  _compute_loss();
  if (_withgrad) _bwd_log(pool); // update `_grad_a`

  _expand(C.nrow(), C.ncol(), a_ind, b_ind,
          -std::numeric_limits<double>::infinity());
  _set_return_code();
}

// forward of log sinkhorn
void Sinkhorn::_fwd_log(ThreadPool &pool) {
  const logdom::Problem p = _prob();
  _u.resize(_M); // f
  _v.resize(_N); // g
  if (_withgrad) {
    _uhist.clear();
    _vhist.clear();
    _uhist.reserve(_maxiter + 1);
    _vhist.reserve(_maxiter + 1);
    _uhist.push_back(_u);
    _vhist.push_back(_v);
  }
  _loga.resize(_M);
  _logb.resize(_N);
  for (la::idx i = 0; i < _M; ++i) _loga[i] = std::log(_a[i]);
  for (la::idx j = 0; j < _N; ++j) _logb[j] = std::log(_b[j]);
  _log_stage("Forward pass:");

  logdom::soft_min_rows(pool, p, _u.data(), _v.data(), _Rminrow.data(), _scratch);

  while (_keep_going()) {
    rr::check_interrupt();
    iter++;
    _tic();

    // update f
    for (la::idx i = 0; i < _M; ++i) _u[i] += _reg * _loga[i] + _Rminrow[i];
    if (_withgrad) _uhist.push_back(_u);

    // update g
    logdom::soft_min_cols(pool, p, _u.data(), _v.data(), _Rmincol.data());
    for (la::idx j = 0; j < _N; ++j) _v[j] += _reg * _logb[j] + _Rmincol[j];
    if (_withgrad) _vhist.push_back(_v);

    // both soft-mins for the termination check (Rminrow is reused next iter)
    logdom::soft_min_rows(pool, p, _u.data(), _v.data(), _Rminrow.data(), _scratch);
    logdom::soft_min_cols(pool, p, _u.data(), _v.data(), _Rmincol.data());
    double e1 = 0.0, e2 = 0.0;
    for (la::idx i = 0; i < _M; ++i) {
      const double d = -_Rminrow[i] / _reg - _loga[i];
      e1 += d * d;
    }
    for (la::idx j = 0; j < _N; ++j) {
      const double d = -_Rmincol[j] / _reg - _logb[j];
      e2 += d * d;
    }
    err = std::sqrt(e1) + std::sqrt(e2);
    _toc_fwd();
  }
}

// backward (reverse) of log sinkhorn
void Sinkhorn::_bwd_log(ThreadPool &pool) {
  const logdom::Problem p = _prob();

  // adjoint of P dot P: PbarP = (C + reg * log(P)) % P;
  // only its row sums and column sums (divided by reg) are needed
  la::Vec PbarP_rows(_M), PbarP_cols(_N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Cj = _C.col(j);
    const double *Pj = _P.col(j);
    double cs = 0.0;
    for (la::idx i = 0; i < _M; ++i) {
      const double v = (Cj[i] + _reg * std::log(Pj[i])) * Pj[i];
      PbarP_rows[i] += v;
      cs += v;
    }
    PbarP_cols[j] = cs;
  }

  la::Vec fbar(_M), gbar(_N);
  _grad_a.resize(_M);
  _log_stage("Backward pass:");

  for (int l = iter; l > 0; --l) {
    _tic();
    const double *f = _uhist[l].data();
    const double *g = _vhist[l].data();

    // adjoint of g: gbar = -X^T fbar  (\ell < L), or PbarP^T 1 / reg (\ell = L)
    if (l == iter) {
      for (la::idx j = 0; j < _N; ++j) gbar[j] = PbarP_cols[j] / _reg;
    } else {
      logdom::apply_XT(pool, p, f, g, -1.0, fbar.data(), gbar.data(), _scratch);
    }

    // adjoint of f: fbar = -W gbar (+ PbarP 1 / reg at \ell = L)
    logdom::apply_W(pool, p, f, g, -1.0, gbar.data(), fbar.data(), _scratch);
    if (l == iter) {
      for (la::idx i = 0; i < _M; ++i) fbar[i] += PbarP_rows[i] / _reg;
    }
    _toc_bwd(l);

    // accumulate abar (_grad_a)
    for (la::idx i = 0; i < _M; ++i) _grad_a[i] += fbar[i] / _a[i];
  }
  for (la::idx i = 0; i < _M; ++i) _grad_a[i] *= _reg;

  // also revert the f and g before returning
  _u = _uhist[iter];
  _v = _vhist[iter];
}
