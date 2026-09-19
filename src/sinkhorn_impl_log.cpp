
// actual implementation of the log-stabilized Sinkhorn (serial + threaded)
//
// The threaded path uses a persistent ThreadPool created once per call to
// compute_log(); every kernel works on a row range or column range of the
// implicit matrix R = C - f 1^T - 1 g^T without materializing it.

#include <cmath>
#include <vector>

#include "common.hpp"

#include "logdomain.hpp"
#include "sinkhorn_impl.hpp"
#include "vformat.hpp"

////////////////////////////////////////////////////////////////
// utilities: minrow, mincol (serial and threaded share the code)
////////////////////////////////////////////////////////////////

// row soft-min of R(f, g) -> _Rminrow (rows split across threads)
void Sinkhorn::_minrow(ThreadPool &pool, const double *f, const double *g) {
  const double *C = _C.data();
  const int M = (int)_M, N = (int)_N;
  double *out = _Rminrow.data();
  double *rowmin = _rowmin.data();
  double *rowsum = _rowsum.data();
  const double reg = _reg;
  pool.parallel_for(M, [=](int i0, int i1) {
    logdom::row_lse(C, M, N, f, g, reg, i0, i1, out, rowmin, rowsum);
  });
}

// column soft-min of R(f, g) -> _Rmincol (columns split across threads)
void Sinkhorn::_mincol(ThreadPool &pool, const double *f, const double *g) {
  const double *C = _C.data();
  const int M = (int)_M, N = (int)_N;
  double *out = _Rmincol.data();
  const double reg = _reg;
  pool.parallel_for(N, [=](int j0, int j1) {
    logdom::col_lse(C, M, N, f, g, reg, j0, j1, out);
  });
}

////////////////////////////////////////////////////////////////
// Algo 4.2/3.2: Log Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_log(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                           double reg, const int &n_threads) {
  // reset the counter
  _reset_counter();

  const la::idx M = C.nrow();
  const la::idx N = C.ncol();
  std::vector<la::idx> a_ind, b_ind;

  // convert problem to reduced form
  _reduce(a, b, C, a_ind, b_ind);
  _reg = reg;

  // scratch for the soft-min kernels
  _Rminrow.resize(_M);
  _Rmincol.resize(_N);
  _rowmin.resize(_M);
  _rowsum.resize(_M);
  _colmin.resize(_N);
  _colsum.resize(_N);

  // worker threads live for the whole computation
  ThreadPool pool(n_threads);

  // start the computation
  this->_fwd_log(pool);

  // recover original solution: P = exp(-R(f, g) / reg)
  _P.resize(_M, _N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Cj = _C.col(j);
    double *Pj = _P.col(j);
    for (la::idx i = 0; i < _M; ++i)
      Pj[i] = std::exp(-(Cj[i] - _u[i] - _v[j]) / _reg);
  }
  const double ninf = -std::numeric_limits<double>::infinity();
  this->u.resize(M, ninf);
  this->v.resize(N, ninf);
  for (la::idx i = 0; i < _M; ++i) this->u[a_ind[i]] = _u[i];
  for (la::idx j = 0; j < _N; ++j) this->v[b_ind[j]] = _v[j];

  // output P
  this->P.resize(M, N);
  for (la::idx j = 0; j < _N; ++j)
    for (la::idx i = 0; i < _M; ++i) this->P(a_ind[i], b_ind[j]) = _P(i, j);

  // record the loss for the reduced problem
  double l1 = 0.0, l2 = 0.0;
  for (la::idx k = 0; k < _P.size(); ++k) {
    l1 += _C[k] * _P[k];
    l2 += _P[k] * (std::log(_P[k]) - 1.0);
  }
  this->loss = l1 + _reg * l2;

  // ifgrad -> backward
  if (this->_withgrad) {
    this->_bwd_log(pool); // update `_grad_a`
    this->grad_a.resize(M);
    for (la::idx i = 0; i < _M; ++i) this->grad_a[a_ind[i]] = _grad_a[i];
  }

  _set_return_code();
}

// forward of log sinkhorn
void Sinkhorn::_fwd_log(ThreadPool &pool) {
  _u.resize(_M); // f
  _v.resize(_N); // g
  if (_withgrad) {
    // reserve space
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

  // logging for forward pass
  if (_verbose != 0) {
    rr::message(("Forward pass:"));
  }

  // update Rminrow
  _minrow(pool, _u.data(), _v.data());

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    rr::check_interrupt();
    this->iter++;
    if (_verbose != 0) {
      _timer.tic();
    }

    // update f
    for (la::idx i = 0; i < _M; ++i) _u[i] += _reg * _loga[i] + _Rminrow[i];
    if (_withgrad) {
      _uhist.push_back(_u);
    }

    // update g
    _mincol(pool, _u.data(), _v.data());
    for (la::idx j = 0; j < _N; ++j) _v[j] += _reg * _logb[j] + _Rmincol[j];
    if (_withgrad) {
      _vhist.push_back(_v);
    }

    // both soft-mins for the termination check (Rminrow is reused next iter)
    _minrow(pool, _u.data(), _v.data());
    _mincol(pool, _u.data(), _v.data());
    double e1 = 0.0, e2 = 0.0;
    for (la::idx i = 0; i < _M; ++i) {
      const double d = -_Rminrow[i] / _reg - _loga[i];
      e1 += d * d;
    }
    for (la::idx j = 0; j < _N; ++j) {
      const double d = -_Rmincol[j] / _reg - _logb[j];
      e2 += d * d;
    }
    this->err = std::sqrt(e1) + std::sqrt(e2);
    if (_verbose != 0) {
      _timer.toc();
    }

    // logging
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("forward", this->iter);
    }
  }
}

// backward (reverse) of log sinkhorn
void Sinkhorn::_bwd_log(ThreadPool &pool) {
  // adjoint of P dot P: PbarP = (C + reg * log(P)) % P
  // only its row sums and column sums (divided by reg) are needed
  la::Vec PbarP_rows(_M), PbarP_cols(_N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Cj = _C.col(j);
    const double *Pj = _P.col(j);
    double cs = 0.0;
    for (la::idx i = 0; i < _M; ++i) {
      const double p = (Cj[i] + _reg * std::log(Pj[i])) * Pj[i];
      PbarP_rows[i] += p;
      cs += p;
    }
    PbarP_cols[j] = cs;
  }

  la::Vec fbar(_M), gbar(_N), xM(_M), xN(_N);
  _E.resize(_M, _N);
  _grad_a.resize(_M);

  const double *C = _C.data();
  const int M = (int)_M, N = (int)_N;
  const double reg = _reg;
  double *E = _E.data();
  double *rowmin = _rowmin.data(), *rowsum = _rowsum.data();
  double *colmin = _colmin.data(), *colsum = _colsum.data();

  // logging for backward pass
  if (_verbose != 0) {
    rr::message(("Backward pass:"));
  }

  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) {
      _timer.tic();
    }
    const double *f = _uhist[l].data();
    const double *g = _vhist[l].data();

    // update adjoint of g
    if (l == this->iter) { // \ell = L
      for (la::idx j = 0; j < _N; ++j) gbar[j] = PbarP_cols[j] / _reg;
    } else { // \ell = L-1, \ldots, 1
      // X: row-stabilized softmax of -R/reg; gbar = -X^T fbar
      pool.parallel_for(M, [=](int i0, int i1) {
        logdom::row_min(C, M, N, f, g, i0, i1, rowmin);
      });
      pool.parallel_for(M, [=](int i0, int i1) {
        logdom::fill_E_rowstab(C, M, N, f, g, reg, rowmin, i0, i1, E, rowsum);
      });
      for (la::idx i = 0; i < _M; ++i) xM[i] = fbar[i] / _rowsum[i];
      const double *x = xM.data();
      double *y = gbar.data();
      pool.parallel_for(N, [=](int j0, int j1) {
        la::gemv(true, M, j1 - j0, -1.0, E + (std::size_t)j0 * M, M, x, 0.0,
                 y + j0);
      });
    }

    // update adjoint of f: W column-stabilized softmax; fbar = -W gbar
    pool.parallel_for(N, [=](int j0, int j1) {
      logdom::fill_E_colstab(C, M, N, f, g, reg, j0, j1, E, colmin, colsum);
    });
    for (la::idx j = 0; j < _N; ++j) xN[j] = gbar[j] / _colsum[j];
    {
      const double *x = xN.data();
      double *y = fbar.data();
      pool.parallel_for(M, [=](int i0, int i1) {
        la::gemv(false, i1 - i0, N, -1.0, E + i0, M, x, 0.0, y + i0);
      });
    }
    if (l == this->iter) {
      for (la::idx i = 0; i < _M; ++i) fbar[i] += PbarP_rows[i] / _reg;
    }

    if (_verbose != 0) {
      _timer.toc();
    }
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("backward", l);
    }

    // accumulate abar (_grad_a)
    for (la::idx i = 0; i < _M; ++i) _grad_a[i] += fbar[i] / _a[i];
  }
  for (la::idx i = 0; i < _M; ++i) _grad_a[i] *= _reg;

  // also revert the f and g before returning
  _u = _uhist[this->iter];
  _v = _vhist[this->iter];
}
