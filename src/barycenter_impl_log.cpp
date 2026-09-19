
// actual implementation of the Barycenter class: log-stabilized algo
// (serial and threaded share the same fused kernels)

#include <cmath>
#include <vector>

#include "common.hpp"

#include "barycenter_impl.hpp"
#include "logdomain.hpp"
#include "vformat.hpp"

/////////////////////////////////////////////////////////////////////////
// Algo 6.2/5.2: Log Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for log barycenter
void Barycenter::compute_log(const int &n_threads) {
  // reset the counter
  _reset_counter();

  // scratch for the soft-min kernels
  _rowmin.resize(_M);
  _rowsum.resize(_M);
  _colmin.resize(_N);
  _colsum.resize(_N);

  // worker threads live for the whole computation
  ThreadPool pool(n_threads);

  // forward loop for the barycenter computation
  _fwd_log(pool);

  // backward loop for the gradients
  if (_withgrad) {
    _bwd_log(pool);
  }

  _set_return_code();
}

/////////////////////////////////////////////////////////////////////////
// soft-min helpers: for every topic s, of R(F[:,s], G[:,s])
/////////////////////////////////////////////////////////////////////////

void Barycenter::_minrow(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
                         la::Mat &out) {
  const double *C = _C.data();
  const int M = (int)_M, N = (int)_N;
  const double reg = _reg;
  double *rowmin = _rowmin.data();
  double *rowsum = _rowsum.data();
  for (la::idx s = 0; s < _S; ++s) {
    const double *f = F.col(s);
    const double *g = G.col(s);
    double *o = out.col(s);
    pool.parallel_for(M, [=](int i0, int i1) {
      logdom::row_lse(C, M, N, f, g, reg, i0, i1, o, rowmin, rowsum);
    });
  }
  // output: out (R_min^row): _M * _S
}

void Barycenter::_mincol(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
                         la::Mat &out) {
  const double *C = _C.data();
  const int M = (int)_M, N = (int)_N;
  const double reg = _reg;
  for (la::idx s = 0; s < _S; ++s) {
    const double *f = F.col(s);
    const double *g = G.col(s);
    double *o = out.col(s);
    pool.parallel_for(N, [=](int j0, int j1) {
      logdom::col_lse(C, M, N, f, g, reg, j0, j1, o);
    });
  }
  // output: out (R_min^col): _N * _S
}

/////////////////////////////////////////////////////////////////////////
// forward / backward
/////////////////////////////////////////////////////////////////////////

// forward pass for the log barycenter
void Barycenter::_fwd_log(ThreadPool &pool) {
  // set logb
  _logb.resize(_N);
  // reset intermediate vars
  this->U.resize(_M, _S); // F
  this->V.resize(_N, _S); // G
  if (_withgrad) {
    // reserve space
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

  // setup _KV (Rminrow) and _KTU (Rmincol)
  _KV.resize(_M, _S);
  _KTU.resize(_N, _S);
  la::Mat err_mat(_M, _S);

  // logging for forward pass
  if (_verbose != 0) {
    rr::message(("Forward pass:"));
  }
  _minrow(pool, this->U, this->V, _KV); // update _KV

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    rr::check_interrupt();
    this->iter++;
    if (_verbose != 0) {
      _timer.tic();
    }

    // update F
    for (la::idx k = 0; k < this->U.size(); ++k)
      this->U[k] += _reg * _logA[k] + _KV[k];
    if (_withgrad) {
      _Uhist.push_back(this->U);
    }

    // update logb = -(G + Rmincol) w / reg
    _mincol(pool, this->U, this->V, _KTU); // update _KTU
    for (la::idx j = 0; j < _N; ++j) _logb[j] = 0.0;
    for (la::idx s = 0; s < _S; ++s) {
      const double *Vs = this->V.col(s);
      const double *Rs = _KTU.col(s);
      const double ws = _w[s];
      for (la::idx j = 0; j < _N; ++j) _logb[j] += (Vs[j] + Rs[j]) * ws;
    }
    for (la::idx j = 0; j < _N; ++j) _logb[j] = -_logb[j] / _reg;
    if (_withgrad) {
      _logbhist.push_back(_logb);
    }

    // update G
    for (la::idx s = 0; s < _S; ++s) {
      double *Vs = this->V.col(s);
      const double *Rs = _KTU.col(s);
      for (la::idx j = 0; j < _N; ++j) Vs[j] += _reg * _logb[j] + Rs[j];
    }
    if (_withgrad) {
      _Vhist.push_back(this->V);
    }

    // err = || -Rminrow / reg - logA ||_2 (spectral norm, as arma::norm(mat, 2))
    _minrow(pool, this->U, this->V, _KV); // update _KV
    for (la::idx k = 0; k < err_mat.size(); ++k)
      err_mat[k] = -_KV[k] / _reg - _logA[k];
    this->err = la::spectral_norm(err_mat);
    if (_verbose != 0) {
      _timer.toc();
    }

    // logging
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("forward", this->iter);
    }
  }
  // recover the b from logb and rescale
  this->b.resize(_N);
  double bsum = 0.0;
  for (la::idx j = 0; j < _N; ++j) {
    this->b[j] = std::exp(_logb[j]);
    bsum += this->b[j];
  }
  for (la::idx j = 0; j < _N; ++j) this->b[j] /= bsum;
  // compute the loss
  if (_withgrad) {
    double l = 0.0;
    for (la::idx j = 0; j < _N; ++j) {
      const double d = this->b[j] - _b_ext[j];
      l += d * d;
    }
    this->loss = l;
  }
}

// backward pass for the log barycenter
void Barycenter::_bwd_log(ThreadPool &pool) {
  // adjoints/gradients of the vars for output
  this->grad_A.resize(_M, _S);
  this->grad_w.resize(_S);

  // adjoints of the intermediate vars
  la::Mat Fbar(_M, _S), Gbar(_N, _S);
  la::Vec logbbar(_N), xM(_M), xN(_N);
  _E.resize(_M, _N);

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

    if (_verbose) {
      _timer.tic();
    }

    // first update adjoint of G only when l < this->iter
    if (l != this->iter) {
      // for each column in S: X row-stabilized softmax of R(F^l_s, G^l_s)
      for (la::idx s = 0; s < _S; ++s) {
        const double *f = _Uhist[l].col(s);
        const double *g = _Vhist[l].col(s);
        pool.parallel_for(M, [=](int i0, int i1) {
          logdom::row_min(C, M, N, f, g, i0, i1, rowmin);
        });
        pool.parallel_for(M, [=](int i0, int i1) {
          logdom::fill_E_rowstab(C, M, N, f, g, reg, rowmin, i0, i1, E,
                                 rowsum);
        });
        // Gbar_s = -X^T Fbar_s = -E^T (Fbar_s / rowsum)
        const double *Fs = Fbar.col(s);
        for (la::idx i = 0; i < _M; ++i) xM[i] = Fs[i] / rowsum[i];
        const double *x = xM.data();
        double *y = Gbar.col(s);
        pool.parallel_for(N, [=](int j0, int j1) {
          la::gemv(true, M, j1 - j0, -1.0, E + (std::size_t)j0 * M, M, x, 0.0,
                   y + j0);
        });
      }
    } // done updating Gbar

    // update adjoint of logb
    if (l == this->iter) {
      for (la::idx j = 0; j < _N; ++j)
        logbbar[j] = 2 * (this->b[j] - _b_ext[j]) * this->b[j];
    } else {
      for (la::idx j = 0; j < _N; ++j) {
        double acc = 0.0;
        for (la::idx s = 0; s < _S; ++s) acc += Gbar(j, s);
        logbbar[j] = _reg * acc;
      }
    } // done with logbbar

    // update adjoint of F: W column-stabilized softmax of R(F^l_s, G^{l-1}_s)
    for (la::idx s = 0; s < _S; ++s) {
      const double *f = _Uhist[l].col(s);
      const double *g = _Vhist[l - 1].col(s);
      pool.parallel_for(N, [=](int j0, int j1) {
        logdom::fill_E_colstab(C, M, N, f, g, reg, j0, j1, E, colmin, colsum);
      });
      const double ws_reg = _w[s] / _reg;
      double *Fs = Fbar.col(s);
      if (l == this->iter) {
        // Fbar_s = (w_s / reg) * (W logbbar)
        for (la::idx j = 0; j < _N; ++j) xN[j] = logbbar[j] / colsum[j];
      } else {
        // Fbar_s = W ((w_s / reg) logbbar - Gbar_s)
        const double *Gs = Gbar.col(s);
        for (la::idx j = 0; j < _N; ++j)
          xN[j] = (ws_reg * logbbar[j] - Gs[j]) / colsum[j];
      }
      const double *x = xN.data();
      pool.parallel_for(M, [=](int i0, int i1) {
        la::gemv(false, i1 - i0, N, 1.0, E + i0, M, x, 0.0, Fs + i0);
      });
      if (l == this->iter) {
        for (la::idx i = 0; i < _M; ++i) Fs[i] *= ws_reg;
      }

      // accumulate adjoints of w (without reg scaling), reusing the column
      // soft-min of R(F^l, G^{l-1}) just computed:
      // grad_w[s] -= (G^{l-1}_s + Rmincol_s)^T logbbar
      double acc = 0.0;
      for (la::idx j = 0; j < _N; ++j) {
        const double rmin = colmin[j] - _reg * std::log(colsum[j]);
        acc += (g[j] + rmin) * logbbar[j];
      }
      this->grad_w[s] -= acc;
    } // done updating Fbar

    if (_verbose) {
      _timer.toc();
    }
    // logging
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("backward", l);
    }

    // accumulate adjoints of A (without reg scaling)
    for (la::idx k = 0; k < Fbar.size(); ++k) this->grad_A[k] += Fbar[k] / _A[k];
  }

  // remember to scale it per the formula
  for (la::idx k = 0; k < this->grad_A.size(); ++k) this->grad_A[k] *= _reg;
  for (la::idx s = 0; s < _S; ++s) this->grad_w[s] /= _reg;

  // revert the F^{L} and G^{L}
  this->U = _Uhist[this->iter];
  this->V = _Vhist[this->iter];
}
