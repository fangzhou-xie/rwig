// actual implementation of the Sinkhorn vanilla algo
// only serial implementation

#include "common.hpp"

#include "sinkhorn_impl.hpp"
#include "vformat.hpp"

////////////////////////////////////////////////////////////////
// shared helpers (used by both vanilla and log)
////////////////////////////////////////////////////////////////

void Sinkhorn::_reduce(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                       std::vector<la::idx> &a_ind, std::vector<la::idx> &b_ind) {
  a_ind.clear();
  b_ind.clear();
  for (la::idx i = 0; i < a.size(); ++i)
    if (a[i] != 0) a_ind.push_back(i);
  for (la::idx j = 0; j < b.size(); ++j)
    if (b[j] != 0) b_ind.push_back(j);

  _M = a_ind.size();
  _N = b_ind.size();
  _a.resize(_M);
  _b.resize(_N);
  for (la::idx i = 0; i < _M; ++i) _a[i] = a[a_ind[i]];
  for (la::idx j = 0; j < _N; ++j) _b[j] = b[b_ind[j]];
  _C.resize(_M, _N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Cj = C.col(b_ind[j]);
    double *out = _C.col(j);
    for (la::idx i = 0; i < _M; ++i) out[i] = Cj[a_ind[i]];
  }
  _C_is_symm = _C.is_symmetric();
}

void Sinkhorn::_set_return_code() {
  if (this->err <= _zerotol) {
    this->return_code = 0;
  } else if (this->iter == _maxiter) {
    this->return_code = 1;
  } else {
    this->return_code = 2;
  }
}

void Sinkhorn::_log_iter(const char *stage, int it) {
  // first format the msg as c-string
  // convert c-string into SEXP and then print via Rcpp::message
  if (stage[0] == 'f') {
    Rcpp::message(Rf_mkString(
        vformat("iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f", it,
                this->err, _timer.speed_last(), _timer.speed_avg())
            .c_str()));
  } else {
    Rcpp::message(Rf_mkString(vformat("iter: %d, last speed: %.3f, avg speed: %.3f",
                                      it, _timer.speed_last(), _timer.speed_avg())
                                  .c_str()));
  }
}

////////////////////////////////////////////////////////////////
// Algo 4.1/3.1: Vanilla Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_vanilla(const la::Vec &a, const la::Vec &b,
                               const la::Mat &C, double reg) {
  // reset the counter
  _reset_counter();

  const la::idx M = C.nrow();
  const la::idx N = C.ncol();
  std::vector<la::idx> a_ind, b_ind;

  // convert problem to reduced form
  _reduce(a, b, C, a_ind, b_ind);
  _reg = reg;

  // Gibbs kernel
  _K.resize(_M, _N);
  for (la::idx k = 0; k < _K.size(); ++k) _K[k] = std::exp(-_C[k] / reg);
  if (_C_is_symm) {
    _K.symmetrize_upper();
  }

  // start the computation
  this->_fwd_vanilla();

  // recover original solution: P = diag(u) K diag(v)
  _P.resize(_M, _N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Kj = _K.col(j);
    double *Pj = _P.col(j);
    for (la::idx i = 0; i < _M; ++i) Pj[i] = _u[i] * Kj[i] * _v[j];
  }
  this->u.resize(M);
  this->v.resize(N);
  for (la::idx i = 0; i < _M; ++i) this->u[a_ind[i]] = _u[i];
  for (la::idx j = 0; j < _N; ++j) this->v[b_ind[j]] = _v[j];

  // output P
  P.resize(M, N);
  for (la::idx j = 0; j < _N; ++j)
    for (la::idx i = 0; i < _M; ++i) P(a_ind[i], b_ind[j]) = _P(i, j);

  // record the loss for the reduced problem
  double l1 = 0.0, l2 = 0.0;
  for (la::idx k = 0; k < _P.size(); ++k) {
    l1 += _C[k] * _P[k];
    l2 += _P[k] * (std::log(_P[k]) - 1.0);
  }
  this->loss = l1 + _reg * l2;

  // ifgrad -> backward
  if (this->_withgrad) {
    this->_bwd_vanilla(); // update `_grad_a`
    this->grad_a.resize(M);
    for (la::idx i = 0; i < _M; ++i) this->grad_a[a_ind[i]] = _grad_a[i];
  }

  _set_return_code();
}

////////////////////////////////////////////////////////////////
// Implementation of the private methods
////////////////////////////////////////////////////////////////

// forward of vanilla sinkhorn
void Sinkhorn::_fwd_vanilla() {
  _u.resize(_M, 1.0);
  _v.resize(_N, 1.0);
  if (_withgrad) {
    // reserve space
    _uhist.clear();
    _vhist.clear();
    _uhist.reserve(_maxiter + 1);
    _vhist.reserve(_maxiter + 1);

    _uhist.push_back(_u);
    _vhist.push_back(_v);
  }

  // logging for forward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }

  _Kv.resize(_M);
  _KTu.resize(_N);
  _Kmul(false, _v, _Kv);

  while ((this->iter < _maxiter) & (this->err >= _zerotol)) {
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) {
      _timer.tic();
    }

    for (la::idx i = 0; i < _M; ++i) _u[i] = _a[i] / _Kv[i];
    if (_withgrad) {
      _uhist.push_back(_u);
    }

    _Kmul(true, _u, _KTu);
    for (la::idx j = 0; j < _N; ++j) _v[j] = _b[j] / _KTu[j];
    if (_withgrad) {
      _vhist.push_back(_v);
    }

    _Kmul(false, _v, _Kv);
    // err = ||u % Kv - a||_2 + ||v % KTu - b||_2
    double e1 = 0.0, e2 = 0.0;
    for (la::idx i = 0; i < _M; ++i) {
      const double d = _u[i] * _Kv[i] - _a[i];
      e1 += d * d;
    }
    for (la::idx j = 0; j < _N; ++j) {
      const double d = _v[j] * _KTu[j] - _b[j];
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

// backward (reverse) of vanilla sinkhorn
void Sinkhorn::_bwd_vanilla() {
  // PbarK = (C + reg * log(P)) % K : adjoint of P dot K
  la::Mat PbarK(_M, _N);
  for (la::idx k = 0; k < PbarK.size(); ++k)
    PbarK[k] = (_C[k] + _reg * std::log(_P[k])) * _K[k];
  la::Vec ubar(_M), vbar(_N), tmpM(_M), tmpN(_N);
  this->_grad_a.resize(_M);

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) {
      _timer.tic();
    }

    if (l == this->iter) {
      // vbar = PbarK^T u
      la::gemv(true, PbarK, _u.data(), vbar.data());
      // ubar = PbarK v - K ((vbar % v) / (K^T u))
      _Kmul(true, _u, _KTu);
      for (la::idx j = 0; j < _N; ++j) tmpN[j] = (vbar[j] * _v[j]) / _KTu[j];
      la::gemv(false, PbarK, _v.data(), ubar.data());
      _Kmul(false, tmpN, tmpM);
      for (la::idx i = 0; i < _M; ++i) ubar[i] -= tmpM[i];
    } else {
      // vbar = -K^T ((ubar % uhist[l+1]) / (K vhist[l]))
      _Kmul(false, _vhist[l], _Kv);
      for (la::idx i = 0; i < _M; ++i)
        tmpM[i] = (ubar[i] * _uhist[l + 1][i]) / _Kv[i];
      _Kmul(true, tmpM, vbar);
      for (la::idx j = 0; j < _N; ++j) vbar[j] = -vbar[j];
      // ubar = -K ((vbar % vhist[l]) / (K^T uhist[l]))
      _Kmul(true, _uhist[l], _KTu);
      for (la::idx j = 0; j < _N; ++j)
        tmpN[j] = (vbar[j] * _vhist[l][j]) / _KTu[j];
      _Kmul(false, tmpN, ubar);
      for (la::idx i = 0; i < _M; ++i) ubar[i] = -ubar[i];
    }

    if (_verbose != 0) {
      _timer.toc();
    }
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("backward", l);
    }

    // accumulate the adjoint of a: grad_a += ubar / (K vhist[l-1])
    _Kmul(false, _vhist[l - 1], _Kv);
    for (la::idx i = 0; i < _M; ++i) this->_grad_a[i] += ubar[i] / _Kv[i];
  }
}
