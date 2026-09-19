// Sinkhorn: shared helpers and the vanilla algorithm (serial)

#include <cmath>

#include "sinkhorn_impl.hpp"

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
}

// scatter _u, _v, _P (and _grad_a) into the full-size outputs; entries
// outside the support get `u_fill` (0 for vanilla, -Inf for log)
void Sinkhorn::_expand(la::idx M, la::idx N, const std::vector<la::idx> &a_ind,
                       const std::vector<la::idx> &b_ind, double u_fill) {
  this->u.resize(M, u_fill);
  this->v.resize(N, u_fill);
  for (la::idx i = 0; i < _M; ++i) this->u[a_ind[i]] = _u[i];
  for (la::idx j = 0; j < _N; ++j) this->v[b_ind[j]] = _v[j];

  this->P.resize(M, N);
  for (la::idx j = 0; j < _N; ++j)
    for (la::idx i = 0; i < _M; ++i) this->P(a_ind[i], b_ind[j]) = _P(i, j);

  if (_withgrad) {
    this->grad_a.resize(M);
    for (la::idx i = 0; i < _M; ++i) this->grad_a[a_ind[i]] = _grad_a[i];
  }
}

// loss = <C, P> + reg * <P, log(P) - 1> on the reduced problem
void Sinkhorn::_compute_loss() {
  double l1 = 0.0, l2 = 0.0;
  for (la::idx k = 0; k < _P.size(); ++k) {
    l1 += _C[k] * _P[k];
    l2 += _P[k] * (std::log(_P[k]) - 1.0);
  }
  this->loss = l1 + _reg * l2;
}

////////////////////////////////////////////////////////////////
// Algo 4.1/3.1: Vanilla Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_vanilla(const la::Vec &a, const la::Vec &b,
                               const la::Mat &C, double reg) {
  _reset_counter();

  std::vector<la::idx> a_ind, b_ind;
  _reduce(a, b, C, a_ind, b_ind);
  _reg = reg;
  _K.set(_C, reg);

  _fwd_vanilla();

  // optimal coupling P = diag(u) K diag(v)
  _P.resize(_M, _N);
  for (la::idx j = 0; j < _N; ++j) {
    const double *Kj = _K.K.col(j);
    double *Pj = _P.col(j);
    for (la::idx i = 0; i < _M; ++i) Pj[i] = _u[i] * Kj[i] * _v[j];
  }
  _compute_loss();
  if (_withgrad) _bwd_vanilla(); // update `_grad_a`

  _expand(C.nrow(), C.ncol(), a_ind, b_ind, 0.0);
  _set_return_code();
}

// forward of vanilla sinkhorn
void Sinkhorn::_fwd_vanilla() {
  _u.resize(_M, 1.0);
  _v.resize(_N, 1.0);
  if (_withgrad) {
    _uhist.clear();
    _vhist.clear();
    _uhist.reserve(_maxiter + 1);
    _vhist.reserve(_maxiter + 1);
    _uhist.push_back(_u);
    _vhist.push_back(_v);
  }
  _log_stage("Forward pass:");

  _Kv.resize(_M);
  _KTu.resize(_N);
  _K.mul(false, _v, _Kv);

  while (_keep_going()) {
    rr::check_interrupt();
    iter++;
    _tic();

    for (la::idx i = 0; i < _M; ++i) _u[i] = _a[i] / _Kv[i];
    if (_withgrad) _uhist.push_back(_u);

    _K.mul(true, _u, _KTu);
    for (la::idx j = 0; j < _N; ++j) _v[j] = _b[j] / _KTu[j];
    if (_withgrad) _vhist.push_back(_v);

    _K.mul(false, _v, _Kv);
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
    err = std::sqrt(e1) + std::sqrt(e2);
    _toc_fwd();
  }
}

// backward (reverse) of vanilla sinkhorn
void Sinkhorn::_bwd_vanilla() {
  // PbarK = (C + reg * log(P)) % K : adjoint of P dot K
  la::Mat PbarK(_M, _N);
  for (la::idx k = 0; k < PbarK.size(); ++k)
    PbarK[k] = (_C[k] + _reg * std::log(_P[k])) * _K.K[k];
  la::Vec ubar(_M), vbar(_N), tmpM(_M), tmpN(_N);
  _grad_a.resize(_M);
  _log_stage("Backward pass:");

  for (int l = iter; l > 0; --l) {
    _tic();

    if (l == iter) {
      // vbar = PbarK^T u
      la::gemv(true, PbarK, _u.data(), vbar.data());
      // ubar = PbarK v - K ((vbar % v) / (K^T u))
      _K.mul(true, _u, _KTu);
      for (la::idx j = 0; j < _N; ++j) tmpN[j] = (vbar[j] * _v[j]) / _KTu[j];
      la::gemv(false, PbarK, _v.data(), ubar.data());
      _K.mul(false, tmpN, tmpM);
      for (la::idx i = 0; i < _M; ++i) ubar[i] -= tmpM[i];
    } else {
      // vbar = -K^T ((ubar % uhist[l+1]) / (K vhist[l]))
      _K.mul(false, _vhist[l], _Kv);
      for (la::idx i = 0; i < _M; ++i)
        tmpM[i] = (ubar[i] * _uhist[l + 1][i]) / _Kv[i];
      _K.mul(true, tmpM, vbar);
      for (la::idx j = 0; j < _N; ++j) vbar[j] = -vbar[j];
      // ubar = -K ((vbar % vhist[l]) / (K^T uhist[l]))
      _K.mul(true, _uhist[l], _KTu);
      for (la::idx j = 0; j < _N; ++j)
        tmpN[j] = (vbar[j] * _vhist[l][j]) / _KTu[j];
      _K.mul(false, tmpN, ubar);
      for (la::idx i = 0; i < _M; ++i) ubar[i] = -ubar[i];
    }
    _toc_bwd(l);

    // accumulate the adjoint of a: grad_a += ubar / (K vhist[l-1])
    _K.mul(false, _vhist[l - 1], _Kv);
    for (la::idx i = 0; i < _M; ++i) _grad_a[i] += ubar[i] / _Kv[i];
  }
}
