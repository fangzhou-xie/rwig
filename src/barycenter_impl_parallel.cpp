// Barycenter: shared helpers and the parallel (vanilla) algorithm

#include <cmath>

#include "barycenter_impl.hpp"

void Barycenter::_normalize_b_and_loss() {
  const double bsum = this->b.sum();
  for (la::idx i = 0; i < _N; ++i) this->b[i] /= bsum;
  if (_withgrad) {
    double l = 0.0;
    for (la::idx i = 0; i < _N; ++i) {
      const double d = this->b[i] - _b_ext[i];
      l += d * d;
    }
    this->loss = l;
  }
}

/////////////////////////////////////////////////////////////////////////
// Algo 6.1/5.1: Parallel Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

void Barycenter::compute_parallel() {
  _reset_counter();
  _fwd_parallel();
  if (_withgrad) _bwd_parallel(); // uses the un-normalized b
  _normalize_b_and_loss();
  _set_return_code();
}

// forward pass for parallel barycenter
void Barycenter::_fwd_parallel() {
  this->b.resize(_N);
  this->U.resize(_M, _S, 1.0);
  this->V.resize(_N, _S, 1.0);
  if (_withgrad) {
    _Uhist.clear();
    _Vhist.clear();
    _bhist.clear();
    _KVhist.clear();
    _KTUhist.clear();
    _Uhist.reserve(_maxiter + 1);
    _Vhist.reserve(_maxiter + 1);
    _bhist.reserve(_maxiter + 1);
    _KVhist.reserve(_maxiter + 1);
    _KTUhist.reserve(_maxiter + 1);
    _Uhist.push_back(this->U);
    _Vhist.push_back(this->V);
    _bhist.push_back(this->b);
    _KTUhist.push_back(la::Mat(_N, _S)); // slot 0 unused (K^T U^0 never needed)
  }

  _K.set(_C, _reg);
  _KV.resize(_M, _S);
  _KTU.resize(_N, _S);
  _K.mul(false, this->V, _KV);
  if (_withgrad) _KVhist.push_back(_KV); // K V^0
  _log_stage("Forward pass:");

  while (_keep_going()) {
    rr::check_interrupt();
    iter++;
    _tic();

    // update U = A / KV
    for (la::idx k = 0; k < this->U.size(); ++k) this->U[k] = _A[k] / _KV[k];
    if (_withgrad) _Uhist.push_back(this->U);

    // update b: b_i = prod_s KTU_is ^ w_s
    _K.mul(true, this->U, _KTU);
    if (_withgrad) _KTUhist.push_back(_KTU); // K^T U^l
    for (la::idx i = 0; i < _N; ++i) {
      double prod = 1.0;
      for (la::idx s = 0; s < _S; ++s) prod *= std::pow(_KTU(i, s), _w[s]);
      this->b[i] = prod;
    }
    if (_withgrad) _bhist.push_back(this->b);

    // update V = (b 1^T) / KTU
    for (la::idx s = 0; s < _S; ++s) {
      const double *KTUs = _KTU.col(s);
      double *Vs = this->V.col(s);
      for (la::idx i = 0; i < _N; ++i) Vs[i] = this->b[i] / KTUs[i];
    }
    if (_withgrad) _Vhist.push_back(this->V);

    // term cond: err = || U % KV - A ||_F
    _K.mul(false, this->V, _KV);
    if (_withgrad) _KVhist.push_back(_KV); // K V^l
    double e = 0.0;
    for (la::idx k = 0; k < this->U.size(); ++k) {
      const double d = this->U[k] * _KV[k] - _A[k];
      e += d * d;
    }
    err = std::sqrt(e);
    _toc_fwd();
  }
}

// backward pass for parallel barycenter
void Barycenter::_bwd_parallel() {
  this->grad_A.resize(_M, _S);
  this->grad_w.resize(_S);
  // adjoints for the intermediate vars
  la::Mat Ubar(_M, _S), Vbar(_N, _S), tmpNS(_N, _S), tmpMS(_M, _S);
  la::Vec bbar(_N), tmpN(_N);
  _log_stage("Backward pass:");

  for (int l = iter; l > 0; --l) {
    _tic();

    // K^T U^l and K V^l, K V^{l-1} were stored by the forward pass
    const la::Mat &KTU = _KTUhist[l];
    const la::Mat &KV = _KVhist[l];
    const la::Mat &KVprev = _KVhist[l - 1];

    if (l == iter) {
      for (la::idx i = 0; i < _N; ++i) bbar[i] = 2 * (this->b[i] - _b_ext[i]);
      // Ubar = K ((bbar w^T) % V)
      for (la::idx s = 0; s < _S; ++s) {
        const double *Vs = this->V.col(s);
        double *ts = tmpNS.col(s);
        for (la::idx i = 0; i < _N; ++i) ts[i] = bbar[i] * _w[s] * Vs[i];
      }
      _K.mul(false, tmpNS, Ubar);
    } else {
      // Vbar = -K^T ((Ubar % Uhist[l+1]) / (K Vhist[l]))
      for (la::idx k = 0; k < tmpMS.size(); ++k)
        tmpMS[k] = (Ubar[k] * _Uhist[l + 1][k]) / KV[k];
      _K.mul(true, tmpMS, Vbar);
      for (la::idx k = 0; k < Vbar.size(); ++k) Vbar[k] = -Vbar[k];
      // bbar = rowsum(Vbar / KTU)
      for (la::idx i = 0; i < _N; ++i) bbar[i] = 0.0;
      for (la::idx s = 0; s < _S; ++s) {
        const double *Vbs = Vbar.col(s);
        const double *KTUs = KTU.col(s);
        for (la::idx i = 0; i < _N; ++i) bbar[i] += Vbs[i] / KTUs[i];
      }
      // Ubar = K ((bbar w^T - Vbar / KTU) % Vhist[l])
      for (la::idx s = 0; s < _S; ++s) {
        const double *Vbs = Vbar.col(s);
        const double *KTUs = KTU.col(s);
        const double *Vs = _Vhist[l].col(s);
        double *ts = tmpNS.col(s);
        for (la::idx i = 0; i < _N; ++i)
          ts[i] = (bbar[i] * _w[s] - Vbs[i] / KTUs[i]) * Vs[i];
      }
      _K.mul(false, tmpNS, Ubar);
    }
    _toc_bwd(l);

    // grad_A += Ubar / (K Vhist[l-1])
    for (la::idx k = 0; k < Ubar.size(); ++k) this->grad_A[k] += Ubar[k] / KVprev[k];
    // grad_w += log(KTU)^T (bbar % bhist[l])
    for (la::idx i = 0; i < _N; ++i) tmpN[i] = bbar[i] * _bhist[l][i];
    for (la::idx s = 0; s < _S; ++s) {
      const double *KTUs = KTU.col(s);
      double acc = 0.0;
      for (la::idx i = 0; i < _N; ++i) acc += std::log(KTUs[i]) * tmpN[i];
      this->grad_w[s] += acc;
    }
  }
}
