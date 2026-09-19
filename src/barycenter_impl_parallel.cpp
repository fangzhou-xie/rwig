
// actual implementation of the Barycenter class
// the parallel algo

#include <cmath>

#include "common.hpp"

#include "barycenter_impl.hpp"
#include "vformat.hpp"

void Barycenter::_log_iter(const char *stage, int it) {
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

/////////////////////////////////////////////////////////////////////////
// Algo 6.1/5.1: Parallel Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for parallel barycenter
void Barycenter::compute_parallel() {
  // reset the counter
  _reset_counter();

  // forward loop for the barycenter computation
  _fwd_parallel();

  // backward loop for the gradients (uses the un-normalized b)
  if (_withgrad) {
    _bwd_parallel();
  }

  // rescale b to 1
  const double bsum = this->b.sum();
  for (la::idx i = 0; i < _N; ++i) this->b[i] /= bsum;
  // compute the loss
  if (_withgrad) {
    double l = 0.0;
    for (la::idx i = 0; i < _N; ++i) {
      const double d = this->b[i] - _b_ext[i];
      l += d * d;
    }
    this->loss = l;
  }

  _set_return_code();
}

/////////////////////////////////////////////////////////////////////////
// private methods for Barycenter parallel
/////////////////////////////////////////////////////////////////////////

// forward pass for parallel barycenter
void Barycenter::_fwd_parallel() {
  // set/reset output b to zero
  this->b.resize(_N);

  // set intermediate vars
  this->U.resize(_M, _S, 1.0);
  this->V.resize(_N, _S, 1.0);
  if (_withgrad) {
    // reserve the space
    _Uhist.clear();
    _Vhist.clear();
    _bhist.clear();
    _Uhist.reserve(_maxiter + 1);
    _Vhist.reserve(_maxiter + 1);
    _bhist.reserve(_maxiter + 1);

    _Uhist.push_back(this->U);
    _Vhist.push_back(this->V);
    _bhist.push_back(this->b);
  }

  // compute K
  _K.resize(_M, _N);
  for (la::idx k = 0; k < _K.size(); ++k) _K[k] = std::exp(-_C[k] / _reg);
  if (_C_is_symm) {
    _K.symmetrize_upper();
  }

  // set size for _KV, _KTU
  _KV.resize(_M, _S);
  _KTU.resize(_N, _S);

  _Kmul(false, this->V, _KV);

  // logging for forward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) {
      _timer.tic();
    }

    // update U = A / KV
    for (la::idx k = 0; k < this->U.size(); ++k) this->U[k] = _A[k] / _KV[k];
    if (_withgrad) {
      _Uhist.push_back(this->U);
    }

    // update b: b_i = prod_s KTU_is ^ w_s
    _Kmul(true, this->U, _KTU);
    for (la::idx i = 0; i < _N; ++i) {
      double p = 1.0;
      for (la::idx s = 0; s < _S; ++s) p *= std::pow(_KTU(i, s), _w[s]);
      this->b[i] = p;
    }
    if (_withgrad) {
      _bhist.push_back(this->b);
    }

    // update V = (b 1^T) / KTU
    for (la::idx s = 0; s < _S; ++s) {
      const double *KTUs = _KTU.col(s);
      double *Vs = this->V.col(s);
      for (la::idx i = 0; i < _N; ++i) Vs[i] = this->b[i] / KTUs[i];
    }
    if (_withgrad) {
      _Vhist.push_back(this->V);
    }

    // term cond: err = || U % KV - A ||_F
    _Kmul(false, this->V, _KV);
    double e = 0.0;
    for (la::idx k = 0; k < this->U.size(); ++k) {
      const double d = this->U[k] * _KV[k] - _A[k];
      e += d * d;
    }
    this->err = std::sqrt(e);
    if (_verbose != 0) {
      _timer.toc();
    }

    // logging
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("forward", this->iter);
    }
  }
}

// backward pass for parallel barycenter
void Barycenter::_bwd_parallel() {
  // gradients (adjoints) for the variables A and w
  this->grad_A.resize(_M, _S);
  this->grad_w.resize(_S);
  // adjoints for the intermediate vars
  la::Mat Ubar(_M, _S), Vbar(_N, _S), KTU(_N, _S), KV(_M, _S), tmpNS(_N, _S),
      tmpMS(_M, _S);
  la::Vec bbar(_N), tmpN(_N);

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  // start the backward loop
  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) {
      _timer.tic();
    }

    // KTU = K^T Uhist[l]
    _Kmul(true, _Uhist[l], KTU);

    if (l == this->iter) {
      for (la::idx i = 0; i < _N; ++i) bbar[i] = 2 * (this->b[i] - _b_ext[i]);
      // Ubar = K ((bbar w^T) % V)
      for (la::idx s = 0; s < _S; ++s) {
        const double *Vs = this->V.col(s);
        double *ts = tmpNS.col(s);
        for (la::idx i = 0; i < _N; ++i) ts[i] = bbar[i] * _w[s] * Vs[i];
      }
      _Kmul(false, tmpNS, Ubar);
    } else {
      // Vbar = -K^T ((Ubar % Uhist[l+1]) / (K Vhist[l]))
      _Kmul(false, _Vhist[l], KV);
      for (la::idx k = 0; k < tmpMS.size(); ++k)
        tmpMS[k] = (Ubar[k] * _Uhist[l + 1][k]) / KV[k];
      _Kmul(true, tmpMS, Vbar);
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
      _Kmul(false, tmpNS, Ubar);
    }

    if (_verbose != 0) {
      _timer.toc();
    }
    // logging
    if ((_verbose != 0) && ((this->iter - 1) % _verbose) == 0) {
      _log_iter("backward", l);
    }

    // accumulate the adjoints of A and w
    // grad_A += Ubar / (K Vhist[l-1])
    _Kmul(false, _Vhist[l - 1], KV);
    for (la::idx k = 0; k < Ubar.size(); ++k) this->grad_A[k] += Ubar[k] / KV[k];
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
