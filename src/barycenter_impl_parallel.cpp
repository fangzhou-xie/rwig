
// actual implementation of the Barycenter class
// the parallel algo

// #ifndef ARMA_DONT_USE_OPENMP
// #define ARMA_DONT_USE_OPENMP
// #endif

// #include <iostream>

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "barycenter_impl.hpp"
#include "vformat.hpp"

// #include "ctrack.hpp"

using namespace arma;

/////////////////////////////////////////////////////////////////////////
// Algo 6.1/5.1: Parallel Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for parallel barycenter
void Barycenter::compute_parallel() {
  // CTRACK;

  // reset the counter
  _reset_counter();

  // set the ones variables
  _onesM = vec(_M, fill::ones);
  _onesN = vec(_N, fill::ones);
  _onesS = vec(_S, fill::ones);

  // forward loop for the barycenter computation
  _fwd_parallel();

  // backward loop for the gradients
  if (_withgrad) { _bwd_parallel(); }

  // determine return code
  if (this->err <= _zerotol) {
    this->return_code = 0;
  } else if (this->iter == _maxiter) {
    this->return_code = 1;
  } else {
    this->return_code = 2;
  }
}

/////////////////////////////////////////////////////////////////////////
// private methods for Barycenter parallel
/////////////////////////////////////////////////////////////////////////

// forward pass for parallel barycenter
void Barycenter::_fwd_parallel() {
  // CTRACK;

  // set/reset output b to zero
  this->b = vec(_N, fill::zeros);

  // set intermediate vars
  this->U = mat(_M, _S, fill::ones);
  this->V = mat(_N, _S, fill::ones);
  if (_withgrad) {
    _Uhist.push_back(this->U);
    _Vhist.push_back(this->V);
    _bhist.push_back(this->b);
  }

  // compute K
  _K = exp(- _C / _reg);
  if (_C.is_symmetric()) { _K = symmatu(_K); }

  mat onesNwT = _onesN * _w.t();

  // _KV = _K * this->V;

  // logging for forward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    // cpp11::check_user_interrupt();
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) { _timer.tic(); }

    // update U
    _KV = _K * this->V;
    this->U = _A / _KV;
    if (_withgrad) { _Uhist.push_back(this->U); }

    // update b
    _KTU = _K.t() * this->U;
    b = prod(pow(_KTU, onesNwT), 1);
    if (_withgrad) { _bhist.push_back(this->b); }

    // update V
    this->V = (this->b * _onesS.t()) / _KTU;
    if (_withgrad) { _Vhist.push_back(this->V); }

    // term cond
    // _KV = _K * this->V;
    this->err = norm((this->U % (_K * this->V)) - _A, 2);
    if (_verbose != 0) { _timer.toc(); }

    // logging
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
          this->iter, this->err,
          _timer.speed_last(), _timer.speed_avg()
      ).c_str()));

    }
  }
  // rescale b to 1
  this->b = this->b / accu(this->b);
  // compute the loss
  if (_withgrad) { this->loss = accu(pow(this->b - this->_b_ext, 2)); }
}

// backward pass for parallel barycenter
void Barycenter::_bwd_parallel() {
  // CTRACK;

  // gradients (adjoints) for the variables A and w
  this->grad_A = mat(_M, _S, fill::zeros);
  this->grad_w = vec(_S, fill::zeros);
  // adjoints for the intermediate vars
  mat Ubar{mat(_M, _S, fill::zeros)};
  mat Vbar{mat(_N, _S, fill::zeros)};
  vec bbar{vec(_N, fill::zeros)};
  mat KTU{mat(_N, _S, fill::zeros)};

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  // start the backward loop
  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) { _timer.tic(); }

    KTU = _K.t() * _Uhist[l];

    if (l == this->iter) {
      bbar = 2 * (this->b - this->_b_ext);
      Ubar = _K * ((bbar * _w.t()) % this->V);
    } else {
      Vbar = - _K.t() * ((Ubar % _Uhist[l+1]) / (_K * _Vhist[l]));
      bbar = sum(Vbar / KTU, 1);
      Ubar = _K * ((bbar * _w.t() - Vbar / KTU) % _Vhist[l]);
    }

    if (_verbose != 0) { _timer.toc(); }
    // logging
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, last speed: %.3f, avg speed: %.3f",
          l, _timer.speed_last(), _timer.speed_avg()
      ).c_str()));

    }

    // accumulate the adjoints of A and w
    this->grad_A += Ubar / (_K * _Vhist[l-1]);
    this->grad_w += log(KTU).t() * (bbar % _bhist[l]);
  }
}
