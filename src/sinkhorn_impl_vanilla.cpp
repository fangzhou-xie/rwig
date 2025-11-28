// actual implementation of the Sinkhorn vanilla algo
// only serial implementation

#include "common.hpp"

#include "sinkhorn_impl.hpp"
#include "vformat.hpp"

// #include "ctrack.hpp"
// #include "timer.hpp"

using namespace arma;

////////////////////////////////////////////////////////////////
// Algo 4.1/3.1: Vanilla Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_vanilla(const vec& a, const vec& b,
                               const mat& C, double reg) {
  // CTRACK;
  // reset the counter
  _reset_counter();

  int M = C.n_rows;
  int N = C.n_cols;
  // indices where a and b are not 0
  // uvec a_ind{find(a != 0)};
  // uvec b_ind{find(b != 0)};
  uvec a_ind = find(a != 0);
  uvec b_ind = find(b != 0);

  // convert problem to reduced form
  _a = a.elem(a_ind);
  _b = b.elem(b_ind);
  _C = C.submat(a_ind, b_ind);
  _C_is_symm = _C.is_symmetric();

  _reg = reg;
  _M = _C.n_rows;
  _N = _C.n_cols;
  _K = exp(- _C / reg);

  if (_C_is_symm) { _K = symmatu(_K); }

  // if (_withgrad) {
  //   // this->_uhist = std::vector<vec>(_maxiter); // FIXME: or plus 1?
  //   // this->_vhist = std::vector<vec>(_maxiter);
  //   this->_uhist = mat(_M, _maxiter+1);
  //   this->_vhist = mat(_N, _maxiter+1);
  // }

  // start the computation
  this->_fwd_vanilla();

  // recover original solution
  this->_P = diagmat(_u) * _K * diagmat(_v);
  this->u = vec(M, fill::zeros);
  this->v = vec(N, fill::zeros);
  this->u.elem(a_ind) = _u;
  this->v.elem(b_ind) = _v;

  // output P
  P = mat(M, N, fill::zeros);
  P.submat(a_ind, b_ind) = _P;

  // record the loss for the reduced problem
  this->loss = accu(_C % _P) + _reg * accu(_P % (log(_P) - 1));

  // ifgrad -> backward
  if (this->_withgrad) {
    this->_bwd_vanilla(); // update `_grad_a`
    this->grad_a = vec(M, fill::zeros);
    this->grad_a.elem( a_ind ) = _grad_a;
  }

  // determine return code
  if (this->err <= _zerotol) {
    this->return_code = 0;
  } else if (this->iter == _maxiter) {
    this->return_code = 1;
  } else {
    this->return_code = 2;
  }
}

////////////////////////////////////////////////////////////////
// Implementation of the private methods
////////////////////////////////////////////////////////////////

// forward of vanilla sinkhorn
void Sinkhorn::_fwd_vanilla() {
  // CTRACK;

  _u = vec(_M, fill::ones);
  _v = vec(_N, fill::ones);
  if (_withgrad) {
    // _uhist.col(this->iter) = _u;
    // _vhist.col(this->iter) = _v;
    _uhist.push_back(_u);
    _vhist.push_back(_v);
    // _Whist.push_back(mat(_M, _N, fill::zeros));
    // _Xhist.push_back(mat(_N, _M, fill::zeros));
    // _W = mat(_M, _N, fill::zeros);
    // _X = mat(_N, _M, fill::zeros);
  }

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }

  while ((this->iter < _maxiter) & (this->err >= _zerotol)) {
    // cpp11::check_user_interrupt();
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) { _timer.tic(); }

    _u = _a / (_K * _v);
    // if (_withgrad) { _uhist.col(this->iter) = _u; }
    if (_withgrad) { _uhist.push_back(_u); }

    _v = _b / (_K.t() * _u);
    // if (_withgrad) { _vhist.col(this->iter) = _v; }
    if (_withgrad) { _vhist.push_back(_v); }

    this->err = norm(_u % (_K*_v) - _a, 2) + norm(_v % (_K.t()*_u) - _b, 2);
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


      // convert c-string into SEXP and then print via Rcpp::message
      // Rcpp::message(Rf_mkString(_msg));

      // using cpp11 is easier
      // cpp11::message(
      //   "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
      //   this->iter, this->err,
      //   _timer.speed_last(), _timer.speed_avg()
      // );
    }
  }
}

// backward (reverse) of vanilla sinkhorn
void Sinkhorn::_bwd_vanilla() {
  // CTRACK;
  // backward loop
  mat PbarK = (_C + _reg * log(_P)) % _K; // adjoint of P dot K
  vec ubar{vec(_M, fill::zeros)};
  vec vbar{vec(_N, fill::zeros)};
  this->_grad_a = vec(_M, fill::zeros);

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) { _timer.tic(); }

    if (l == this->iter) {
      vbar = PbarK.t() * _u;
      ubar = PbarK * _v - _K * ((vbar % _v) / (_K.t() * _u));
    } else {
      vbar = - _K.t() * ((ubar % _uhist[l+1]) / (_K * _vhist[l]));
      ubar = - _K * ((vbar % _vhist[l]) / (_K.t() * _uhist[l]));
      // vbar = - _K.t() * diagmat(_uhist.col(l+1) / (_K * _vhist.col(l))) * ubar;
      // ubar = - _K * diagmat(_vhist.col(l) / (_K.t() * _uhist.col(l))) * vbar;
    }

    if (_verbose != 0) { _timer.toc(); }
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, last speed: %.3f, avg speed: %.3f",
          l, _timer.speed_last(), _timer.speed_avg()
      ).c_str()));

    }

    // accumulate the adjoint of a
    this->_grad_a += ubar / (_K * _vhist[l-1]);
  }
}
