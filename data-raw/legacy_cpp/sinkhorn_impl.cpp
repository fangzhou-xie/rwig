
// actual implementation of the Sinkhorn class

#include <cpp11armadillo.hpp>

#include "ctrack.hpp"

#include "sinkhorn.hpp"

using namespace arma;


////////////////////////////////////////////////////////////////
// Algo 4.1/3.1: Vanilla Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_vanilla(vec& a, vec& b, mat& C, double reg) {
  CTRACK;
  // reset the counter
  reset_counter();

  int M = C.n_rows;
  int N = C.n_cols;
  // indices where a and b are not 0
  uvec _a_ind = find(a != 0);
  uvec _b_ind = find(b != 0);

  // convert problem to reduced form
  _a = a.elem(_a_ind);
  _b = b.elem(_b_ind);
  _C = C.submat(_a_ind, _b_ind);

  // std::cout << "_a: \n" << _a << std::endl;
  // std::cout << "_b: \n" << _b << std::endl;
  // std::cout << "_C: \n" << _C << std::endl;

  _reg = reg;
  _M = _C.n_rows;
  _N = _C.n_cols;

  // start the computation
  impl_vanilla();

  // optimal coupling for reduce form
  mat P = diagmat(_u) * _K * diagmat(_v);
  // std::cout << "P: \n" << P << std::endl;
  // compute the grad
  if (_withgrad) {
    const vec logu = log(_u);
    const vec logv = log(_v);
    _grad_a = _reg * (
      _v.t() * (diagmat(logu) * _K + _K * diagmat(logv)).t() * _Ju +
        _u.t() * (diagmat(logu) * _K + _K * diagmat(logv)) * _Jv
    ).t();
  }

  // recover the original solution
  // recover the u,v
  vec u = vec(M, fill::zeros);
  vec v = vec(N, fill::zeros);
  u.elem( _a_ind ) = _u;
  v.elem( _b_ind ) = _v;
  // _u = u;
  // _v = v;
  _u.swap(u);
  _v.swap(v);
  _K = exp(-C/reg);

  if (_withgrad) {
    vec grad_a = vec(M, fill::zeros);
    grad_a(_a_ind) = _grad_a;
    _grad_a = grad_a;
  }
  // _P = diagmat(_u) * _K * diagmat(_v);
  _P = mat(M, N, fill::zeros);
  _P.submat(_a_ind, _b_ind) = P;

  // ctrack::result_print();

}

////////////////////////////////////////////////////////////////
// Algo 4.3/3.3: Log Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_log(vec& a, vec& b, mat& C, double reg) {
  CTRACK;

  // reset the counter
  reset_counter();

  int M = C.n_rows;
  int N = C.n_cols;
  // indices where a and b are not 0
  uvec _a_ind = find(a != 0);
  uvec _b_ind = find(b != 0);

  // convert problem to reduced form
  _a = a.elem(_a_ind);
  _b = b.elem(_b_ind);
  _C = C.submat(_a_ind, _b_ind);

  _reg = reg;
  _M = _C.n_rows;
  _N = _C.n_cols;
  _onesM = vec(_M, fill::ones);
  _onesN = vec(_N, fill::ones);

  // start the computation
  impl_log();

  mat P = exp(- _K / _reg);
  // compute the grad
  if (_withgrad) {
    _grad_a = (
      (_onesN.t() * P.t() * diagmat(_u) + _v.t() * P.t()) * _Ju +
        (_onesM.t() * P * diagmat(_v) + _u.t() * P) * _Jv
    ).t() / _reg;
  }

  // recover the u,v
  vec u = vec(M, fill::value(-datum::inf));
  vec v = vec(N, fill::value(-datum::inf));
  // u.fill(-datum::inf);
  // v.fill(-datum::inf);
  u.elem( _a_ind ) = _u;
  v.elem( _b_ind ) = _v;
  // _u = u;
  // _v = v;
  _u.swap(u);
  _v.swap(v);

  if (_withgrad) {
    vec grad_a = vec(M, fill::zeros);
    grad_a(_a_ind) = _grad_a;
    // _grad_a = grad_a;
    _grad_a.swap(grad_a);
  }

  // recover the optimal coupling
  _P = mat(M, N, fill::zeros);
  _P.submat(_a_ind, _b_ind) = P;
}


////////////////////////////////////////////////////////////////
// Implementation of the private methods
////////////////////////////////////////////////////////////////

void Sinkhorn::impl_vanilla() {
  CTRACK;

  // first init
  _u = vec(_M, fill::ones);
  _v = vec(_N, fill::ones);
  _K = exp( - _C / _reg);
  if (_K.is_symmetric()) {
    _K = symmatu(_K);
  }
  // std::cout << "K: \n" << _K << std::endl;
  if (_withgrad) {
    _Ju = mat(_M, _M, fill::zeros);
    _Jv = mat(_N, _M, fill::zeros);
  }
  //temp vec mats
  vec _Kv = vec(_M, fill::zeros);
  vec _KTu = vec(_N, fill::zeros);

  while ((_iter < _maxiter) & (_err >= _zerotol)) {
    _timer.tic();
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    _Kv = _K * _v;
    _update_u(_Kv);
    if (_withgrad) { _update_Ju(_Kv); }
    // std::cout << "Kv: " << size(_Kv) << std::endl;
    // std::cout << "u: " << size(_u) << std::endl;

    _KTu = _K.t() * _u;
    _update_v(_KTu);
    if (_withgrad) { _update_Jv(_KTu); }
    // std::cout << "KTu: \n" << size(_KTu) << std::endl;
    // std::cout << "v: \n" << size(_v) << std::endl;

    ++_iter;
    _norm_vanilla(_KTu);
    _timer.toc();
    // logging
    if ((_verbose != 0) && ((_iter-1) % _verbose) == 0) {
      cpp11::message(
        "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
        _iter, _err,
        _timer.speed_last(), _timer.speed_avg()
      );
    }
  }
}

void Sinkhorn::_update_u(vec& _Kv) {
  CTRACK;
  _u = _a / _Kv;
}

void Sinkhorn::_update_Ju(vec& _Kv) {
  CTRACK;
  _Ju = diagmat(1 / _Kv) - diagmat(_u / _Kv) * _K * _Jv;
  // _Ju = diagmat(1 / _Kv) - (_K.each_col() % (_u / _Kv)) * _Jv;
  // std::cout << _Ju << std::endl;
}

void Sinkhorn::_update_v(vec& _KTu) {
  CTRACK;
  _v = _b / _KTu;
}

void Sinkhorn::_update_Jv(vec& _KTu) {
  CTRACK;
  _Jv = - diagmat(_v / _KTu) * _K.t() * _Ju;
  // _Jv = - (_K.each_col() % (_v / _KTu)).t() * _Ju;
  // std::cout << _Jv << std::endl;
}

// void Sinkhorn::_update_Ju2(vec& _Kv, vec& _KTu) {
//   CTRACK;
//   if (_iter == 0) {
//     _Ju = diagmat(1 / _Kv);
//   } else {
//     _Ju = diagmat(1 / _Kv)
//     + (_K.each_col() % (_u / _Kv)) * (_K.each_col() % (_v / _KTu)).t() * _Ju;
//   }
// }

void Sinkhorn::_norm_vanilla(vec& _KTu) {
  CTRACK;
  _err = norm(_u % (_K*_v) - _a, 2) + norm(_v % _KTu - _b, 2);
}


void Sinkhorn::impl_log() {
  CTRACK;
  // first init
  _u = vec(_M, fill::zeros); // f
  _v = vec(_N, fill::zeros); // g
  if (_withgrad) {
    _Ju = mat(_M, _M, fill::zeros);
    _Jv = mat(_N, _M, fill::zeros);
  }
  mat W = mat(_M, _N, fill::zeros);
  mat X = mat(_N, _M, fill::zeros);
  vec Rminrow = vec(_M, fill::zeros);
  vec Rmincol = vec(_N, fill::zeros);
  const vec loga = log(_a);
  const vec logb = log(_b);
  const mat regdiaga = _reg * diagmat(1 / _a);

  // _K = _C - _u * _onesN.t() - _onesM * _v.t();

  while ((_iter < _maxiter) && (_err >= _zerotol)) {
    _timer.tic();
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    _K = _C - _u * _onesN.t() - _onesM * _v.t();
    _minrowjac(W, Rminrow);
    _u += _reg * loga + Rminrow;
    if (_withgrad) {
      _update_Jf(W, regdiaga);
    }

    _K = _C - _u * _onesN.t() - _onesM * _v.t();
    _mincoljac(X, Rmincol);
    _v += _reg * logb + Rmincol;
    if (_withgrad) {
      _update_Jg(X);
    }

    // terminate condition
    _K = _C - _u * _onesN.t() - _onesM * _v.t();
    _minrow(Rminrow);
    _mincol(Rmincol);
    _iter++;
    _err = norm(-Rminrow/_reg - loga, 2) + norm(-Rmincol/_reg - logb, 2);
    _timer.toc();
    // logging
    if ((_verbose != 0) && ((_iter-1) % _verbose) == 0) {
      cpp11::message(
        "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
        _iter, _err,
        _timer.speed_last(), _timer.speed_avg()
      );
    }
  }
}

void Sinkhorn::_minrow(vec& Rminrow) {CTRACK;
  // modify Rminrow in-place
  double Rmin;
  for (int i = 0; i < _M; ++i) { // i = 1, ..., M
    Rmin = _K.row(i).min();
    Rminrow(i) = Rmin - _reg * log(accu(exp(-(_K.row(i)-Rmin)/_reg)));
  }
}

void Sinkhorn::_mincol(vec& Rmincol) {CTRACK;
  // modify Rmincol in-place
  double Rmin;
  for (int j = 0; j < _N; ++j) { // j = 1, ..., N
    Rmin = _K.col(j).min();
    Rmincol(j) = Rmin - _reg * log(accu(exp(-(_K.col(j)-Rmin)/_reg)));
  }
}

void Sinkhorn::_minrowjac(mat& W, vec& Rminrow) {CTRACK;
  // modify W and Rminrow in-place
  double Rmin;
  rowvec sr = rowvec(_N, fill::zeros);
  for (int i = 0; i < _M; ++i) { // i = 1, ..., M
    Rmin = _K.row(i).min();
    sr = exp(-(_K.row(i)-Rmin)/_reg);
    Rminrow(i) = Rmin - _reg * log(accu(sr));
    W.row(i) = sr / accu(sr);
  }
}

void Sinkhorn::_mincoljac(mat& V, vec& Rmincol) {CTRACK;
  // modify Rmincol in-place
  double Rmin;
  vec sr = vec(_M, fill::zeros);
  for (int j = 0; j < _N; ++j) { // j = 1, ..., N
    Rmin = _K.col(j).min();
    sr = exp(-(_K.col(j)-Rmin)/_reg);
    Rmincol(j) = Rmin - _reg * log(accu(sr));
    V.row(j) = sr.t() / accu(sr);
  }
}

void Sinkhorn::_update_Jf(mat& W, const mat& regdiaga) {
  CTRACK;
  _Ju = regdiaga - W * _Jv;
}

void Sinkhorn::_update_Jg(mat& X) {
  CTRACK;
  _Jv = - X * _Ju;
}
