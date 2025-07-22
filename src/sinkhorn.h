
// implement the Sinkhorn algorithm and its gradients with class
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#ifndef WIG_SINKHORN_H
#define WIG_SINKHORN_H

#include "timer.h"                // TicToc timer class

#include <cpp11armadillo.hpp>
using namespace arma;


// implement the Sinkhorn algos in this class

class Sinkhorn {

private:

  bool _withgrad;
  int _maxiter;
  double _zerotol;
  bool _verbose;

  // variables for the reduced form
  int _M, _N; // dimensions
  // data
  vec _a,_b; // a and b whose elements are not 0
  mat _C; // cost matrix corresponding to a and b whose elements are not 0
  double _reg;

  // additional vars for log solution
  vec _onesM, _onesN;

  int _iter = 0;
  double _err = 1000.;

  TicToc _timer; // timer class

  void _minrow(vec& Rminrow) {
    // modify Rminrow in-place
    double Rmin;
    for (int i = 0; i < _M; ++i) { // i = 1, ..., M
      Rmin = _K.row(i).min();
      Rminrow(i) = Rmin - _reg * log(accu(exp(-(_K.row(i)-Rmin)/_reg)));
    }
  }

  void _mincol(vec& Rmincol) {
    // modify Rmincol in-place
    double Rmin;
    for (int j = 0; j < _N; ++j) { // j = 1, ..., N
      Rmin = _K.col(j).min();
      Rmincol(j) = Rmin - _reg * log(accu(exp(-(_K.col(j)-Rmin)/_reg)));
    }
  }

  void _minrowjac(mat& W, vec& Rminrow) {
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

  void _mincoljac(mat& V, vec& Rmincol) {
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

  // implementation of the vanilla sinkhorn algorithm
  void compute_vanilla_impl() {

    // first init
    _u = vec(_M, fill::ones);
    _v = vec(_N, fill::ones);
    _K = exp( - _C / _reg);
    if (_withgrad) {
      _Ju = mat(_M, _M, fill::zeros);
      _Jv = mat(_N, _M, fill::zeros);
    }
    //temp vec mats
    vec _Kv = vec(_N, fill::zeros);
    vec _KTu = vec(_M, fill::zeros);

    while ((_iter < _maxiter) & (_err >= _zerotol)) {
      _timer.tic();
      // check user interrupt if computation is stuck
      cpp11::check_user_interrupt();

      _Kv = _K * _v;
      _u = _a / _Kv;
      if (_withgrad) {
        _Ju = diagmat(1 / _Kv) - diagmat(_a / pow(_Kv, 2)) * _K * _Jv;
        // std::cout << _Ju << std::endl;
      }

      _KTu = _K.t() * _u;
      _v = _b / _KTu;
      if (_withgrad) {
        _Jv = - diagmat(_b / pow(_KTu, 2)) * _K.t() * _Ju;
        // std::cout << _Jv << std::endl;
      }

      ++_iter;
      _err = norm(_u % (_K*_v) - _a, 2) + norm(_v % _KTu - _b, 2);
      _timer.toc();
      // logging
      if (((_iter-1) % _verbose) == 0) {
        cpp11::message(
          "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
          _iter, _err,
          _timer.speed_last(), _timer.speed_avg()
        );
      }
    }
  }

  // implementation of the log sinkhorn algorithm
  void compute_log_impl() {
    // first init
    _u = vec(_M, fill::zeros); // f
    _v = vec(_N, fill::zeros); // g
    if (_withgrad) {
      _Ju = mat(_M, _M, fill::zeros);
      _Jv = mat(_N, _M, fill::zeros);
    }
    mat W = mat(_M, _N, fill::zeros);
    mat V = mat(_N, _M, fill::zeros);
    vec Rminrow = vec(_M, fill::zeros);
    vec Rmincol = vec(_N, fill::zeros);
    vec loga = log(_a);
    vec logb = log(_b);

    _K = _C - _u * _onesN.t() - _onesM * _v.t();

    while ((_iter < _maxiter) && (_err >= _zerotol)) {
      _timer.tic();
      // check user interrupt if computation is stuck
      cpp11::check_user_interrupt();

      _K = _C - _u * _onesN.t() - _onesM * _v.t();
      _minrowjac(W, Rminrow);
      _u += _reg * loga + Rminrow;
      if (_withgrad) {
        _Ju = _reg * diagmat(1 / _a) - W * _Jv;
      }

      _K = _C - _u * _onesN.t() - _onesM * _v.t();
      _mincoljac(V, Rmincol);
      _v += _reg * logb + Rmincol;
      if (_withgrad) {
        _Jv = - V * _Ju;
      }

      // terminate condition
      _K = _C - _u * _onesN.t() - _onesM * _v.t();
      _minrow(Rminrow);
      _mincol(Rmincol);
      _iter++;
      _err = norm(-Rminrow/_reg - loga, 2) + norm(-Rmincol/_reg - logb, 2);
      _timer.toc();
      // logging
      if (((_iter-1) % _verbose) == 0) {
        cpp11::message(
          "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
          _iter, _err,
          _timer.speed_last(), _timer.speed_avg()
        );
      }
    }
  }

  void reset_counter() {
    _iter = 0;
    _err = 1000.;
  }

public:

  // intermediate variables
  vec _u, _v; // u,v in vanilla, f,g in log
  mat _Ju, _Jv; // Ju,Jv in vanilla, Jf,Jg in log
  mat _K; // K in vanilla, R in log
  vec _grad_a;
  mat _P;

  // init sinkhorn method
  Sinkhorn(bool withgrad = false, // bool reduce = false,
           int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    _withgrad = withgrad;
    // _reduce = reduce;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }

  // TODO: automatically determine the mode? maybe in the R side

  ////////////////////////////////////////////////////////////////
  // Algo 4.1/3.1: Vanilla Sinkhorn with/without Gradient wrt a
  ////////////////////////////////////////////////////////////////

  void compute_vanilla(vec& a, vec& b, mat& C, double reg) {
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

    // start the computation
    compute_vanilla_impl();

    // optimal coupling for reduce form
    mat P = diagmat(_u) * _K * diagmat(_v);

    // compute the grad
    if (_withgrad) {
      _grad_a = _reg * (
        _v.t() * (diagmat(log(_u)) * _K + _K * diagmat(log(_v))).t() * _Ju +
          _u.t() * (diagmat(log(_u)) * _K + _K * diagmat(log(_v))) * _Jv
      ).t();
    }

    // recover the original solution
    // recover the u,v
    vec u = vec(M, fill::zeros);
    vec v = vec(N, fill::zeros);
    u.elem( _a_ind ) = _u;
    v.elem( _b_ind ) = _v;
    _u = u;
    _v = v;
    _K = exp(-C/reg);

    if (_withgrad) {
      vec grad_a = vec(M, fill::zeros);
      grad_a(_a_ind) = _grad_a;
      _grad_a = grad_a;
    }
    // _P = diagmat(_u) * _K * diagmat(_v);
    _P = mat(M, N, fill::zeros);
    _P.submat(_a_ind, _b_ind) = P;

  }


  ////////////////////////////////////////////////////////////////
  // Algo 4.3/3.3: Log Sinkhorn with/without Gradient wrt a
  ////////////////////////////////////////////////////////////////

  void compute_log(vec& a, vec& b, mat& C, double reg) {
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
    compute_log_impl();

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
    _u = u;
    _v = v;

    if (_withgrad) {
      vec grad_a = vec(M, fill::zeros);
      grad_a(_a_ind) = _grad_a;
      _grad_a = grad_a;
    }

    // recover the optimal coupling
    _P = mat(M, N, fill::zeros);
    _P.submat(_a_ind, _b_ind) = P;
  }
};


#endif // WIG_SINKHORN_H
