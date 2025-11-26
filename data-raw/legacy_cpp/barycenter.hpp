
// implement the Wasserstein Barycenter algorithm and its gradients
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 5 & 6

#ifndef WIG_BARYCENTER_H
#define WIG_BARYCENTER_H

#include "timer.hpp"                // TicToc timer class

// https://github.com/Compaile/ctrack
// #include "ctrack.hpp"             // profiler
// #include <easy/profiler.h>


#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace arma;


class Barycenter {

private:

  // class init control parameters
  bool _withjac;
  int _maxiter;
  double _zerotol;
  int _verbose;

  // data
  mat _A, _C;
  vec _w;
  double _reg;
  // dimentions
  uword _M, _N, _S;
  // aux vars
  mat _JUA, _JVA, _JUw, _JVw;     // JFA,JGA,JFw,JGw in log
  mat _K, _KT;                    // R, R.t() in log
  mat _KV, _KTU;                  // Rminrow, Rmincol in log

  // additional aux vars
  vec _onesM, _onesN, _onesS;
  // aux vars for the jacobians: hopefully speed up
  // mat _diag1kv, _diag1ktu, _diagukv, _diagvktu; // loop by t

  // log-only vars
  cube _Ws, _Xs;                  // weights for the Jacobians
  mat _JlogbA, _Jlogbw;
  mat _P;                         // exp(-R/reg);

  // timer for logging purpose
  TicToc _timer;

  void update_parallel_U() {
    // CTRACK;
    _KV = _K * _V;
    _U = _A / _KV;
    if (_withjac) update_parallel_JU();
  }
  void update_parallel_V() {
    // CTRACK;
    _V = (b * _onesS.t()) / _KTU;
    if (_withjac) update_parallel_JV();
  }
  void update_parallel_b(mat& onesNwT) {
    // CTRACK;
    _KTU = _KT * _U;
    b = prod(pow(_KTU, onesNwT), 1);
    if (_withjac) update_parallel_Jb();
  }

  // void update_parallel_JU2();
  // void update_parallel_JU3();

  // utils for computing parallel Jacobians
  void update_parallel_JU();
  void update_parallel_JV();
  void update_parallel_Jb() ;

  // utils for computing the Rminrow and Rmincol
  void update_log_Rminrow();
  void update_log_Rmincol();

  // utils for computing log Jacobians
  void update_log_JF();
  void update_log_JG();
  void update_log_Jlogb();

  // implementation of the parallel Barycenter
  void compute_parallel_impl();

  // implementation of the log Barycenter
  void compute_log_impl();

  void reset_counter() {
    _iter = 0;
    _err = 1000.;
  }

  // init all vars for the parallel method
  void init_parallel();

  // init the vars for the log method
  void init_log();

public:

  // scaling vars
  mat _U, _V;                     // F,G in log

  // output centroid b
  vec b;
  // output Jacobians of b
  mat _JbA, _Jbw;

  // termination
  int _iter = 0;
  double _err = 1000.;

  // using the timer
  // TicToc timer_main;
  // TicToc timer_grad;

  // init the class
  Barycenter(int M, int N, int S, bool withjac = false,
             int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    // all the dimension
    _M = M;
    _N = N;
    _S = S;
    // compute jacobian?
    _withjac = withjac;
    // other control parameters
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
    // aux vars
    _onesM = vec(_M, fill::ones);
    _onesN = vec(_N, fill::ones);
    _onesS = vec(_S, fill::ones);
  }
  ~Barycenter() {}

  // setters to update private attributes
  void update_C(mat& C) {                                   // once per model
    _C = C;
    // check the dimensions
    if (_M != _C.n_rows) cpp11::stop("M not match");
    if (_N != _C.n_cols) cpp11::stop("N not match");
  }
  void update_reg(double reg) {_reg = reg;}                 // once per model
  void update_A(mat& A) {_A = A;}                           // once per batch
  void update_w(vec& w) {_w = w;}                           // once per doc
  void update_withjac(bool withjac) {_withjac = withjac;}   // turn off jacobian

  // actual compute methods
  void compute_parallel();
  void compute_log();
};




#endif // WIG_BARYCENTER_H
