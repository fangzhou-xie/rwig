
// implement the Sinkhorn algorithm and its gradients with class
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#ifndef WIG_SINKHORN_H
#define WIG_SINKHORN_H

#include <vector> // std::vector

#include "common.hpp"

using namespace arma;

#include "timer.hpp"

class Sinkhorn {

private:

  // class init control parameters
  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _C_is_symm;
  int _n_threads;

  uword _M, _N;
  // data
  vec _a, _b;
  mat _C, _P;
  double _reg;
  mat _K; // K in vanilla, R in log
  vec _u, _v;
  vec _grad_a;

  // history of u and v in the forward pass for vanilla (f,g for log)
  std::vector<vec> _uhist, _vhist;
  // std::vector<mat> _Whist, _Xhist;  // history of W and X matrices
  // mat _uhist, _vhist;
  // cube _Whist, _Xhist; // history of W and X matrices
  // mat _W, _X;             // W and X matrices for the backward adjoints

  // for log algo
  vec _onesM, _onesN, _Rmincol, _Rminrow;

  // timer for logging purpose
  TicToc _timer;
  // C-string for message purpose
  char* _msg;

  // forward and backward loop for the vanilla Sinkhorn
  void _fwd_vanilla();
  void _bwd_vanilla();

  // forward and backward loop for the log Sinkhorn
  void _fwd_log(const int& n_threads);
  void _bwd_log(const int& n_threads);
  // void _compute_log_serial(const vec& a, const vec& b, const mat& C, double reg);

  // serial version for the mincol and minrow
  void _minrow_serial();
  void _mincol_serial();
  void _minrowcol_serial(); // only before the err for termination

  // thread version for the mincol and minrow
  void _minrow_thread(const int& n_threads);
  void _mincol_thread(const int& n_threads);
  void _minrowcol_thread(const int& n_threads); // only before the err for termination

  // TODO: wrap the fbar and gbar into functions and threading
  void _update_fbar_serial(vec& fbar, vec& gbar, mat& PbarP, int& l);
  void _update_gbar_serial(vec& fbar, vec& gbar, mat& PbarP, int& l);
  void _update_fbar_thread(vec& fbar, vec& gbar, mat& PbarP, int& l, const int& n_threads);
  void _update_gbar_thread(vec& fbar, vec& gbar, mat& PbarP, int& l, const int& n_threads);

  // test faster R compute
  void _compute_R();

  void _reset_counter() {
    this->iter = 0;
    this->err = 1000;
  }

public:

  // intermediate variables
  vec u, v; // u,v in vanilla, f,g in log
  vec grad_a;
  mat P;
  double loss;
  int return_code; // 0: convergence, 1: max iter reached, 2: else

  int iter;
  double err;

  // init sinkhorn method
  Sinkhorn(bool withgrad = false,
           int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    // _n_threads = n_threads;
    _withgrad = withgrad;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }
  ~Sinkhorn() {}

  // compute vanilla Sinkhorn
  void compute_vanilla(const vec& a, const vec& b, const mat& C, double reg);

  // compute log Sinkhorn
  void compute_log(const vec& a, const vec& b, const mat& C, double reg,
                   const int& n_threads);

};

#endif // WIG_SINKHORN_H
