
// implement the Wasserstein Barycenter algorithm and its gradients
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 5 & 6

#ifndef RWIG_BARYCENTER_H
#define RWIG_BARYCENTER_H

#include "common.hpp"

#include "timer.hpp"           // TicToc timer class

using namespace arma;

class Barycenter {

private:

  // class init control parameters
  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _C_is_symm;

  // data
  mat _A;     // basis/dictionary/topics
  mat _C;     // cost matrix
  vec _w;     // weight vector
  vec _b_ext; // true (data) vector b only for loss and gradient computation
  vec _logb;  // logb only used for log algo

  double _reg; // regularization epsilon

  // dimentions
  uword _M, _N, _S;
  mat _K;   // M * N, K in parallel, R in log
  mat _KV;  // M * S, KV in parallel, Rminrow in log
  mat _KTU; // N * S, KTU in parallel, Rmincol in log

  // history of U and V (F/G in log) for the backward pass
  // cube _Uhist, _Vhist;
  std::vector<mat> _Uhist, _Vhist;    // keep track of history for U and V
  std::vector<vec> _bhist, _logbhist; // history of barycenter

  // additional aux vars
  vec _onesM, _onesN, _onesS;

  // timer for logging purpose
  TicToc _timer;
  // C-string for message purpose
  char* _msg;

  // forward and backward loop for the parallel barycenter
  void _fwd_parallel();
  void _bwd_parallel();

  // forward and backward loop for the parallel barycenter
  void _fwd_log(const int& n_threads);
  void _bwd_log(const int& n_threads);

  // other util functions for the log algo
  // serial computing minrow and mincol
  void _minrow_serial();
  void _mincol_serial();
  // parallel/threading computation minrow and mincol
  void _minrow_thread(const int& n_threads);
  void _mincol_thread(const int& n_threads);

  // test performance
  // vec _comp_b1(mat& onesNwT);
  // vec _comp_b2();

  // set up the ones vecs, should only happen after the `update_C`
  void _init() {
    _onesM = vec(_M, fill::ones);
    _onesN = vec(_N, fill::ones);
    _onesS = vec(_S, fill::ones);
  }
  // reset the counter and err (for the rerunning of the same class)
  void _reset_counter() {
    this->iter = 0;
    this->err = 1000.;
  }

public:

  // scaling vars
  mat U, V;   // F, G in log

  // output barycenter b
  vec b;
  // output gradient of b (with data b_ext)
  mat grad_A;  // gradient wrt A
  vec grad_w;  // gradient wrt w

  double loss;
  int iter;
  double err;
  int return_code;

  // init Barycenter
  Barycenter(int S, bool withgrad = false,
             int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    _S = S;
    _withgrad = withgrad;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }
  ~Barycenter() {}

  // setters to update private attributes
  void update_C(const mat& C) {                               // once per model
    _C = C;
    // also update the dimensions
    _M = _C.n_rows;
    _N = _C.n_cols;
  }
  void update_reg(const double reg) { _reg = reg; }          // once per model
  void update_A(const mat& A) { _A = A; }                     // once per batch
  void update_w(const vec& w) { _w = w; }                     // once per doc
  // update the "true data" b vector (only used for loss and grad)
  void update_b_ext(const vec& b) { _b_ext = b; }             // once per doc
  void update_withgrad(bool withgrad) {_withgrad = withgrad;} // turn off grad

  // compute parallel barycenter
  void compute_parallel();

  // compute log-stabilized barycenter
  void compute_log(const int& n_threads);

};


#endif // RWIG_BARYCENTER_H
