
// implement the Wasserstein Barycenter algorithm and its gradients
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 5 & 6

#ifndef RWIG_BARYCENTER_H
#define RWIG_BARYCENTER_H

#include <vector>

#include "common.hpp"
#include "thread_pool.hpp"
#include "timer.hpp" // TicToc timer class

class Barycenter {

private:
  // class init control parameters
  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _C_is_symm;

  // data
  la::Mat _A;     // basis/dictionary/topics (M x S)
  la::Mat _C;     // cost matrix (M x N)
  la::Vec _w;     // weight vector (S)
  la::Vec _b_ext; // true (data) vector b only for loss and gradient computation
  la::Vec _logb;  // logb only used for log algo

  double _reg; // regularization epsilon

  // dimensions
  la::idx _M, _N, _S;
  la::Mat _K;   // M x N, Gibbs kernel (parallel only)
  la::Mat _KV;  // M x S, KV in parallel, Rminrow in log
  la::Mat _KTU; // N x S, KTU in parallel, Rmincol in log

  // history of U and V (F/G in log) for the backward pass
  std::vector<la::Mat> _Uhist, _Vhist;    // keep track of history for U and V
  std::vector<la::Vec> _bhist, _logbhist; // history of barycenter

  // scratch for the log algo
  la::Mat _logA;
  la::Vec _rowmin, _rowsum, _colmin, _colsum;
  la::Mat _E; // M x N buffer for the backward pass

  // timer for logging purpose
  TicToc _timer;

  // forward and backward loop for the parallel barycenter
  void _fwd_parallel();
  void _bwd_parallel();

  // forward and backward loop for the log barycenter
  void _fwd_log(ThreadPool &pool);
  void _bwd_log(ThreadPool &pool);

  // soft-mins for the log algo: for every topic s, of R(F[:,s], G[:,s])
  void _minrow(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
               la::Mat &out);
  void _mincol(ThreadPool &pool, const la::Mat &F, const la::Mat &G,
               la::Mat &out);

  // K * X (or K^T * X) using the symmetric kernel when possible
  void _Kmul(bool trans, const la::Mat &X, la::Mat &Y) const {
    if (_C_is_symm) {
      la::symm(_K, X, Y);
    } else {
      la::gemm(trans, _K, X, Y);
    }
  }
  void _Kmul(bool trans, const la::Vec &x, la::Vec &y) const {
    if (_C_is_symm) {
      la::symv(_K, x.data(), y.data());
    } else {
      la::gemv(trans, _K, x.data(), y.data());
    }
  }

  // reset the counter and err (for the rerunning of the same class)
  void _reset_counter() {
    this->iter = 0;
    this->err = 1000.;
  }
  void _set_return_code() {
    if (this->err <= _zerotol) {
      this->return_code = 0;
    } else if (this->iter == _maxiter) {
      this->return_code = 1;
    } else {
      this->return_code = 2;
    }
  }
  void _log_iter(const char *stage, int it);

public:
  // scaling vars
  la::Mat U, V; // F, G in log

  // output barycenter b
  la::Vec b;
  // output gradient of b (with data b_ext)
  la::Mat grad_A; // gradient wrt A
  la::Vec grad_w; // gradient wrt w

  double loss;
  int iter;
  double err;
  int return_code;

  // init Barycenter
  Barycenter(int S, bool withgrad = false, int maxiter = 1000,
             double zerotol = 1e-6, int verbose = 0) {
    _S = S;
    _withgrad = withgrad;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }
  ~Barycenter() {}

  // setters to update private attributes
  void update_C(const la::Mat &C) { // once per model
    _C = C;
    // also update the dimensions
    _M = _C.nrow();
    _N = _C.ncol();
    _C_is_symm = _C.is_symmetric();
  }
  void update_reg(const double reg) { _reg = reg; } // once per model
  void update_A(const la::Mat &A) { _A = A; }       // once per batch
  void update_w(const la::Vec &w) { _w = w; }       // once per doc
  // update the "true data" b vector (only used for loss and grad)
  void update_b_ext(const la::Vec &b) { _b_ext = b; }           // once per doc
  void update_withgrad(bool withgrad) { _withgrad = withgrad; } // turn off grad

  // compute parallel barycenter
  void compute_parallel();

  // compute log-stabilized barycenter
  void compute_log(const int &n_threads);
};

#endif // RWIG_BARYCENTER_H
