
// implement the Sinkhorn algorithm and its gradients with class
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#ifndef RWIG_SINKHORN_H
#define RWIG_SINKHORN_H

#include <vector>

#include "common.hpp"
#include "thread_pool.hpp"
#include "timer.hpp"

class Sinkhorn {

private:
  // class init control parameters
  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _C_is_symm;

  la::idx _M, _N;
  // data (reduced problem: rows/cols with a != 0 / b != 0)
  la::Vec _a, _b;
  la::Mat _C, _P;
  double _reg;
  la::Mat _K; // Gibbs kernel exp(-C/reg) (vanilla only)
  la::Vec _u, _v;
  la::Vec _grad_a;
  // temp vars
  la::Vec _Kv, _KTu;

  // history of u and v in the forward pass for vanilla (f,g for log)
  std::vector<la::Vec> _uhist, _vhist;

  // for log algo
  la::Vec _loga, _logb, _Rminrow, _Rmincol;
  la::Vec _rowmin, _rowsum, _colmin, _colsum; // scratch for the lse kernels
  la::Mat _E;                                 // M x N buffer for the backward pass

  // timer for logging purpose
  TicToc _timer;

  // forward and backward loop for the vanilla Sinkhorn
  void _fwd_vanilla();
  void _bwd_vanilla();

  // forward and backward loop for the log Sinkhorn
  void _fwd_log(ThreadPool &pool);
  void _bwd_log(ThreadPool &pool);

  // soft-min helpers for the log algo (threaded through the pool)
  void _minrow(ThreadPool &pool, const double *f, const double *g);
  void _mincol(ThreadPool &pool, const double *f, const double *g);

  // K * x (or K^T * x) using the symmetric kernel when possible
  void _Kmul(bool trans, const la::Vec &x, la::Vec &y) const {
    if (_C_is_symm) {
      la::symv(_K, x.data(), y.data());
    } else {
      la::gemv(trans, _K, x.data(), y.data());
    }
  }

  void _reset_counter() {
    this->iter = 0;
    this->err = 1000;
  }

  // reduce the problem to the support of a and b; fills _a, _b, _C, _M, _N
  void _reduce(const la::Vec &a, const la::Vec &b, const la::Mat &C,
               std::vector<la::idx> &a_ind, std::vector<la::idx> &b_ind);
  void _set_return_code();
  void _log_iter(const char *stage, int it);

public:
  // intermediate variables
  la::Vec u, v; // u,v in vanilla, f,g in log
  la::Vec grad_a;
  la::Mat P;
  double loss;
  int return_code; // 0: convergence, 1: max iter reached, 2: else

  int iter;
  double err;

  // init sinkhorn method
  Sinkhorn(bool withgrad = false, int maxiter = 1000, double zerotol = 1e-6,
           int verbose = 0) {
    _withgrad = withgrad;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }
  ~Sinkhorn() {}

  // compute vanilla Sinkhorn
  void compute_vanilla(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                       double reg);

  // compute log Sinkhorn
  void compute_log(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                   double reg, const int &n_threads);
};

#endif // RWIG_SINKHORN_H
