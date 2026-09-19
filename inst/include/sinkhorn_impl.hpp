
// implement the Sinkhorn algorithm and its gradients with class
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#ifndef RWIG_SINKHORN_H
#define RWIG_SINKHORN_H

#include <vector>

#include "common.hpp"
#include "iter_solver.hpp"
#include "logdomain.hpp"
#include "thread_pool.hpp"

class Sinkhorn : public IterSolver {

private:
  bool _withgrad;

  la::idx _M, _N;
  // data (reduced problem: rows/cols with a != 0 / b != 0)
  la::Vec _a, _b;
  la::Mat _C, _P;
  double _reg;
  la::KernelOp _K; // Gibbs kernel (vanilla only)
  la::Vec _u, _v;  // u, v in vanilla; f, g in log
  la::Vec _grad_a;
  // temp vars
  la::Vec _Kv, _KTu;

  // history of u and v in the forward pass for vanilla (f,g for log)
  std::vector<la::Vec> _uhist, _vhist;

  // for log algo
  la::Vec _loga, _logb, _Rminrow, _Rmincol;
  logdom::Scratch _scratch;
  logdom::Problem _prob() const {
    return logdom::Problem{_C.data(), (int)_M, (int)_N, _reg};
  }

  // forward and backward loop for the vanilla Sinkhorn
  void _fwd_vanilla();
  void _bwd_vanilla();

  // forward and backward loop for the log Sinkhorn
  void _fwd_log(ThreadPool &pool);
  void _bwd_log(ThreadPool &pool);

  // reduce the problem to the support of a and b (fills _a, _b, _C, _M, _N)
  // and scatter the reduced solution back into the full-size outputs
  void _reduce(const la::Vec &a, const la::Vec &b, const la::Mat &C,
               std::vector<la::idx> &a_ind, std::vector<la::idx> &b_ind);
  void _expand(la::idx M, la::idx N, const std::vector<la::idx> &a_ind,
               const std::vector<la::idx> &b_ind, double u_fill);
  // regularized transport loss of the reduced solution _P
  void _compute_loss();

public:
  // intermediate variables
  la::Vec u, v; // u,v in vanilla, f,g in log
  la::Vec grad_a;
  la::Mat P;
  double loss;

  // init sinkhorn method
  Sinkhorn(bool withgrad = false, int maxiter = 1000, double zerotol = 1e-6,
           int verbose = 0)
      : IterSolver(maxiter, zerotol, verbose), _withgrad(withgrad) {}

  // compute vanilla Sinkhorn
  void compute_vanilla(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                       double reg);

  // compute log Sinkhorn
  void compute_log(const la::Vec &a, const la::Vec &b, const la::Mat &C,
                   double reg, const int &n_threads);
};

#endif // RWIG_SINKHORN_H
