
// implement the Wasserstein Barycenter algorithm and its gradients
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 5 & 6

#ifndef RWIG_BARYCENTER_H
#define RWIG_BARYCENTER_H

#include <vector>

#include "common.hpp"
#include "iter_solver.hpp"
#include "logdomain.hpp"
#include "thread_pool.hpp"

class Barycenter : public IterSolver {

private:
  bool _withgrad;

  // data
  la::Mat _A;     // basis/dictionary/topics (M x S)
  la::Mat _C;     // cost matrix (M x N)
  la::Vec _w;     // weight vector (S)
  la::Vec _b_ext; // true (data) vector b only for loss and gradient computation
  la::Vec _logb;  // logb only used for log algo

  double _reg; // regularization epsilon

  // dimensions
  la::idx _M, _N, _S;
  la::KernelOp _K; // Gibbs kernel (parallel only)
  la::Mat _KV;     // M x S, KV in parallel, Rminrow in log
  la::Mat _KTU;    // N x S, KTU in parallel, Rmincol in log

  // history of U and V (F/G in log) for the backward pass
  std::vector<la::Mat> _Uhist, _Vhist;    // keep track of history for U and V
  std::vector<la::Vec> _bhist, _logbhist; // history of barycenter
  // parallel algo only: K V^l and K^T U^l from the forward pass, so the
  // backward pass does not recompute them (3 GEMMs per step)
  std::vector<la::Mat> _KVhist, _KTUhist;

  // scratch for the log algo
  la::Mat _logA;
  logdom::Scratch _scratch;
  logdom::Problem _prob() const {
    return logdom::Problem{_C.data(), (int)_M, (int)_N, _reg};
  }

  // forward and backward loop for the parallel barycenter
  void _fwd_parallel();
  void _bwd_parallel();

  // forward and backward loop for the log barycenter
  void _fwd_log(ThreadPool &pool);
  void _bwd_log(ThreadPool &pool);

  // soft-mins for the log algo: for every topic s, of R(F[:,s], G[:,s])
  void _minrow(ThreadPool &pool, const la::Mat &F, const la::Mat &G, la::Mat &out);
  void _mincol(ThreadPool &pool, const la::Mat &F, const la::Mat &G, la::Mat &out);

  // b <- b / sum(b), then the quadratic loss against b_ext if requested
  void _normalize_b_and_loss();

public:
  // scaling vars
  la::Mat U, V; // F, G in log

  // output barycenter b
  la::Vec b;
  // output gradient of b (with data b_ext)
  la::Mat grad_A; // gradient wrt A
  la::Vec grad_w; // gradient wrt w

  double loss;

  // init Barycenter
  Barycenter(int S, bool withgrad = false, int maxiter = 1000,
             double zerotol = 1e-6, int verbose = 0)
      : IterSolver(maxiter, zerotol, verbose), _withgrad(withgrad), _S(S) {}

  // setters to update private attributes
  void update_C(const la::Mat &C) { // once per model
    _C = C;
    _M = _C.nrow();
    _N = _C.ncol();
  }
  void update_reg(const double reg) { _reg = reg; } // once per model
  void update_A(const la::Mat &A) { _A = A; }       // once per batch
  void update_w(const la::Vec &w) { _w = w; }       // once per doc
  // update the "true data" b vector (only used for loss and grad)
  void update_b_ext(const la::Vec &b) { _b_ext = b; } // once per doc

  // compute parallel barycenter
  void compute_parallel();

  // compute log-stabilized barycenter
  void compute_log(const int &n_threads);
};

#endif // RWIG_BARYCENTER_H
