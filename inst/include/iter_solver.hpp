// State and verbose logging shared by the iterative solvers
// (Sinkhorn and Barycenter): iteration counter, convergence test,
// return code, and the per-iteration timing messages.

#ifndef RWIG_ITER_SOLVER_H
#define RWIG_ITER_SOLVER_H

#include "r_glue.hpp"
#include "timer.hpp"
#include "vformat.hpp"

class IterSolver {
public:
  int iter = 0;
  double err = 1000.;
  int return_code = 2; // 0: convergence, 1: max iter reached, 2: else

protected:
  int _maxiter;
  double _zerotol;
  int _verbose; // 0: silent, k > 0: report every k-th iteration
  TicToc _timer;

  IterSolver(int maxiter, double zerotol, int verbose)
      : _maxiter(maxiter), _zerotol(zerotol), _verbose(verbose) {}

  // reset the counter and err (for rerunning the same object)
  void _reset_counter() {
    iter = 0;
    err = 1000.;
  }

  bool _keep_going() const { return iter < _maxiter && err >= _zerotol; }

  void _set_return_code() {
    if (err <= _zerotol) {
      return_code = 0;
    } else if (iter == _maxiter) {
      return_code = 1;
    } else {
      return_code = 2;
    }
  }

  // logging helpers: no-ops unless verbose
  void _log_stage(const char *msg) {
    if (_verbose != 0) rr::message(msg);
  }
  void _tic() {
    if (_verbose != 0) _timer.tic();
  }
  // forward pass: report iteration and error every `_verbose` iterations
  void _toc_fwd() {
    if (_verbose == 0) return;
    _timer.toc();
    if ((iter - 1) % _verbose == 0) {
      rr::message(vformat("iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
                          iter, err, _timer.speed_last(), _timer.speed_avg()));
    }
  }
  // backward pass: report the (reverse) step l
  void _toc_bwd(int l) {
    if (_verbose == 0) return;
    _timer.toc();
    if ((iter - 1) % _verbose == 0) {
      rr::message(vformat("iter: %d, last speed: %.3f, avg speed: %.3f", l,
                          _timer.speed_last(), _timer.speed_avg()));
    }
  }
};

#endif // RWIG_ITER_SOLVER_H
