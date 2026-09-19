
// implement the 3 optimizers: vanilla SGD, Adam, and AdamW
// https://arxiv.org/abs/2504.08722
// Section 7.1

#ifndef RWIG_OPTIMIZER_H
#define RWIG_OPTIMIZER_H

#include <cmath>

#include "common.hpp"

class Optimizer {

private:
  // all the optimizer parameters
  double _eta, _gamma, _beta1, _beta2, _eps;

  // first / second moment estimates (same dimension as theta)
  la::Mat _mtheta, _vtheta;

  // counter for step;
  int _step = 0;

  // Adam update with decoupled weight decay `decay` (0 gives plain Adam)
  void _adam_step(la::Mat &theta, const la::Mat &g_theta, double decay) {
    _step++;
    const double bc1 = 1 - std::pow(_beta1, _step);
    const double bc2 = 1 - std::pow(_beta2, _step);
    for (la::idx k = 0; k < theta.size(); ++k) {
      const double g = g_theta[k];
      _mtheta[k] = _beta1 * _mtheta[k] + (1 - _beta1) * g;
      _vtheta[k] = _beta2 * _vtheta[k] + (1 - _beta2) * g * g;
      const double mhat = _mtheta[k] / bc1;
      const double vhat = _vtheta[k] / bc2;
      theta[k] = (1 - _eta * decay) * theta[k] -
                 _eta * (mhat / (std::sqrt(vhat) + _eps));
    }
  }

public:
  // init Optimizer for an M x N parameter matrix
  void init(const la::idx M, const la::idx N, const double eta,
            const double gamma, const double beta1, const double beta2,
            const double eps) {
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;
    _mtheta.resize(M, N);
    _vtheta.resize(M, N);
    _step = 0;
  }

  // one update: mode 0 = SGD, 1 = Adam, 2 = AdamW
  void step(int mode, la::Mat &theta, const la::Mat &g_theta) {
    if (mode == 0) {
      _step++;
      for (la::idx k = 0; k < theta.size(); ++k) theta[k] -= _eta * g_theta[k];
    } else {
      _adam_step(theta, g_theta, mode == 2 ? _gamma : 0.0);
    }
  }
};

#endif // RWIG_OPTIMIZER_H
