
// implement the 3 optimizers: vanilla SGD, Adam, and AdamW
// https://arxiv.org/abs/2504.08722
// Section 7.1

#ifndef RWIG_OPTIMIZER_H
#define RWIG_OPTIMIZER_H

#include <cmath>

#include "common.hpp"

// implement the Optimizer class

class Optimizer {

private:
  // dimensions
  la::idx _M, _N;

  // all the optimizer parameters
  double _eta, _gamma, _beta1, _beta2, _eps;

  // the aux parameters (same dimension as theta)
  la::Mat _mtheta, _vtheta;

  // counter for step;
  int _step;

public:
  Optimizer() {}
  ~Optimizer() {}

  // init Optimizer
  void init(const la::idx M, const la::idx N, const double eta,
            const double gamma, const double beta1, const double beta2,
            const double eps) {
    // load the parameters
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;

    _M = M; // number of rows
    _N = N; // number of cols

    // setup the aux vars
    _mtheta.resize(_M, _N);
    _vtheta.resize(_M, _N);

    // counter
    _step = 0;
  }

  // SGD update
  void sgd(la::Mat &theta, const la::Mat &g_theta) {
    _step++;
    for (la::idx k = 0; k < theta.size(); ++k) theta[k] -= _eta * g_theta[k];
  }

  // Adam update
  void adam(la::Mat &theta, const la::Mat &g_theta) {
    _step++;
    const double bc1 = 1 - std::pow(_beta1, _step);
    const double bc2 = 1 - std::pow(_beta2, _step);
    for (la::idx k = 0; k < theta.size(); ++k) {
      const double g = g_theta[k];
      _mtheta[k] = _beta1 * _mtheta[k] + (1 - _beta1) * g;
      _vtheta[k] = _beta2 * _vtheta[k] + (1 - _beta2) * g * g;
      const double mhat = _mtheta[k] / bc1;
      const double vhat = _vtheta[k] / bc2;
      theta[k] = theta[k] - _eta * (mhat / (std::sqrt(vhat) + _eps));
    }
  }

  // AdamW update
  void adamw(la::Mat &theta, const la::Mat &g_theta) {
    _step++;
    const double bc1 = 1 - std::pow(_beta1, _step);
    const double bc2 = 1 - std::pow(_beta2, _step);
    for (la::idx k = 0; k < theta.size(); ++k) {
      const double g = g_theta[k];
      _mtheta[k] = _beta1 * _mtheta[k] + (1 - _beta1) * g;
      _vtheta[k] = _beta2 * _vtheta[k] + (1 - _beta2) * g * g;
      const double mhat = _mtheta[k] / bc1;
      const double vhat = _vtheta[k] / bc2;
      theta[k] = (1 - _eta * _gamma) * theta[k] -
                 _eta * (mhat / (std::sqrt(vhat) + _eps));
    }
  }
};

#endif // RWIG_OPTIMIZER_H
