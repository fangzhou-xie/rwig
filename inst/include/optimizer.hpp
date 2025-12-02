
// implement the 3 optimizers: vanilla SGD, Adam, and AdamW
// https://arxiv.org/abs/2504.08722
// Section 7.1

#ifndef RWIG_OPTIMIZER_H
#define RWIG_OPTIMIZER_H

#include "common.hpp"

using namespace arma;

// implement the Optimizer class

class Optimizer {

private:

  // dimensions
  uword _M, _N;

  // all the optimizer parameters
  double _eta, _gamma, _beta1, _beta2, _eps;

  // the aux parameters
  // mat _gtheta;
  mat _mtheta, _vtheta; // should be same dimension with theta
  mat _mhat, _vhat;

  // counter for step;
  int _step;

public:

  // output var: param
  // mat theta;

  Optimizer() {}
  ~Optimizer() {}

  // init Optimizer
  void init(
      const uword M, const uword N,
      const double eta, const double gamma,
      const double beta1, const double beta2,
      const double eps
  ) {
    // load the parameters
    _eta = eta;
    _gamma = gamma;
    _beta1 = beta1;
    _beta2 = beta2;
    _eps = eps;

    // TODO: also set up the dimensions?
    _M = M; // number of rows in M
    _N = N; // number of cols in N

    // setup the aux vars
    // _gtheta = mat(_M, _N, fill::zeros);
    _mtheta = mat(_M, _N, fill::zeros);
    _vtheta = mat(_M, _N, fill::zeros);

    // setup the target (optimized) var
    // theta = mat(_M, _N, fill::zeros);

    // counter
    _step = 0;
  }

  // SGD update
  void sgd(mat& theta, mat& g_theta) {
    _step++;

    theta = theta - _eta * g_theta;
  }

  // Adam update
  void adam(mat& theta, mat& g_theta) {
    _step++;

    _mtheta = _beta1 * _mtheta + (1 - _beta1) * g_theta;
    _vtheta = _beta2 * _vtheta + (1 - _beta2) * pow(g_theta, 2);
    _mhat = _mtheta / (1 - pow(_beta1, _step));
    _vhat = _vtheta / (1 - pow(_beta2, _step));
    theta = theta - _eta * (_mhat / (sqrt(_vhat) + _eps));
  }

  // AdamW update
  void adamw(mat& theta, mat& g_theta) {
    _step++;

    _mtheta = _beta1 * _mtheta + (1 - _beta1) * g_theta;
    _vtheta = _beta2 * _vtheta + (1 - _beta2) * pow(g_theta, 2);
    _mhat = _mtheta / (1 - pow(_beta1, _step));
    _vhat = _vtheta / (1 - pow(_beta2, _step));
    // theta = theta - eta * (mhat / (sqrt(vhat) + eps)) - eta * gamma * theta;
    theta = (1 - _eta*_gamma) * theta - _eta * (_mhat / (sqrt(_vhat) + _eps));
  }

};


#endif // RWIG_OPTIMIZER_H

