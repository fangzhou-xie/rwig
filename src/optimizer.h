
// implement the 3 optimizers: vanilla SGD, Adam, and AdamW
// https://arxiv.org/abs/2504.08722
// Section 7.1

#ifndef WIG_OPTIMIZER_H
#define WIG_OPTIMIZER_H

// #include <iostream> // for std::cout

#include <cpp11armadillo.hpp>
using namespace arma;


// implement the Optimizer class

class Optimizer {

private:

  // dimensions
  int _M, _N;
  // all the optimizer parameters
  double _eta, _gamma, _beta1, _beta2, _eps;
  // the aux parameters
  mat gtheta, mtheta, vtheta; // should be same dimension with theta
  // counter for step;
  int step;

public:

  // output var: param
  // mat theta;

  // init Optimizer
  Optimizer(
    const int M, const int N,
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
    gtheta = mat(_M, _N, fill::zeros);
    mtheta = mat(_M, _N, fill::zeros);
    vtheta = mat(_M, _N, fill::zeros);
    // counter
    step = 1;
  }

  // SGD update
  void sgd(mat& theta) {

    theta = theta - _eta * gtheta;

    step++;
  }

  // Adam update
  void adam(mat& theta) {

    mtheta = _beta1 * mtheta + (1 - _beta1) * gtheta;
    vtheta = _beta2 * vtheta + (1 - _beta2) * pow(gtheta, 2);
    mat mhat = mtheta / (1 - pow(_beta1, step));
    mat vhat = vtheta / (1 - pow(_beta2, step));
    theta = theta - _eta * (mhat / (sqrt(vhat) + _eps));

    step++;
  }

  // AdamW update
  void adamw(mat& theta) {

    mtheta = _beta1 * mtheta + (1 - _beta1) * gtheta;
    vtheta = _beta2 * vtheta + (1 - _beta2) * pow(gtheta, 2);
    mat mhat = mtheta / (1 - pow(_beta1, step));
    mat vhat = vtheta / (1 - pow(_beta2, step));
    // theta = theta - eta * (mhat / (sqrt(vhat) + eps)) - eta * gamma * theta;
    theta = (1 - _eta*_gamma) * theta - _eta * (mhat / (sqrt(vhat) + _eps));

    step++;
  }

};


#endif // WIG_OPTIMIZER_H

