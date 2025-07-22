
#ifndef WIG_UTILS_H
#define WIG_UTILS_H

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
using namespace arma;

/////////////////////////////////////////////////////////////////////
// Optimizers
/////////////////////////////////////////////////////////////////////

// optimizer: SGD
void optimizer_sgd(mat &theta, mat &gtheta,
                   mat &mtheta, mat &vtheta,
                   int &t,
                   const double eta = .001, const double gamma = .01,
                   double beta1 = .9, double beta2 = .999,
                   double eps = 1e-8);

// optimizer: Adam
void optimizer_adam(mat &theta, mat &gtheta,
                    mat &mtheta, mat &vtheta,
                    int &t,
                    const double eta = .001, const double gamma = .01,
                    double beta1 = .9, double beta2 = .999,
                    double eps = 1e-8);


// optimizer: AdamW
void optimizer_adamw(mat &theta, mat &gtheta,
                     mat &mtheta, mat &vtheta,
                     int &t,
                     const double eta = .001, const double gamma = .01,
                     double beta1 = .9, double beta2 = .999,
                     double eps = 1e-8);


#endif // WIG_UTILS_H
