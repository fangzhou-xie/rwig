
// header file for the utility functions

#ifndef WIG_UTILS_LEGACY_H
#define WIG_UTILS_LEGACY_H

// #define ARMA_DONT_USE_WRAPPER
#define ARMA_USE_BLAS
#define ARMA_USE_LAPACK

#include <cpp11armadillo.hpp>
using namespace arma;

// helper functions for initing matrix
mat I(const int n);
vec ones(const int n);
vec zeros(const int n);

// R function, R = C - f * onesN' - onesM * g';
void Rf(mat &R, mat &C, vec f, vec g);

// minrow and mincol functions for sinkhorn
void minrow(vec &Rminrow, mat &R, const double reg);
void mincol(vec &Rminrow, mat &R, const double reg);

// Minrow and Mincol functions for sinkhorn WITH Jacobian
void minrowjac(mat &W, vec &Rminrow, mat &R, const double reg);
void mincoljac(mat &V, vec &Rmincol, mat &R, const double reg) ;

// minrow and mincol functions for barycenter
void minrow(mat &Rminrow, mat &C, mat &F, mat &G, const double reg);
void mincol(mat &Rmincol, mat &C, mat &F, mat &G, const double reg);

// minrow and mincol functions for barycenter WITH Jacobian
void minrowjac(sp_mat &W, mat &Rminrow, mat &C, mat &F, mat &G, const double reg);
void mincoljac(sp_mat &V, mat &Rmincol, mat &C, mat &F, mat &G, const double reg);


// Optimizers
// void optimizer_sgd(mat &theta, mat &gtheta,
//                    mat &mtheta, mat &vtheta,
//                    int &t,
//                    const double eta = .001, const double gamma = .01,
//                    double beta1 = .9, double beta2 = .999,
//                    double eps = 1e-8);
// void optimizer_adam(mat &theta, mat &gtheta,
//                     mat &mtheta, mat &vtheta,
//                     int &t,
//                     const double eta = .001, const double gamma = .01,
//                     double beta1 = .9, double beta2 = .999,
//                     double eps = 1e-8);
// void optimizer_adamw(mat &theta, mat &gtheta,
//                      mat &mtheta, mat &vtheta,
//                      int &t,
//                      const double eta = .001, const double gamma = .01,
//                      double beta1 = .9, double beta2 = .999,
//                      double eps = 1e-8);

// Softmax and its Jacobian
void softmax(mat& softmaxA, mat& A);
void softmaxjac(mat& JsoftmaxA, mat& softmaxA, mat& A);
void softmax_jac_only(mat &JsoftmaxA, mat &A);


#endif // WIG_UTILS_LEGACY_H
