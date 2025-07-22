
// header for the barycenter class with cpp11 doubles_matrix and BLAS

#ifndef WIG_BARYCENTER_BLAS_H
#define WIG_BARYCENTER_BLAS_H

#include <cpp11.hpp>

// https://github.com/Compaile/ctrack
#include "ctrack.hpp"             // profiler


using namespace cpp11;
// using namespace cpp11::literals; // so we can use ""_nm syntax
namespace writable = cpp11::writable; // writable list from cpp11

class Barycenter {

private:

  // class init control parameters
  bool _withjac;
  int _maxiter;
  double _zerotol;
  // dimentions
  int _M, _N, _S;

public:

  // scaling vars
  writable::doubles _U, _V;                     // F,G in log

  // output centroid b
  writable::doubles b;
  // output Jacobians of b
  writable::doubles _JbA, _Jbw;

  // termination
  int _iter = 0;
  double _err = 1000.;

  // init the class
  Barycenter(int M, int N, int S, bool withjac = false,
             int maxiter = 1000, double zerotol = 1e-6) {
    // all the dimension
    _M = M;
    _N = N;
    _S = S;
    // compute jacobian?
    _withjac = withjac;
    // other control parameters
    _maxiter = maxiter;
    _zerotol = zerotol;
    // aux vars
    // _onesM = vec(_M, fill::ones);
    // _onesN = vec(_N, fill::ones);
    // _onesS = vec(_S, fill::ones);
  }
};


#endif // WIG_BARYCENTER_BLAS_H
