
// implement Sinkhorn algorithm with the custom wrapped BLAS routines


#ifndef WIG_SINKHORN_BLAS_HPP
#define WIG_SINKHORN_BLAS_HPP

#include "blas_kernels.hpp"  // BLAS kernels

#include "timer.hpp"         // TicToc timer class

using namespace blas;

namespace wig {

class Sinkhorn {

private:

  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _K_is_symm = false;

  unsigned int _M, _N; // dimensions
  // data
  doubles _a,_b; // a and b whose elements are not 0
  doubles _C; // cost matrix corresponding to a and b whose elements are not 0
  double _reg;

  // additional vars for log solution
  doubles _onesM, _onesN;

  TicToc _timer; // timer class

  void update_Kv(writable::doubles& _Kv);
  void update_KTu(writable::doubles& _KTu);
  void update_u(writable::doubles& _u, const doubles& _Kv);
  void update_v(writable::doubles& _v, const doubles& _KTu);

  void impl_vanilla();
  void impl_log();

  void reset_counter() { _iter = 0; _err = 1000.; }

public:

  // intermediate variables
  writable::doubles _u, _v; // u,v in vanilla, f,g in log
  writable::doubles _Ju, _Jv; // Ju,Jv in vanilla, Jf,Jg in log
  writable::doubles _K; // K in vanilla, R in log
  writable::doubles _grad_a;
  writable::doubles _P;

  // counter for termination
  int _iter = 0;
  double _err = 1000.;

  // init sinkhorn method
  Sinkhorn(bool withgrad = false, // bool reduce = false,
           int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    _withgrad = withgrad;
    // _reduce = reduce;
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
  }
  ~Sinkhorn() {}

  // compute vanilla Sinkhorn
  void compute_vanilla(const doubles& a, const doubles&  b, const doubles& C,
                       const double reg);

  // compute log Sinkhorn
  void compute_log(const doubles& a, const doubles&  b, const doubles& C,
                   const double reg);

};


}

#endif // WIG_SINKHORN_BLAS_HPP
