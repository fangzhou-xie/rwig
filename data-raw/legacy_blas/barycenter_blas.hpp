

#ifndef WIG_BARYCENTER_BLAS_HPP
#define WIG_BARYCENTER_BLAS_HPP

// #include "blas_kernels.hpp"  // BLAS kernels
#include "blas_utils.hpp"

#include "timer.hpp"

using namespace blas;

using uint = unsigned int;

// using namespace
namespace wig {

class Barycenter {

private:

  // class init control parameters
  bool _withjac;
  int _maxiter;
  double _zerotol;
  int _verbose;
  bool _C_is_symm = false;

  // data
  doubles _A, _C, _w, _K;
  double _reg;
  uint _M, _N, _S;
  doubles _onesM, _onesN, _onesS;
  doubles onesNwT;

  TicToc _timer;


  void update_parallel_U(writable::doubles& _KV);
  void update_parallel_V(writable::doubles& _KTU);
  void update_parallel_b(const doubles& onesNwT);

  void impl_parallel();
  void reset_counter() { _iter = 0; _err = 1000.; }


public:

  writable::doubles _U, _V, b;

  int _iter = 0;
  double _err = 1000.;

  // init the class
  Barycenter(unsigned int M, unsigned int N, unsigned int S,
             bool withjac = false,
             int maxiter = 1000, double zerotol = 1e-6, int verbose = 0) {
    // all the dimension
    _M = M;
    _N = N;
    _S = S;
    // compute jacobian?
    _withjac = withjac;
    // other control parameters
    _maxiter = maxiter;
    _zerotol = zerotol;
    _verbose = verbose;
    // aux vars
    _onesM = zeros(_M);
    _onesN = zeros(_N);
    _onesS = zeros(_S);
  }
  ~Barycenter() {}

  void update_C(const doubles& C) {
    _C = C;
    _C_is_symm = is_symmetric(C);
  }
  void update_reg(double reg) { _reg = reg; }
  void update_A(const doubles& A) { _A = A; }
  void update_w(const doubles& w) { _A = w; }
  void update_withjac(bool withjac) { _withjac = withjac; }

  void compute_parallel();

};

}

#endif // WIG_BARYCENTER_BLAS_HPP
