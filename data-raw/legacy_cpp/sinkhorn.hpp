
// implement the Sinkhorn algorithm and its gradients with class
// in this header file
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#ifndef WIG_SINKHORN_H
#define WIG_SINKHORN_H

#include "timer.hpp"                // TicToc timer class

#include <cpp11armadillo.hpp>
using namespace arma;


// implement the Sinkhorn algos in this class

class Sinkhorn {

private:

  bool _withgrad;
  int _maxiter;
  double _zerotol;
  int _verbose;

  // variables for the reduced form
  int _M, _N; // dimensions
  // data
  vec _a,_b; // a and b whose elements are not 0
  mat _C; // cost matrix corresponding to a and b whose elements are not 0
  double _reg;

  // additional vars for log solution
  vec _onesM, _onesN;

  TicToc _timer; // timer class


  // implementation of the vanilla sinkhorn algorithm
  void impl_vanilla();

  void _update_u(vec& _Kv);
  void _update_Ju(vec& _Kv);
  // void _update_Ju2(vec& _Kv, vec& _KTu);
  void _update_v(vec& _KTu);
  void _update_Jv(vec& _KTu);
  void _norm_vanilla(vec& _KTu);

  // implementation of the log sinkhorn algorithm
  void impl_log();

  void _minrow(vec& Rminrow);
  void _mincol(vec& Rmincol);
  void _minrowjac(mat& W, vec& Rminrow);
  void _mincoljac(mat& X, vec& Rmincol);
  void _update_Jf(mat& W, const mat& regdiaga);
  void _update_Jg(mat& X);


  void reset_counter() {
    _iter = 0;
    _err = 1000.;
  }

public:

  // intermediate variables
  vec _u, _v; // u,v in vanilla, f,g in log
  mat _Ju, _Jv; // Ju,Jv in vanilla, Jf,Jg in log
  mat _K; // K in vanilla, R in log
  vec _grad_a;
  mat _P;

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
  void compute_vanilla(vec& a, vec& b, mat& C, double reg);

  // compute log Sinkhorn
  void compute_log(vec& a, vec& b, mat& C, double reg);

};


#endif // WIG_SINKHORN_H
