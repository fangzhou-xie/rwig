
#include <cpp11.hpp>

#include "ctrack.hpp"

#include "fastexp.h"

#include "sinkhorn_blas.hpp"
#include "blas_utils.hpp"

using namespace wig;

// std::ostream& operator<<(std::ostream& os, const doubles& x) {
//   for (int i{0}; i < Rf_nrows(x); ++i) {
//     os << "  ";
//     for (int j{0}; j < Rf_ncols(x); ++j) {
//       os << std::setw(6) << std::setprecision(3) << at(x, i, j) << "  ";
//     }
//     os << std::endl;
//   }
//   return os;
// }

// doubles diaguKdiagv(const doubles& u, const doubles& K, const doubles& v) {
//   doubles diagu{diag(u)};
//   doubles diagv{diag(v)};
//   writable::doubles P{zeros(u.size(), v.size())};
//   dtrmm(P, diagu, false, false, false, 1);
//   dtrmm(P, diagv, true, false, false, 1);
//   return P;
// }

// y = a / x
// void over(writable::doubles& y, const doubles& a, const doubles& x) {
//   CTRACK;
//   for (int i = 0; i < y.size(); ++i) {
//     y[i] = a[i] / x[i];
//   }
// }

void Sinkhorn::compute_vanilla(const doubles& a, const doubles& b,
                               const doubles& C, const double reg) {
  CTRACK;
  reset_counter();

  unsigned int M = static_cast<unsigned int>(Rf_nrows(C));
  unsigned int N = static_cast<unsigned int>(Rf_ncols(C));
  integers _a_ind{find_value(a, 0, false)};
  integers _b_ind{find_value(b, 0, false)};

  // print_vec(_a_ind, "a ind");
  // print_vec(_b_ind, "b ind");

  // reduced form
  _a = select(a, _a_ind);
  _b = select(b, _b_ind);
  _C = select(C, _a_ind, _b_ind);
  // print_vec(_a, "_a");
  // print_vec(_b, "_b");
  // print_mat(_C, "_C");
  _reg = reg;
  _M = nrow(_C);
  _N = ncol(_C);

  // compute the reduced form K and effective symmetry
  _K = C2K(_C, _reg);
  _K_is_symm = is_symmetric(_K);

  impl_vanilla();

  // compute the grad wrt a

  // recover the original solution
  writable::doubles u{zeros(M)};
  writable::doubles v{zeros(N)};
  set(u, _u, _a_ind);
  set(v, _v, _b_ind);

  // use matrix multiplication
  // _K = C2K(C, reg);
  // doubles diagu = diag(_u);
  // doubles diagv = diag(_v);
  // dtrmm(_K, diagu, true, false, false, 1);
  // dtrmm(_K, diagv, false, false, false, 1);

  // use direct kernel
  // writable::doubles P{zeros(M, N)};
  doubles P{uKv(_u, _K, _v)};

  _P = zeros(M, N);
  set(_P, P, _a_ind, _b_ind);
  _u = u;
  _v = v;
  // print_mat(P, "P");

  // only for testing performance!
  // ctrack::result_print();

}

void Sinkhorn::update_Kv(writable::doubles& _Kv) {
  // arma: _Kv = _K * _v;
  if (_K_is_symm) {
    dsymv(_Kv, _K, _v, false, 1, 0);
  } else {
    dgemv(_Kv, _K, _v, false, 1, 0);
  }
  // dgemv(_Kv, _K, _v, false, 1, 0);
}

void Sinkhorn::update_KTu(writable::doubles& _KTu) {
  // arma: _KTu = _K.t() * _u;
  if (_K_is_symm) {
    dsymv(_KTu, _K, _u, false, 1, 0);
  } else {
    dgemv(_KTu, _K, _u, true, 1, 0);
  }
  // dgemv(_KTu, _K, _u, true, 1, 0);
}

void Sinkhorn::update_u(writable::doubles& _u, const doubles& _Kv) {
  over(_u, _a, _Kv);
}

void Sinkhorn::update_v(writable::doubles& _v, const doubles& _KTu) {
  over(_v, _b, _KTu);
}


void Sinkhorn::impl_vanilla() {
  CTRACK;
  _u = ones(_M);
  _v = ones(_N);
  // doubles _u_prev(_u); // copy
  // doubles _v_prev(_v); // copy
  // _K = C2K(_C, _reg);
  // _K = exp(- _C  / _reg);
  // if (_K_is_symm) {
  //   std::cout << "_K is symm" << std::endl;
  // } else {
  //   std::cout << "_K is not symm" << std::endl;
  // }
  // print_mat(_K, "K");

  writable::doubles _Kv = zeros(_M);
  writable::doubles _KTu = zeros(_N);

  // dgemv(_Kv, _K, _v, false, 1, 0);
  // print_vec(_Kv, "Kv");
  // dsymv(_Kv, _K, _v, false, 1, 0);
  // print_vec(_Kv, "Kv");


  update_Kv(_Kv);

  while ((_iter < _maxiter) && (_err >= _zerotol)) {
    _timer.tic();
    cpp11::check_user_interrupt();

    // update _u
    // update_Kv(_Kv);
    // _u = _a / _Kv;
    update_u(_u, _Kv);
    // print_vec(_Kv, "Kv");
    // print_vec(_u, "u");

    // update _v
    update_KTu(_KTu);
    // _v = _b / _KTu;
    update_v(_v, _KTu);
    // print_vec(_KTu, "KTu");
    // print_vec(_v, "v");

    update_Kv(_Kv);

    _iter++;
    _err = norm(_u, _Kv, _a) + norm(_v, _KTu, _b);
    _timer.toc();
  }
}
