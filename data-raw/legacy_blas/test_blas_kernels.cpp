
#include <cpp11.hpp>

#include "blas_kernels.hpp"
#include "blas_utils.hpp"

using namespace cpp11;
using namespace blas;

/////////////////////////////////////////////////////////////////
// utilities:
/////////////////////////////////////////////////////////////////

[[cpp11::register]]
double test_mat_at(const doubles& x, int i, int j) {
  return at(x, i, j);
}

[[cpp11::register]]
double test_vec_at(const doubles& x, int i) {
  return at(x, i);
}

[[cpp11::register]]
doubles test_diag(const doubles& x) {
  return diag(x);
}

/////////////////////////////////////////////////////////////////
// Overloading Operators
/////////////////////////////////////////////////////////////////

[[cpp11::register]]
doubles test_div1(const doubles& x, const double a) {
  return x / a;
}

[[cpp11::register]]
doubles test_div2(const double a, const doubles& x) {
  return a / x;
}

[[cpp11::register]]
doubles test_times1(const doubles& x, const double a) {
  return x * a;
}

[[cpp11::register]]
doubles test_times2(const double a, const doubles& x) {
  return a * x;
}

/////////////////////////////////////////////////////////////////
// Extending BLAS: Hadamard stuff
/////////////////////////////////////////////////////////////////

// [[cpp11::register]]
// doubles test_dhmpd(const doubles& x, const doubles& y) {
//   writable::doubles x_(x);
//   dhmpd(x_, y);
//   return x_;
// }
//
// [[cpp11::register]]
// doubles test_dhmdv(const doubles& x, const doubles& y) {
//   writable::doubles x_(x);
//   dhmdv(x_, y);
//   return x_;
// }
//
// [[cpp11::register]]
// doubles test_dhmpw(const doubles& x, const doubles& y) {
//   writable::doubles x_(x);
//   dhmpw(x_, y);
//   return x_;
// }

[[cpp11::register]]
doubles test_pow(const doubles& x, const double a) {
  return pow(x, a);
}

[[cpp11::register]]
doubles test_exp(const doubles& x) {
  return exp(x);
}

[[cpp11::register]]
doubles test_log(const doubles& x) {
  return log(x);
}


/////////////////////////////////////////////////////////////////
// Level 1 BLAS: daxpy, ddot, dnrm2, dasum, idamax
/////////////////////////////////////////////////////////////////

[[cpp11::register]]
doubles test_daxpy(const doubles& y, const doubles& x, const double alpha) {
  writable::doubles y_(y);
  daxpy(y_, x, alpha);
  return y_;
}

[[cpp11::register]]
doubles test_dscal(const doubles& y, const double alpha) {
  writable::doubles y_(y);
  dscal(y_, alpha);
  return y_;
}

[[cpp11::register]]
double test_ddot(const doubles& a, const doubles& b) { return ddot(a, b); }
[[cpp11::register]]
double test_dnrm2(const doubles& a) { return dnrm2(a); }
[[cpp11::register]]
double test_dasum(const doubles& a) { return dasum(a); }
[[cpp11::register]]
double test_idamax(const doubles& a) { return idamax(a); }


/////////////////////////////////////////////////////////////////
// Level 2 BLAS: dgemv, dsymv, dtrmv, dtrsv
/////////////////////////////////////////////////////////////////

[[cpp11::register]]
doubles test_dgemv(const doubles& y, const doubles& A, const doubles& x,
                   const bool trans, const double alpha, const double beta) {
  writable::doubles y_(y);
  dgemv(y_, A, x, trans, alpha, beta);
  return y_;
}

[[cpp11::register]]
doubles test_dsymv(const doubles& y, const doubles& A, const doubles& x,
                   const bool upper, const double alpha, const double beta) {
  writable::doubles y_(y);
  dsymv(y_, A, x, upper, alpha, beta);
  return y_;
}

[[cpp11::register]]
doubles test_dtrmv(const doubles& x, const doubles& A,
                   const bool upper, const bool trans) {
  writable::doubles x_(x);
  dtrmv(x_, A, upper, trans);
  return x_;
}

[[cpp11::register]]
doubles test_dtrsv(const doubles& x, const doubles& A,
                   const bool upper, const bool trans) {
  writable::doubles x_(x);
  dtrsv(x_, A, upper, trans);
  return x_;
}


/////////////////////////////////////////////////////////////////
// Level 3 BLAS: dgemm, dsymm, dtrmm, dtrsm
/////////////////////////////////////////////////////////////////

[[cpp11::register]]
doubles test_dgemm(const doubles& C, const doubles& A, const doubles& B,
                   const bool transA, const bool transB,
                   const double alpha, const double beta) {
  writable::doubles C_(C);
  dgemm(C_, A, B, transA, transB, alpha, beta);
  return C_;
}

[[cpp11::register]]
doubles test_dsymm(const doubles& C, const doubles& A, const doubles& B,
                   const bool left, const bool upper,
                   const double alpha, const double beta) {
  writable::doubles C_(C);
  dsymm(C_, A, B, left, upper, alpha, beta);
  return C_;
}

[[cpp11::register]]
doubles test_dtrmm(const doubles& B, const doubles& A,
                   const bool left, const bool upper, const bool trans,
                   const double alpha) {
  writable::doubles B_(B);
  dtrmm(B_, A, left, upper, trans, alpha);
  return B_;
}

[[cpp11::register]]
doubles test_dtrsm(const doubles& B, const doubles& A,
                   const bool left, const bool upper, const bool trans,
                   const double alpha) {
  writable::doubles B_(B);
  dtrsm(B_, A, left, upper, trans, alpha);
  return B_;
}
