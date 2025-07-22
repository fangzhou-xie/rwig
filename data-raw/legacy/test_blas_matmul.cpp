
// using cpp11 for calling BLAS


///////////////////////////////////////////////////////////////
// working example: direct calling
///////////////////////////////////////////////////////////////

// #include <cpp11.hpp>
//
// #include <R_ext/BLAS.h>           // BLAS routines
// // #include <R_ext/Lapack.h>      // LAPACK routines
//
// using namespace cpp11;
//
// [[cpp11::register]]
// SEXP cpp11_matmul(const doubles_matrix<>& A_, const doubles_matrix<>& B_) {
//   const char * TA = "N";
//   const char * TB = "N";
//
//   double * A = REAL(A_.data());
//   double * B = REAL(B_.data());
//
//   int K = A_.ncol();
//   int M = A_.nrow();
//   int N = B_.ncol();
//
//   int LDA = K;
//   int LDB = N;
//
//   double ALPHA = 1.0;
//   double BETA = 0.0;
//
//   SEXP result_ = PROTECT(safe[Rf_allocMatrix](REALSXP, M, N));
//   double * result = REAL(result_);
//   F77_CALL(dgemm)(TA, TB, &M, &N, &K, &ALPHA,A, &LDA, B, &LDB, &BETA, result, &N,1,1);
//   // additional two 1's as the end, indicating the length of the two characters
//   // this is different from the C interface
//
//   UNPROTECT(1);
//
//   return(result_);
// }


///////////////////////////////////////////////////////////////
// working example: overloading the `*` operator
///////////////////////////////////////////////////////////////

#include <iostream>

#include <cpp11.hpp>

#include "cpp11blas.h"

using namespace cpp11;

[[cpp11::register]]
doubles test_hadamard_prod(const doubles& a, const doubles& b) {
  auto c = a % b;
  return(c);
}

[[cpp11::register]]
doubles test_hadamard_div(const doubles& a, const doubles& b) {
  auto c = a / b;
  return(c);
}

[[cpp11::register]]
doubles kernel_aKv(const doubles& a, const doubles& K, const doubles& v) {

  // using the overloaded operators
  // auto u = a / (K * v);

  // using the direct kernel
  // auto u = aKv(a, K, v);

  return(aKv(a, K, v));
}

// [[cpp11::register]]
// doubles kernel_aKv(const doubles& a, const doubles_matrix<>& K, const doubles& v) {
//   return(aKv(a, K, v));
// }
//
// [[cpp11::register]]
// doubles cpp11_aKv(const doubles& a, const doubles_matrix<>& K, const doubles& v) {
//   // compuate a / Kv
//   doubles Kv = K * v;
//   return(a / Kv);
// }

// [[cpp11::register]]
// doubles cpp11_matvec(const doubles_matrix<>& A, const doubles& B) {
//   auto C = A * B;
//   return (C);
// }

// [[cpp11::register]]
// doubles_matrix<> cpp11_matmul(const doubles_matrix<>& A, const doubles_matrix<>& B) {
//
//   auto C = A * B;
//   // SEXP C = cpp11::operator*(A, B);
//
//   return(C);
//   // return(A * B);
// }

