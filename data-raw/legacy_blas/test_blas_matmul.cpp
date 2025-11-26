
// using cpp11 for calling BLAS


///////////////////////////////////////////////////////////////
// working example: direct calling
///////////////////////////////////////////////////////////////

#include <iostream>

#include <R_ext/BLAS.h>           // BLAS routines
// #include <R_ext/Lapack.h>      // LAPACK routines

#include <cpp11.hpp>
#include <cpp11/doubles.hpp>

using namespace cpp11;

[[cpp11::register]]
SEXP cpp11_trmatmul(const doubles& A_, const doubles& B_) {
  const char * TA = "T";
  const char * TB = "N";

  double * A = REAL(A_.data());
  double * B = REAL(B_.data());

  int M = Rf_nrows(A_);
  int K = Rf_ncols(A_);
  int N = Rf_ncols(B_);

  int LDA = M;
  int LDB = M;
  int LDC = K;

  double ALPHA = 1.0;
  double BETA = 0.0;

  SEXP result_ = PROTECT(safe[Rf_allocMatrix](REALSXP, K, N));
  double * result = REAL(result_);
  F77_CALL(dgemm)(
      TA, TB, &K, &N, &M, &ALPHA,A, &LDA, B, &LDB, &BETA, result, &LDC,1,1);
  // additional two 1's as the end, indicating the length of the two characters
  // this is different from the C interface

  UNPROTECT(1);
  // Rf_setAttrib(result_, R_DimSymbol, r_vector<int>({K, N}));

  return(result_);
}

///////////////////////////////////////////////////////////////
// testing example: use Matrix class
///////////////////////////////////////////////////////////////

// #include <cpp11.hpp>
//
// #include "cpp11matrix.hpp"
//
// using namespace cpp11;
//
//
// [[cpp11::register]]
// doubles test_load_matrix(const doubles& a) {
//   // Matrix amat = as_matrix(a);
//   Matrix amat = Matrix(a);
//   // amat.check_symmetric();
//   // message("nrow: %d, ncol: %d", amat.n_rows, amat.n_cols);
//   // amat.is_symmetric ? message("true") : message("false");
//
//   return(amat.data());
// }



///////////////////////////////////////////////////////////////
// working example: overloading the `*` operator
///////////////////////////////////////////////////////////////

// #include <iostream>
//
// #include <cpp11.hpp>
//
// #include "cpp11blas.hpp"
//
// using namespace cpp11;
//

//
// [[cpp11::register]]
// doubles test_hadamard_div(const doubles& a, const doubles& b) {
//   auto c = a / b;
//   return(c);
// }
//
// [[cpp11::register]]
// doubles kernel_aKv(const doubles& a, const doubles& K, const doubles& v) {
//
//   // using the overloaded operators
//   // auto u = a / (K * v);
//
//   // using the direct kernel
//   // auto u = aKv(a, K, v);
//
//   return(aKv(a, K, v));
// }
//
// [[cpp11::register]]
// doubles blas_aKv(const doubles& a, const doubles& K, const doubles& v) {
//
//   // using the overloaded operators
//   auto u = a / (K * v);
//
//   return(u);
// }
//
// [[cpp11::register]]
// doubles kernel_matmul(const doubles& A, const doubles& B) {
//   return(A * B);
// }

// [[cpp11::register]]
// doubles kernel_aKv(const doubles& a, const doubles_matrix<>& K, const doubles& v) {
//   return(aKv(a, K, v));
// }

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

