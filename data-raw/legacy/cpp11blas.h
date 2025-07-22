
// header file to extend the BLAS operations to cpp11::doubles_matrix<>

#ifndef CPP11_BLAS_H
#define CPP11_BLAS_H

#include <iostream>

// BLAS routines
#include <R_ext/BLAS.h>

// TODO: reduce the headers to the minimal?
// #include <cpp11.hpp>
#include <cpp11/R.hpp>           // for SEXP
#include <cpp11/r_vector.hpp>    // for r_vector
#include <cpp11/doubles.hpp>     // for doubles aka r_vector<double>

using namespace cpp11;

// namespace writable = cpp11::writable; // writable list from cpp11

// return a readable/writable cpp11 matrix?

namespace cpp11 {

///////////////////////////////////////////////////////////
// Extending BLAS
///////////////////////////////////////////////////////////

// 1. Hadamard product/division

auto operator%(const doubles& a, const doubles& b) ->
  writable::r_vector<double>
{
  writable::r_vector<double> out(a.size());
  for (int i = 0; i < out.size(); ++i) {
    out[i] = a[i] * b[i];
  }
  return(out);
}

auto operator/(const doubles& a, const doubles& b) ->
  writable::r_vector<double>
  {
    writable::r_vector<double> out(a.size());
    for (int i = 0; i < out.size(); ++i) {
      out[i] = a[i] / b[i];
    }
    return(out);
  }

// custom kernel
auto aKv(const doubles& a, const doubles& K, const doubles& v) ->
  writable::doubles
  {
  // first Kv
  // Mat * Vec
  const char * TA = "N";
  const char * TB = "N";

  int * dimA = INTEGER(K.attr(R_DimSymbol));
  int * dimB = INTEGER(v.attr(R_DimSymbol));

  int M_ = dimA[0];
  int K_ = dimA[1];
  int N_ = dimB[1];

  double * A = REAL(K.data());
  double * B = REAL(v.data());

  int LDA = M_;
  int LDB = K_;
  int LDC = M_;

  double ALPHA = 1.0;
  double BETA = 0.0;


  SEXP Kv_ = PROTECT(safe[Rf_allocVector](REALSXP, M_*N_));
  double * Kv = REAL(Kv_);
  // F77_CALL(dgemv)(TA,&M,&N,&ALPHA,A,&LDA,x,&incx,&BETA,Kv,&incy,1);
  F77_CALL(dgemm)(TA,TB,&M_,&N_,&K_,&ALPHA,A,&LDA,B,&LDB,&BETA,Kv,&LDC,1,1);
  UNPROTECT(1);

  writable::doubles r(M_*N_);
  for (int i = 0; i < a.size(); ++i) {
    r[i] = a[i] / *(Kv + i);
  }
  r.attr(R_DimSymbol) = writable::r_vector<int>({M_, N_});
  return(r);
}



///////////////////////////////////////////////////////////
// Level 1 BLAS
///////////////////////////////////////////////////////////

// template<typename V, typename T, typename S = by_column>
// auto operator*(const matrix<V,T,S>& lhs, const double a) ->
//   doubles_matrix<>
//   {
//     //
//     int N = lhs.nco
//   }


///////////////////////////////////////////////////////////
// Level 2 BLAS
///////////////////////////////////////////////////////////

template <typename V, typename T, typename S = by_column>
auto operator*(const matrix<V,T,S>& lhs, const doubles& rhs) ->
  doubles
  {
    // Mat * Vec
    const char * TA = "N";

    int M = lhs.nrow();
    int N = lhs.ncol();

    double * A = REAL(lhs.data());
    double * x = REAL(rhs.data());

    int LDA = N;

    double ALPHA = 1.0;
    double BETA = 0.0;

    int incx = 1;
    int incy = 1;

    SEXP r_ = PROTECT(safe[Rf_allocVector](REALSXP, M));
    double * result = REAL(r_);
    F77_CALL(dgemv)(TA,&M,&N,&ALPHA,A,&LDA,x,&incx,&BETA,result,&incy,1);

    UNPROTECT(1);

    doubles r(r_);

    return r;
  }

///////////////////////////////////////////////////////////
// Level 3 BLAS
///////////////////////////////////////////////////////////

// template<typename V, typename T, typename S = by_column>
// auto operator*(const matrix<V,T1,S>& lhs, const double rhs) ->
//   doubles_matrix<>
//   {
//
//   }

auto operator*(const doubles& lhs, const doubles& rhs) -> writable::doubles
{
  // dgemm wrapper
  const char * TA = "N";
  const char * TB = "N";

  double * A = REAL(lhs.data());
  double * B = REAL(rhs.data());

  int * dimA = INTEGER(lhs.attr(R_DimSymbol));
  int * dimB = INTEGER(rhs.attr(R_DimSymbol));

  int M = dimA[0];
  int K = dimA[1];
  int N = dimB[1];

  // std::cout << M << " " << N << " " << K << std::endl;

  // int K = lhs.ncol();
  // int M = lhs.nrow();
  // int N = rhs.ncol();

  int LDA = M;
  int LDB = K;
  int LDC = M;

  double ALPHA = 1.0;
  double BETA = 0.0;

  // doubles aa(10);

  SEXP result_ = PROTECT(safe[Rf_allocMatrix](REALSXP, M, N));
  double * result = REAL(result_);
  F77_CALL(dgemm)(TA,TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,result,&LDC,1,1);
  // additional two 1's as the end, indicating the length of the two characters
  // this is different from the C interface

  UNPROTECT(1);

  // convert SEXP to matrix
  writable::doubles r(result_);
  r.attr(R_DimSymbol) = writable::r_vector<int>({M, N});

  return(r);
}

// // matrix gemm: C = alpha * A * B + beta * C
// template <typename V, typename T1, typename T2, typename S>
// auto operator*(const matrix<V,T1,S>& lhs, const matrix<V,T2,S>& rhs) ->
//   writable::doubles_matrix<S>
//   {
//     // dgemm wrapper
//     const char * TA = "N";
//     const char * TB = "N";
//
//     double * A = REAL(lhs.data());
//     double * B = REAL(rhs.data());
//
//     int K = lhs.ncol();
//     int M = lhs.nrow();
//     int N = rhs.ncol();
//
//     int LDA = K;
//     int LDB = N;
//
//     double ALPHA = 1.0;
//     double BETA = 0.0;
//
//     // doubles aa(10);
//
//     SEXP result_ = PROTECT(safe[Rf_allocMatrix](REALSXP, M, N));
//     double * result = REAL(result_);
//     F77_CALL(dgemm)(TA, TB, &M, &N, &K, &ALPHA,A, &LDA, B, &LDB, &BETA, result, &N,1,1);
//     // additional two 1's as the end, indicating the length of the two characters
//     // this is different from the C interface
//
//     UNPROTECT(1);
//
//     // convert SEXP to matrix
//     doubles_matrix<> r(result_);
//
//     return(r);
//   }

} // END of namespace cpp11




#endif // CPP11_BLAS_H



// // [[cpp11::register]]
// SEXP dynamic_matrix_approach1() {
//   // Method 1: Start with a vector, resize and reshape as needed
//   writable::r_vector<double> vec;  // Start empty
//
//   // Dynamically add data
//   vec.push_back(1.0);
//   vec.push_back(2.0);
//   vec.push_back(3.0);
//   vec.push_back(4.0);
//   vec.push_back(5.0);
//   vec.push_back(6.0);
//
//   // Now decide dimensions (2 rows x 3 columns)
//   vec.attr(R_DimSymbol) = writable::r_vector<int>({2, 3});
//
//   // Create matrix from the vector
//   writable::doubles_matrix<> mat(vec.data());
//
//   return mat;
// }
