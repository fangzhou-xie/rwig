
// the BLAS kernels for computing the OT routines

// 1. Vanilla Sinkhorn

#ifndef WIG_KERNELS_H
#define WIG_KERNELS_H

// BLAS routines
#include <R_ext/BLAS.h>

// only import required headers?
#include <cpp11/R.hpp>           // for SEXP
#include <cpp11/r_vector.hpp>    // for r_vector
#include <cpp11/matrix.hpp>      // for matrix

using namespace cpp11;


namespace kernels {

// a / Kv
// TODO: there is no Hadamard-family operators in BLAS

template<typename V, typename T, typename S = by_column>
auto a_over_Kv(
    const matrix<V,T,S>& a, const matrix<V,T,S>& K, const matrix<V,T,S>& v) ->
  doubles_matrix<> {
    // update u <- a / (K * v)

    // first calculate Kv
  }








}


#endif // WIG_KERNELS_H
