
// test armadillo matmul

// #define ARMA_USE_BLAS

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace cpp11;

[[cpp11::register]]
doubles_matrix<> matmul_arma(
    const doubles_matrix<>& A, const doubles_matrix<>& B) {
  arma::mat AA = as_Mat(A);
  arma::mat BB = as_Mat(B);
  mat C = AA * BB;
  return as_doubles_matrix(C);
}
