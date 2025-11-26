
// #define ARMA_DONT_USE_OPENMP

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace arma;

[[cpp11::register]]
doubles_matrix<> arma_matmul(const doubles_matrix<>& A, const doubles_matrix<>& B) {
  arma::mat A_{as_Mat(A)};
  arma::mat B_{as_Mat(B)};
  arma::mat C = A_ * B_;
  return(as_doubles_matrix(C));
}

[[cpp11::register]]
doubles_matrix<> arma_trmatmul(const doubles_matrix<>& A, const doubles_matrix<>& B) {
  arma::mat A_{as_Mat(A)};
  arma::mat B_{as_Mat(B)};
  arma::mat C = A_.t() * B_;
  return(as_doubles_matrix(C));
}

// [[cpp11::register]]
// doubles_matrix<> arma_matmul2(const doubles_matrix<>& A, const doubles_matrix<>& B) {
//   arma::mat A_{as_Mat(A)};
//   arma::mat B_{as_Mat(B)};
// }

// [[cpp11::register]]
// doubles_matrix<> arma_aKv(const doubles_matrix<>& a,
//                  const doubles_matrix<>& K,
//                  const doubles_matrix<>& v) {
//   mat a_{as_Mat(a)};
//   mat K_{as_Mat(K)};
//   mat v_{as_Mat(v)};
//   mat aKv = a_ / (K_ * v_);
//   return(as_doubles_matrix(aKv));
// }
//
//
// [[cpp11::register]]
// doubles_matrix<> arma_hadamard_prod(
//     const doubles_matrix<>& A, const doubles_matrix<>& B) {
//   vec a{as_Mat(A)};
//   vec b{as_Mat(B)};
//   vec c = a % b;
//   // doubles_matrix<> c = as_doubles_matrix(a % b);
//   return(as_doubles_matrix(c));
// }
//
// [[cpp11::register]]
// doubles_matrix<> arma_hadamard_div(
//     const doubles_matrix<>& A, const doubles_matrix<>& B) {
//   vec a{as_Mat(A)};
//   vec b{as_Mat(B)};
//   vec c = a / b;
//   return(as_doubles_matrix(c));
// }
