
// test performance of different functions

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace arma;
using namespace cpp11;

[[cpp11::register]]
doubles_matrix<>& f1(doubles_matrix<>& CR, double reg) {
  mat C = as_Mat(CR);
  vec u = vec(C.n_rows, fill::ones);
  vec v = vec(C.n_cols, fill::ones);
  mat K = exp(-C/reg);
  mat P = diagmat(u) * K * diagmat(v);
  return as_doubles_matrix(P);
}

[[cpp11::register]]
doubles_matrix<>& f2(doubles_matrix<>& CR, double reg) {
  mat C = as_Mat(CR);
  vec u = vec(C.n_rows, fill::ones);
  vec v = vec(C.n_cols, fill::ones);
  vec onesN = vec(C.n_cols, fill::ones);
  vec onesM = vec(C.n_rows, fill::ones);
  mat K = exp(-C/reg);
  mat R = C - reg*log(u) * onesN.t() - onesM * reg*log(v).t();
  mat P = exp(-R/reg);
  return as_doubles_matrix(P);
}
