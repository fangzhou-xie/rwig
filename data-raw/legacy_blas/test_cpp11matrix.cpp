

#include <cpp11.hpp>

#include "cpp11matrix.hpp"

using namespace cpp11;

[[cpp11::register]]
doubles test_init_matrix(int m, int n) {
  return Matrix(m,n);
}

[[cpp11::register]]
double test_index_matrix(const doubles& a, int i, int j) {
  Matrix m{Matrix(a)};
  return m(i,j);
}

[[cpp11::register]]
doubles test_tran_matrix(const doubles& a) {
  return Matrix(a).t().as_doubles();
}


