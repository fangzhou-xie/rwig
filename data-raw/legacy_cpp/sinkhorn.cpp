

// #include <iostream> // cout

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "sinkhorn.hpp" // implementations in this header file

#include "ctrack.hpp"


using namespace arma;
using namespace cpp11;
using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable; // writable list from cpp11


//////////////////////////////////////////////////////////////////////
// Interfaces for the R side
//////////////////////////////////////////////////////////////////////


[[cpp11::register]]
writable::list sinkhorn_vanilla_cpp(
  const doubles_matrix<>& a, const doubles_matrix<>& b,
  const doubles_matrix<>& C, double reg,
  bool withgrad = false, // bool reduce = false,
  int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {
  // convert the R vectors/matrices into arma ones
  vec a_{as_Mat(a)};
  vec b_{as_Mat(b)};
  mat C_{as_Mat(C)};

  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_vanilla(a_, b_, C_, reg);

  // ctrack::result_print();

  // output list
  writable::list res;
  res.push_back({"P"_nm = as_doubles_matrix(s._P)});
  if (withgrad) {
    res.push_back({"grad_a"_nm = s._grad_a});
  }
  res.push_back({"u"_nm = s._u});
  // if (withgrad) {
  //   res.push_back({"Ju"_nm = as_doubles_matrix(s.Ju)});
  // }
  res.push_back({"v"_nm = s._v});
  // if (withgrad) {
  //   res.push_back({"Jv"_nm = as_doubles_matrix(s.Jv)});
  // }
  // res.push_back({"K"_nm = as_doubles_matrix(s.K)});
  res.push_back({"iter"_nm = s._iter});
  res.push_back({"err"_nm = s._err});
  return res;
}

[[cpp11::register]]
writable::list sinkhorn_log_cpp(
  const doubles_matrix<>& a, const doubles_matrix<>& b,
  const doubles_matrix<>& C, double reg,
  bool withgrad = false,
  int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {
  // convert the R vectors/matrices into arma ones
  vec a_{as_Mat(a)};
  vec b_{as_Mat(b)};
  mat C_{as_Mat(C)};

  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_log(a_, b_, C_, reg);

  ctrack::result_print();

  // output list
  writable::list res;
  res.push_back({"P"_nm = as_doubles_matrix(s._P)});
  if (withgrad) {
    res.push_back({"grad_a"_nm = s._grad_a});
  }
  res.push_back({"f"_nm = s._u});
  // if (withgrad) {
  //   res.push_back({"Jf"_nm = as_doubles_matrix(s.Ju)});
  // }
  res.push_back({"g"_nm = s._v});
  // if (withgrad) {
  //   res.push_back({"Jg"_nm = as_doubles_matrix(s.Jv)});
  // }
  res.push_back({"iter"_nm = s._iter});
  res.push_back({"err"_nm = s._err});
  return res;
}

