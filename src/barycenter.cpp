

// #define ARMA_USE_BLAS // force using BLAS to be faster
// #define ARMA_NO_DEBUG // disable boundary checks


// #include <iostream>             // std::cout
// #include <iomanip>              // std::precision
// #include "R_ext/Print.h"        // for REprintf

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "barycenter.h"  // implementation of the barycenter in header file

using namespace arma;
using namespace cpp11;
using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable; // writable list from cpp11


//////////////////////////////////////////////////////////////////////
// Interfaces for the R side
//////////////////////////////////////////////////////////////////////

[[cpp11::register]]
writable::list barycenter_parallel_cpp(
    const doubles_matrix<>& A, const doubles_matrix<>& C,
    const doubles_matrix<>& w, double reg,
    bool withjac = false,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {
  // convert R matrices into arma ones
  mat A_ = as_Mat(A);
  mat C_ = as_Mat(C);
  vec w_ = as_Mat(w);

  // init the class
  Barycenter bc(C_.n_rows, C_.n_cols, A_.n_cols,
                withjac, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  bc.init_parallel();

  // start the computation
  bc.compute_parallel();

  // output list
  writable::list res;
  res.push_back({"b"_nm = bc.b});
  if (withjac) {
    res.push_back({"JbA"_nm = as_doubles_matrix(bc._JbA)});
    res.push_back({"Jbw"_nm = as_doubles_matrix(bc._Jbw)});
  }
  res.push_back({"U"_nm = as_doubles_matrix(bc._U)});
  res.push_back({"V"_nm = as_doubles_matrix(bc._V)});
  res.push_back({"iter"_nm = bc._iter});
  res.push_back({"err"_nm = bc._err});
  return res;
}

[[cpp11::register]]
writable::list barycenter_log_cpp(
    const doubles_matrix<>& A, const doubles_matrix<>& C,
    const doubles_matrix<>& w, double reg,
    bool withjac = false,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {
  // convert R matrices into arma ones
  mat A_ = as_Mat(A);
  mat C_ = as_Mat(C);
  vec w_ = as_Mat(w);

  // init the class
  Barycenter bc(C_.n_rows, C_.n_cols, A_.n_cols,
                withjac, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C_);
  bc.update_reg(reg);
  bc.update_A(A_);
  bc.update_w(w_);
  bc.init_log();

  // start the computation
  bc.compute_log();

  // output list
  writable::list res;
  res.push_back({"b"_nm = bc.b});
  if (withjac) {
    res.push_back({"JbA"_nm = as_doubles_matrix(bc._JbA)});
    res.push_back({"Jbw"_nm = as_doubles_matrix(bc._Jbw)});
  }
  res.push_back({"F"_nm = as_doubles_matrix(bc._U)});
  res.push_back({"G"_nm = as_doubles_matrix(bc._V)});
  res.push_back({"iter"_nm = bc._iter});
  res.push_back({"err"_nm = bc._err});
  return res;
}
