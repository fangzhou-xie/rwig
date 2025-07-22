
// directly calling BLAS kernels for barycenter

#include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include "barycenter_blas.h"  // implementation of the barycenter in header file

using namespace cpp11;
using namespace cpp11::literals; // so we can use ""_nm syntax
namespace writable = cpp11::writable; // writable list from cpp11


[[cpp11::register]]
writable::list barycenter_parallel_blas(
  const doubles& A, const doubles& C,
  const doubles& w, double reg,
  bool withjac = false,
  int maxiter = 1000, double zerotol = 1e-6
) {

  // // init the class
  // Barycenter bc(C.nrows(), C.n_cols, A.n_cols, withjac, maxiter, zerotol);
  // // update/load all the data
  // bc.update_C(C);
  // bc.update_reg(reg);
  // bc.update_A(A);
  // bc.update_w(w);
  // bc.init_parallel();
  //
  // // start the computation
  // bc.compute_parallel();

  // output list
  writable::list res;
  // res.push_back({"b"_nm = bc.b});
  // if (withjac) {
  //   res.push_back({"JbA"_nm = bc._JbA});
  //   res.push_back({"Jbw"_nm = bc._Jbw});
  // }
  // res.push_back({"U"_nm = bc._U});
  // res.push_back({"V"_nm = bc._V});
  // res.push_back({"iter"_nm = bc._iter});
  // res.push_back({"err"_nm = bc._err});
  return res;
}
