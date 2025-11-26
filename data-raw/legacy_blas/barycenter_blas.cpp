
#include <cpp11.hpp>

#include "barycenter_blas.hpp"

#include "ctrack.hpp"

using namespace wig;

[[cpp11::register]]
writable::list barycenter_parallel_blas(
    const doubles& A, const doubles& C,
    const doubles& w, double reg,
    bool withjac = false,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {

  unsigned int M = Rf_nrows(C);
  unsigned int N = Rf_ncols(C);
  unsigned int S = Rf_ncols(A);

  // init the class
  Barycenter bc(M, N, S,
                withjac, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C);
  bc.update_reg(reg);
  bc.update_A(A);
  bc.update_w(w);

  // start the computation
  bc.compute_parallel();

  // output list
  writable::list res;
  res.push_back({"b"_nm = bc.b});
  // if (withjac) {
  //   res.push_back({"JbA"_nm = bc._JbA});
  //   res.push_back({"Jbw"_nm = bc._Jbw});
  // }
  res.push_back({"U"_nm = bc._U});
  res.push_back({"V"_nm = bc._V});
  res.push_back({"iter"_nm = bc._iter});
  res.push_back({"err"_nm = bc._err});
  return res;
}
