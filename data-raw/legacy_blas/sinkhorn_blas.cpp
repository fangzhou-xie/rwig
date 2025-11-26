
#include <cpp11.hpp>

#include "sinkhorn_blas.hpp"

#include "ctrack.hpp"

using namespace wig;

[[cpp11::register]]
writable::list sinkhorn_vanilla_blas(
  const doubles& a, const doubles& b, const doubles& C, double reg,
  bool withgrad = false, // bool reduce = false,
  int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_vanilla(a, b, C, reg);

  if (verbose) { ctrack::result_print(); }

  // output list
  writable::list res;
  res.push_back({"P"_nm = s._P});
  if (withgrad) {
    res.push_back({"grad_a"_nm = s._grad_a});
  }
  res.push_back({"u"_nm = s._u});
  res.push_back({"v"_nm = s._v});
  res.push_back({"iter"_nm = s._iter});
  res.push_back({"err"_nm = s._err});
  return res;
}
