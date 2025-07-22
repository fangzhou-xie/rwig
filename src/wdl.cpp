
// implement the WDL algorithm
// https://arxiv.org/abs/2504.08722
// Section 7.2

// #include <iostream> // std::cout
#include "R_ext/Print.h"    // for REprintf

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

// #include "utils.h"
#include "wdl.h"     // actual WDL implementation

using namespace arma;
using namespace cpp11;
namespace writable = cpp11::writable;
using namespace cpp11::literals; // so we can use ""_nm syntax


/////////////////////////////////////////////////////////////////////
// R interfaces for WDL
/////////////////////////////////////////////////////////////////////

[[cpp11::register]]
writable::list wdl_cpp(
    const doubles_matrix<>& YR, const doubles_matrix<>& CR,
    const double reg,
    const int S,
    const int batch_size,
    const int epochs,
    int sinkhorn_mode = 2,
    // const double sinkhorn_mode_threshold = .1,
    const int max_iter = 1000, const double zero_tol = 1e-6,
    const int optimizer = 2,
    const double eta = .001, const double gamma = .01,
    const double beta1 = .9, const double beta2 = .999,
    const double eps = 1e-8,
    const int rng_seed = 123, const bool verbose = false
) {
  // convert R matrices into arma ones
  mat Y = as_Mat(YR);
  mat C = as_Mat(CR);

  // check sinkhorn mode
  if ((sinkhorn_mode != 1) && (sinkhorn_mode != 2)) {
    cpp11::stop("Sinkhorn mode not supported");
  }

  // check optimizer mode
  if ((optimizer != 0) && (optimizer != 1) && (optimizer != 2)) {
    cpp11::stop("optimizer must be: 0, 1, 2!");
  }

  // init the WDL class
  WassersteinDictionaryLearning wdl(
    batch_size,epochs,sinkhorn_mode,max_iter,zero_tol,
    optimizer,eta,gamma,beta1,beta2,eps,rng_seed,verbose
  );
  // start the actual WDL computation
  wdl.compute(Y, C, reg, S);

  // output the optimized topics A and weights W
  writable::list res;
  res.push_back({"A"_nm = as_doubles_matrix(wdl.A)});
  res.push_back({"W"_nm = as_doubles_matrix(wdl.W)});
  res.push_back({"Yhat"_nm = as_doubles_matrix(wdl.Yhat)});
  return res;
}
