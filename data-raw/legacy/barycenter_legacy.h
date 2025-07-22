
// header file for the barycenter functions
// to be called in wdl algorithm

#ifndef WIG_BARYCENTER_LEGACY_H
#define WIG_BARYCENTER_LEGACY_H

// #include <cpp11.hpp>
#include <cpp11armadillo.hpp>

using namespace arma;

// barycenter algorithm wrapped for WDL with auto-switching
void barycenter_wdl(
    vec& b, mat& JbA, mat& Jbw,
    mat& A, mat& C, vec& w, double reg,
    int sinkhorn_mode, double sinkhorn_mode_threshold,
    int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
);

#endif // WIG_BARYCENTER_LEGACY_H
