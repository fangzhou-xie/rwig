
// implement the Wasserstein Barycenter algorithm and its gradients
// https://arxiv.org/abs/2504.08722
// Section 5 & 6

// #include <iostream> // std::cout
#include "R_ext/Print.h"    // for REprintf

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "utils_legacy.h" // utils plus armadillo

using namespace arma;
using namespace cpp11;
namespace writable = cpp11::writable;
using namespace cpp11::literals; // so we can use ""_nm syntax


// TODO: the barycenter algorithms need to be accessed from both R side
// (direct calling) and from c++ (for WDL)
// TODO: separate the conversion cpp11/arma conversion from the computation

// TODO: pass the references to modify values in-place
// hence no need for cpp11 list conversion between arma

// TODO: implement a wrapper function to only return b,JbA,Jbw
// with auto-switch methods




/////////////////////////////////////////////////////////////////////
// Section 5: Barycenter Algorithms
/////////////////////////////////////////////////////////////////////

// Algo 5.1 Parallel Barycenter
void barycenter_parallel_withoutjac_impl(
  mat& U, mat& V, vec& b, mat& K,
  mat& A, mat& C, vec& w, double reg,
  int& iter, double& err,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // tmp vars
  mat KT = K.t();
  mat KV = mat(M, S);
  mat KTU = mat(N, S);
  vec onesN = vec(N, fill::ones);
  vec onesM = vec(M, fill::ones);
  vec onesS = vec(S, fill::ones);
  mat onesNwT = onesN * w.t();
  mat B = mat(N, S);

  // tmp matrices
  mat ISkronK = kron(I(S), K);
  mat ISkronKT = kron(I(S), KT);
  mat onesSkronIN = kron(onesS, I(N));
  mat wTkronIN = kron(w.t(), I(N));

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    // verbose print
    if (verbose & (iter % 10 == 0)) {
      REprintf("iter: %.4i, err: %4.3f\n", iter, err);
    }

    KV = K * V;
    U = A / KV;

    KTU = KT * U;
    b = prod(pow(KTU, onesNwT), 1);
    B = b * onesS.t();
    V = B / KTU;

    iter++;
    err = norm((U % (K*V)) - A, 2);
  }
  // rescale b to sum to one
  b = b / accu(b);
}

// Algo 5.2 Log Barycenter
void barycenter_log_withoutjac_impl(
  mat& F, mat& G, vec& b,
  mat& A, mat& C, mat& w, double reg,
  int& iter, double& err,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // tmp vars
  vec logb = vec(N, fill::zeros);
  mat Rminrow = mat(M, S, fill::zeros);
  mat Rmincol = mat(N, S, fill::zeros);
  mat logA = log(A);
  vec onesS = vec(S, fill::ones);

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    // verbose print
    if (verbose & (iter % 10 == 0)) {
      REprintf("iter: %.4i, err: %4.3f\n", iter, err);
    }

    minrow(Rminrow, C, F, G, reg);
    F = F + reg * logA + Rminrow;

    mincol(Rmincol, C, F, G, reg);
    logb = - G * w / reg - Rmincol * w / reg;
    G = G + reg * logb * onesS.t() + Rmincol;

    iter++;
    minrow(Rminrow, C, F, G, reg);
    err = norm(-Rminrow / reg - logA, 2);
  }
  // scale output vec b to sum to one
  b = exp(logb);
  b = b / accu(b);
}

/////////////////////////////////////////////////////////////////////
// Section 6: Barycenter Algorithms with Jacobians
/////////////////////////////////////////////////////////////////////

// Algo 6.1 Parallel Barycenter
void barycenter_parallel_withjac_impl(
  mat& U, mat& JUA, mat& JUw,
  mat& V, mat& JVA, mat& JVw,
  vec& b, mat& JbA, mat& Jbw,
  mat& K,
  mat& A, mat& C, vec& w, double reg,
  int& iter, double& err,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  mat KT = K.t();
  mat KV = mat(M, S);
  mat KTU = mat(N, S);
  vec onesN = vec(N, fill::ones);
  vec onesM = vec(M, fill::ones);
  vec onesS = vec(S, fill::ones);
  mat onesNwT = onesN * w.t();
  mat B = mat(N, S);

  // tmp matrices
  mat ISkronK = kron(I(S), K);
  mat ISkronKT = kron(I(S), KT);
  mat onesSkronIN = kron(onesS, I(N));
  mat wTkronIN = kron(w.t(), I(N));

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    // verbose print
    if (verbose & (iter % 10 == 0)) {
      REprintf("iter: %.4i, err: %4.3f\n", iter, err);
    }

    KV = K * V;
    U = A / KV;
    JUA = diagmat(vectorise(1 / KV))
      - diagmat(vectorise(A / pow(KV,2))) * ISkronK * JVA;
    JUw = - diagmat(vectorise(A / pow(KV,2))) * ISkronK * JVw;

    KTU = KT * U;
    b = prod(pow(KTU, onesNwT), 1);
    JbA = diagmat(b) * wTkronIN * diagmat(vectorise(1 / KTU)) *
      ISkronKT * JUA;
    Jbw = diagmat(b) * wTkronIN * diagmat(vectorise(1 / KTU)) *
      ISkronKT * JUw + diagmat(b) * log(KTU);

    B = b * onesS.t();
    V = B / KTU;
    JVA = diagmat(vectorise(1 / KTU)) * onesSkronIN * JbA -
      diagmat(vectorise(B / pow(KTU, 2))) * ISkronKT * JUA;
    JVw = diagmat(vectorise(1 / KTU)) * onesSkronIN * Jbw -
      diagmat(vectorise(B / pow(KTU, 2))) * ISkronKT * JUw;

    iter++;
    err = norm((U % (K*V)) - A, 2);
  }
  // rescale b to sum to one
  b = b / accu(b);
}

// Algo 6.2 Log Barycenter
void barycenter_log_withjac_impl(
  mat& F, mat& JFA, mat& JFw,
  mat& G, mat& JGA, mat& JGw,
  vec& b, mat& JbA, mat& Jbw,
  sp_mat& W, sp_mat& V,
  mat& A, mat& C, vec& w, double reg,
  int& iter, double& err,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // tmp vars
  mat Rminrow = mat(M, S, fill::zeros);
  mat Rmincol = mat(N, S, fill::zeros);
  mat logA = log(A);
  vec onesS = vec(S, fill::ones);

  // logb
  vec logb = vec(N, fill::zeros);
  mat JlogbA = mat(N, M*S, fill::zeros);
  mat Jlogbw = mat(N, S, fill::zeros);

  // intermediate results
  mat epsdiag1vecA = reg * diagmat(1 / vectorise(A));
  mat epswTkronIN = kron(w.t(), I(N)) / reg;
  mat epsonesSkronIN = reg * kron(ones(S), I(N));

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    // verbose print
    if (verbose & (iter % 10 == 0)) {
      REprintf("iter: %.4i, err: %4.3f\n", iter, err);
    }

    minrowjac(W, Rminrow, C, F, G, reg);
    F = F + reg * logA + Rminrow;
    JFA = epsdiag1vecA - W * JGA;
    JFw = - W * JGw;

    mincoljac(V, Rmincol, C, F, G, reg);
    logb = - G * w / reg - Rmincol * w / reg;
    JlogbA = epswTkronIN * V * JFA;
    Jlogbw = epswTkronIN * V * JFw - (G + Rmincol) / reg;

    G = G + reg * logb * onesS.t() + Rmincol;
    JGA = epsonesSkronIN * JlogbA - V * JFA;
    JGw = epsonesSkronIN * Jlogbw - V * JFw;

    iter++;
    minrow(Rminrow, C, F, G, reg);
    err = norm(-Rminrow / reg - logA, 2);
  }

  // convert logb to b
  b = exp(logb);
  JbA = diagmat(b) * JlogbA;
  Jbw = diagmat(b) * Jlogbw;

  // scale output vec b to sum to one
  b = b / accu(b);
}

/////////////////////////////////////////////////////////////////////
// Functions for the header and WDL calling
/////////////////////////////////////////////////////////////////////

// TODO: only need b, JbA, Jbw for WDL
// TODO: but also need to switch mode automatically
void barycenter_wdl(
  vec& b, mat& JbA, mat& Jbw,
  mat& A, mat& C, vec& w, double reg,
  int sinkhorn_mode, double sinkhorn_mode_threshold,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // b, JbA, Jbw: reference output

  // if auto mode, decide on parallel or log
  if (sinkhorn_mode == 0) {
    // decide the mode: max or min
    double c1 = exp( - C.min() / reg);
    double c2 = exp( - C.max() / reg);
    c1 = c1 < c2 ? c1 : c2; // c1 is the minimum of K, to be compared with thresh
    if (c1 < sinkhorn_mode_threshold) {
      // log mode
      sinkhorn_mode = 2;
    } else {
      // parallel mode
      sinkhorn_mode = 1;
    }
  }

  // M,N,S: common dimensions
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;
  // iter and err
  int iter = 0;
  double err = 1000.;

  // start computation
  if (sinkhorn_mode == 1) {
    // parallel

    // init U,V
    mat U(M, S, fill::ones);
    mat V(N, S, fill::ones);
    // init JU,JV
    mat JUA = mat(M*S, M*S, fill::zeros);
    mat JVA = mat(N*S, M*S, fill::zeros);
    mat JbA = mat(N, M*S, fill::zeros);
    mat JUw = mat(M*S, S, fill::zeros);
    mat JVw = mat(N*S, S, fill::zeros);
    mat Jbw = mat(N, S, fill::zeros);
    // K
    mat K = exp(-(C/reg));

    // call the function
    barycenter_parallel_withjac_impl(
      U,JUA,JUw,V,JVA,JVw,b,JbA,Jbw,K,A,C,w,reg,iter,err,maxIter,zeroTol,verbose
    );
  } else if (sinkhorn_mode == 2) {
    // log
    // init F,G,logb
    mat F = mat(M, S, fill::zeros);
    mat G = mat(N, S, fill::zeros);
    // init Jacs of F,G,logb
    mat JFA = mat(M*S, M*S, fill::zeros);
    mat JGA = mat(N*S, M*S, fill::zeros);
    mat JbA = mat(N, M*S, fill::zeros);
    mat JFw = mat(M*S, S, fill::zeros);
    mat JGw = mat(N*S, S, fill::zeros);
    mat Jbw = mat(N, S, fill::zeros);
    // the sparse matrices W and V
    sp_mat W(M*S, N*S);
    sp_mat V(N*S, M*S);

    // call the function
    barycenter_log_withjac_impl(
      F,JFA,JFw,G,JGA,JGw,b,JbA,Jbw,W,V,A,C,w,reg,iter,err,maxIter,zeroTol,verbose
    );
  } else {
    cpp11::stop("sinkhorn_mode not 1 or 2!");
  }
  // b, JbA, Jbw modified in-place
}


/////////////////////////////////////////////////////////////////////
// Functions for the R interface for direct calling
/////////////////////////////////////////////////////////////////////

[[cpp11::register]]
writable::list barycenter_parallel_withoutjac_cpp(
  const doubles_matrix<>& AR, const doubles_matrix<>& CR,
  const doubles_matrix<>& wR, double reg,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // convert R matrices into arma ones
  mat A = as_Mat(AR);
  mat C = as_Mat(CR);
  vec w = as_Mat(wR);

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init U,V,b,K
  mat U(M, S, fill::ones);
  mat V(N, S, fill::ones);
  vec b(N, fill::zeros);
  mat K = exp(-(C/reg));
  int iter = 0;
  double err = 1000.;

  // call the function
  barycenter_parallel_withoutjac_impl(U,V,b,K,A,C,w,reg,iter,err,
                                      maxIter,zeroTol,verbose);

  // return output list
  writable::list res;
  res.push_back({"b"_nm = b});
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"U"_nm = as_doubles_matrix(U)});
  res.push_back({"V"_nm = as_doubles_matrix(V)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}

[[cpp11::register]]
writable::list barycenter_log_withoutjac_cpp(
  const doubles_matrix<>& AR, const doubles_matrix<>& CR,
  const doubles_matrix<>& wR, double reg,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // convert R matrices into arma ones
  mat A = as_Mat(AR);
  mat C = as_Mat(CR);
  vec w = as_Mat(wR);

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init F,G,logb
  mat F = mat(M, S, fill::zeros);
  mat G = mat(N, S, fill::zeros);
  vec b = vec(N, fill::zeros);
  int iter = 0;
  double err = 1000.;

  // call the function
  barycenter_log_withoutjac_impl(F,G,b,A,C,w,reg,iter,err,
                                 maxIter,zeroTol,verbose);

  // return output as list
  writable::list res;
  res.push_back({"b"_nm = b});
  res.push_back({"F"_nm = as_doubles_matrix(F)});
  res.push_back({"G"_nm = as_doubles_matrix(G)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}

[[cpp11::register]]
writable::list barycenter_parallel_withjac_cpp(
  const doubles_matrix<>& AR, const doubles_matrix<>& CR,
  const doubles_matrix<>& wR, double reg,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // convert R matrices into arma ones
  mat A = as_Mat(AR);
  mat C = as_Mat(CR);
  vec w = as_Mat(wR);

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init U,V
  mat U(M, S, fill::ones);
  mat V(N, S, fill::ones);
  // init JU,JV
  mat JUA = mat(M*S, M*S, fill::zeros);
  mat JVA = mat(N*S, M*S, fill::zeros);
  mat JbA = mat(N, M*S, fill::zeros);
  mat JUw = mat(M*S, S, fill::zeros);
  mat JVw = mat(N*S, S, fill::zeros);
  mat Jbw = mat(N, S, fill::zeros);
  // b
  vec b(N, fill::zeros);
  // K
  mat K = exp(-(C/reg));
  int iter = 0;
  double err = 1000.;

  // call the function
  barycenter_parallel_withjac_impl(
    U,JUA,JUw,V,JVA,JVw,b,JbA,Jbw,K,A,C,w,reg,iter,err,maxIter,zeroTol,verbose
  );

  // return output list
  writable::list res;
  res.push_back({"b"_nm = b});
  res.push_back({"JbA"_nm = as_doubles_matrix(JbA)});
  res.push_back({"Jbw"_nm = as_doubles_matrix(Jbw)});
  res.push_back({"U"_nm = as_doubles_matrix(U)});
  res.push_back({"JUA"_nm = as_doubles_matrix(JUA)});
  res.push_back({"JUw"_nm = as_doubles_matrix(JUw)});
  res.push_back({"V"_nm = as_doubles_matrix(V)});
  res.push_back({"JVA"_nm = as_doubles_matrix(JVA)});
  res.push_back({"JVw"_nm = as_doubles_matrix(JVw)});
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}

[[cpp11::register]]
writable::list barycenter_log_withjac_cpp(
  const doubles_matrix<>& AR, const doubles_matrix<>& CR,
  const doubles_matrix<>& wR, double reg,
  int maxIter = 1000, double zeroTol = 1e-6, bool verbose = false
) {
  // convert R matrices into arma ones
  mat A = as_Mat(AR);
  mat C = as_Mat(CR);
  vec w = as_Mat(wR);

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init F,G,logb
  mat F = mat(M, S, fill::zeros);
  mat G = mat(N, S, fill::zeros);
  vec b = vec(N, fill::zeros);
  // init Jacs of F,G,logb
  mat JFA = mat(M*S, M*S, fill::zeros);
  mat JGA = mat(N*S, M*S, fill::zeros);
  mat JbA = mat(N, M*S, fill::zeros);
  mat JFw = mat(M*S, S, fill::zeros);
  mat JGw = mat(N*S, S, fill::zeros);
  mat Jbw = mat(N, S, fill::zeros);
  // the sparse matrices W and V
  sp_mat W(M*S, N*S);
  sp_mat V(N*S, M*S);

  int iter = 0;
  double err = 1000.;

  // call the function
  barycenter_log_withjac_impl(
    F,JFA,JFw,G,JGA,JGw,b,JbA,Jbw,W,V,A,C,w,reg,iter,err,maxIter,zeroTol,verbose
  );

  writable::list res;
  res.push_back({"b"_nm = b});
  res.push_back({"JbA"_nm = as_doubles_matrix(JbA)});
  res.push_back({"Jbw"_nm = as_doubles_matrix(Jbw)});
  res.push_back({"F"_nm = as_doubles_matrix(F)});
  res.push_back({"JFA"_nm = as_doubles_matrix(JFA)});
  res.push_back({"JFw"_nm = as_doubles_matrix(JFw)});
  res.push_back({"G"_nm = as_doubles_matrix(G)});
  res.push_back({"JGA"_nm = as_doubles_matrix(JGA)});
  res.push_back({"JGw"_nm = as_doubles_matrix(JGw)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}
