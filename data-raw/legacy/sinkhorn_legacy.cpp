
// implement the Sinkhorn algorithm and its gradients
// https://arxiv.org/abs/2504.08722
// Section 3 & 4

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "utils_legacy.h" // utils plus armadillo

using namespace arma;
using namespace cpp11;
namespace writable = cpp11::writable;
using namespace cpp11::literals; // so we can use ""_nm syntax


/////////////////////////////////////////////////////////////////////
// Section 3: Sinkhorn Algorithms
/////////////////////////////////////////////////////////////////////

// Algo 3.1: Vanilla Sinkhorn
[[cpp11::register]]
writable::list sinkhorn_vanilla_withoutgrad_cpp(
    const doubles_matrix<>& aR, const doubles_matrix<>& bR,
    const doubles_matrix<>& CR, double reg,
    int maxIter = 1000, double zeroTol = 1e-6) {

  // convert the R vectors/matrices into arma ones
  vec a = as_Mat(aR);
  vec b = as_Mat(bR);
  mat C = as_Mat(CR);

  // a: source
  // b: target
  // C: cost matrix
  // reg: regularization parameter

  const int M = C.n_rows;
  const int N = C.n_cols;

  // scaling vectors
  vec u = ones(M);
  vec v = ones(N);

  mat K = exp(-C/reg);

  int iter = 0;
  double err = 1000.;

  // temp vec/mats
  vec Kv = zeros(N);
  vec KTu = zeros(M);

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    Kv = K*v;
    u = a / Kv;

    KTu = K.t() * u;
    v = b / KTu;

    ++iter;
    err = norm(u % (K*v) - a, 2) + norm(v % KTu - b, 2);
  }

  // optimal coupling matrix
  mat P = diagmat(u) * K * diagmat(v);
  // sinkhorn loss
  // double loss = accu(P % C) + reg * accu(P % (log(P) - 1));

  // https://cpp11.r-lib.org/articles/FAQ.html?q=named#how-do-i-add-elements-to-a-list
  // output: return the R list
  writable::list res;
  // res.push_back({"loss"_nm = loss});
  res.push_back({"P"_nm = as_doubles_matrix(P)});
  res.push_back({"u"_nm = u});
  res.push_back({"v"_nm = v});
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});
  return res;
}

// Algo 3.2: Parallel Sinkhorn
[[cpp11::register]]
writable::list sinkhorn_parallel_withoutjac_cpp(
  const doubles_matrix<>& AR, const doubles_matrix<>& BR,
  const doubles_matrix<>& CR, double reg,
  int maxIter = 1000, double zeroTol = 1e-6
) {
  // convert the R matrices into arma ones
  mat A = as_Mat(AR);
  mat B = as_Mat(BR);
  mat C = as_Mat(CR);

  // A: sources
  // B: targets
  // C: cost matrix
  // reg: regularization parameter

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init U,V
  mat U(size(A), fill::ones);
  mat V(size(B), fill::ones);

  // K
  mat K = exp(-C/reg);
  mat KT = K.t();

  // tmp mats
  mat KV = mat(M, S, fill::zeros);
  mat KTU = mat(N, S, fill::zeros);
  mat kronISK = kron(I(S), K);
  mat kronISKT = kron(I(S), KT);

  int iter = 0;
  double err = 1000.;

  while ((iter <= maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    KV = K * V;
    U = A / KV;

    KTU = KT * U;
    V = B / KTU;

    iter++;
    err = norm(U % (K*V) - A, 2) + norm(V % (K.t() * U) - B, 2);
  }

  // output cpp11 list
  writable::list res;
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"U"_nm = as_doubles_matrix(U)});
  res.push_back({"V"_nm = as_doubles_matrix(V)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});
  return res;
}

// Algo 3.3: Log Sinkhorn
[[cpp11::register]]
writable::list sinkhorn_log_withoutgrad_cpp(
    const doubles_matrix<>& aR, const doubles_matrix<>& bR,
    const doubles_matrix<>& CR, double reg,
    int maxIter = 1000, double zeroTol = 1e-6) {

  // convert the R vectors/matrices into arma ones
  vec a = as_Mat(aR);
  vec b = as_Mat(bR);
  mat C = as_Mat(CR);

  // a: source
  // b: target
  // C: cost matrix
  // reg: regularization parameter

  const int M = C.n_rows;
  const int N = C.n_cols;

  // init f,g
  vec f = vec(M, fill::zeros);
  vec g = vec(N, fill::zeros);

  // init tmp vec/mats
  mat R = mat(size(C), fill::zeros);
  vec Rminrow = vec(M, fill::zeros);
  vec Rmincol = vec(N, fill::zeros);
  vec loga = log(a);
  vec logb = log(b);
  vec onesM = vec(M, fill::ones);
  vec onesN = vec(N, fill::ones);

  int iter = 0;
  double err = 1000.;

  while ((iter <= maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    Rf(R, C, f, g);
    minrow(Rminrow, R, reg);
    f += reg * loga + Rminrow;

    Rf(R, C, f, g);
    mincol(Rmincol, R, reg);
    g += reg * logb + Rmincol;

    // terminate condition
    iter++;
    Rf(R, C, f, g);
    minrow(Rminrow, R, reg);
    mincol(Rmincol, R, reg);
    err = norm(-Rminrow/reg - loga, 2) + norm(-Rmincol/reg - logb, 2);
  }
  // optimal coupling
  mat P = exp(-R/reg);
  // sinkhorn loss
  // double loss = accu(P % C) + reg * accu(P % (log(P) - 1));

  // output list
  writable::list res;
  res.push_back({"P"_nm = as_doubles_matrix(P)});
  // res.push_back({"loss"_nm = loss});
  res.push_back({"f"_nm = as_doubles(f)});
  res.push_back({"g"_nm = as_doubles(g)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}



/////////////////////////////////////////////////////////////////////
// Section 4: Sinkhorn Algorithms with Gradients/Jacobians
/////////////////////////////////////////////////////////////////////

[[cpp11::register]]
writable::list sinkhorn_vanilla_withgrad_cpp(
    const doubles_matrix<>& aR, const doubles_matrix<>& bR,
    const doubles_matrix<>& CR, double reg,
    int maxIter = 1000, double zeroTol = 1e-6) {
  // a: source
  // b: target
  // C: cost matrix
  // reg: regularization parameter

  // convert the R vectors/matrices into arma ones
  vec a = as_Mat(aR);
  vec b = as_Mat(bR);
  mat C = as_Mat(CR);

  const int M = C.n_rows;
  const int N = C.n_cols;

  // scaling vectors
  vec u = ones(M);
  vec v = ones(N);
  // jacobians of u,v wrt a
  mat Ju = mat(M, M, fill::zeros);
  mat Jv = mat(N, M, fill::zeros);

  mat K = exp(-C/reg);

  int iter = 0;
  double err = 1000.;

  // temp vec/mats
  vec Kv = zeros(N);
  vec KTu = zeros(M);

  while ((iter < maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    Kv = K*v;
    u = a / Kv;
    Ju = diagmat(1 / Kv) - diagmat(a / pow(Kv, 2)) * K * Jv;

    KTu = K.t() * u;
    v = b / KTu;
    Jv = - diagmat(b / pow(KTu, 2)) * K.t() * Ju;

    ++iter;
    err = norm(u % (K*v) - a, 2) + norm(v % KTu - b, 2);
  }

  // optimal coupling matrix
  mat P = diagmat(u) * K * diagmat(v);
  // sinkhorn loss
  // double loss = accu(P % C) + reg * accu(P % (log(P) - 1));

  // compute gradient of loss w.r.t. a
  vec grad_a = (vectorise(C + reg * log(P)).t() * diagmat(vectorise(P)) * (
    kron(ones(N), I(M)) * diagmat(1 / u) * Ju +
      kron(I(N), ones(M)) * diagmat(1 / v) * Jv
  )).t();

  // https://cpp11.r-lib.org/articles/FAQ.html?q=named#how-do-i-add-elements-to-a-list
  // output: return the R list
  writable::list res;
  // res.push_back({"loss"_nm = loss});
  res.push_back({"P"_nm = as_doubles_matrix(P)});
  res.push_back({"grad_a"_nm = grad_a});
  res.push_back({"u"_nm = u});
  res.push_back({"Ju"_nm = as_doubles_matrix(Ju)});
  res.push_back({"v"_nm = v});
  res.push_back({"Jv"_nm = as_doubles_matrix(Jv)});
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});
  return res;
}


// Algo 4.2: Parallel Sinkhorn with Jacobian
[[cpp11::register]]
writable::list sinkhorn_parallel_withjac_cpp(
    const doubles_matrix<>& AR, const doubles_matrix<>& BR,
    const doubles_matrix<>& CR, double reg,
    int maxIter = 1000, double zeroTol = 1e-6
) {
  // convert the R matrices into arma ones
  mat A = as_Mat(AR);
  mat B = as_Mat(BR);
  mat C = as_Mat(CR);

  // A: sources
  // B: targets
  // C: cost matrix
  // reg: regularization parameter

  // M,N,S
  const int M = C.n_rows;
  const int N = C.n_cols;
  const int S = A.n_cols;

  // init U,V
  mat U(size(A), fill::ones);
  mat V(size(B), fill::ones);
  // Jac of U,V
  mat JU(M*S, M*S, fill::zeros);
  mat JV(N*S, M*S, fill::zeros);

  // K
  mat K = exp(-C/reg);
  mat KT = K.t();

  // tmp mats
  mat KV = mat(M, S, fill::zeros);
  mat KTU = mat(N, S, fill::zeros);
  mat kronISK = kron(I(S), K);
  mat kronISKT = kron(I(S), KT);

  int iter = 0;
  double err = 1000.;

  while ((iter <= maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    KV = K * V;
    U = A / KV;
    JU = diagmat(vectorise(1 / KV)) - diagmat(vectorise(A / pow(KV, 2))
    ) * kronISK * JV;

    KTU = KT * U;
    V = B / KTU;
    JV = -diagmat(vectorise(B / pow(KTU,2))) *  kronISKT * JU;

    iter++;
    err = norm(U % (K*V) - A, 2) + norm(V % (K.t() * U) - B, 2);
  }

  // output cpp11 list
  writable::list res;
  res.push_back({"K"_nm = as_doubles_matrix(K)});
  res.push_back({"U"_nm = as_doubles_matrix(U)});
  res.push_back({"JU"_nm = as_doubles_matrix(JU)});
  res.push_back({"V"_nm = as_doubles_matrix(V)});
  res.push_back({"JV"_nm = as_doubles_matrix(JV)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});
  return res;
}


// Algo 4.3: Log Sinkhorn with gradient
[[cpp11::register]]
writable::list sinkhorn_log_withgrad_cpp(
    const doubles_matrix<>& aR, const doubles_matrix<>& bR,
    const doubles_matrix<>& CR, double reg,
    int maxIter = 1000, double zeroTol = 1e-6) {

  // convert the R vectors/matrices into arma ones
  vec a = as_Mat(aR);
  vec b = as_Mat(bR);
  mat C = as_Mat(CR);

  // a: source
  // b: target
  // C: cost matrix
  // reg: regularization parameter

  const int M = C.n_rows;
  const int N = C.n_cols;

  // init f,g
  vec f = vec(M, fill::zeros);
  vec g = vec(N, fill::zeros);
  // Jacobianss of f,g
  mat Jf = mat(M, M, fill::zeros);
  mat Jg = mat(N, M, fill::zeros);
  // weight matrices W and V
  mat W = mat(M, N, fill::zeros);
  mat V = mat(N, M, fill::zeros);

  // init tmp vec/mats
  mat R = mat(size(C), fill::zeros);
  vec Rminrow = vec(M, fill::zeros);
  vec Rmincol = vec(N, fill::zeros);
  vec loga = log(a);
  vec logb = log(b);
  vec onesM = vec(M, fill::ones);
  vec onesN = vec(N, fill::ones);
  sp_mat onesN_IM;
  onesN_IM = kron(onesN, I(M));
  sp_mat IN_onesM;
  IN_onesM = kron(I(N), onesM);

  int iter = 0;
  double err = 1000.;

  while ((iter <= maxIter) & (err >= zeroTol)) {
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    Rf(R, C, f, g);
    minrowjac(W, Rminrow, R, reg);
    f += reg * loga + Rminrow;
    Jf = reg * diagmat(1/a) - W * Jg;

    Rf(R, C, f, g);
    mincoljac(V, Rmincol, R, reg);
    g += reg * logb + Rmincol;
    Jg = - V * Jf;

    // terminate condition
    iter++;
    Rf(R, C, f, g);
    minrow(Rminrow, R, reg);
    mincol(Rmincol, R, reg);
    err = norm(-Rminrow/reg - loga, 2) + norm(-Rmincol/reg - logb, 2);
  }
  // optimal coupling
  mat P = exp(-R/reg);
  // sinkhorn loss
  // double loss = accu(P % C) + reg * accu(P % (log(P) - 1));

  // grad wrt a
  vec grad_a = (vectorise(C + reg * log(P)).t() * diagmat(vectorise(P)) * (
    onesN_IM * Jf + IN_onesM * Jg
  )).t() / reg;

  // output list
  writable::list res;
  res.push_back({"P"_nm = as_doubles_matrix(P)});
  res.push_back({"grad_a"_nm = as_doubles(grad_a)});
  // res.push_back({"loss"_nm = loss});
  res.push_back({"f"_nm = as_doubles(f)});
  res.push_back({"Jf"_nm = as_doubles_matrix(Jf)});
  res.push_back({"g"_nm = as_doubles(g)});
  res.push_back({"Jg"_nm = as_doubles_matrix(Jg)});
  res.push_back({"iter"_nm = iter});
  res.push_back({"err"_nm = err});

  return res;
}
