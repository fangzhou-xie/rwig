
// this is the file defining the functions exporting to R side
// sinkhorn algos

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "sinkhorn_impl.hpp"
// #include "ctrack.hpp"

// using namespace arma;
// using namespace cpp11;
// using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable; // writable list from cpp11

//////////////////////////////////////////////////////////////////////
// Interfaces for the R side
//////////////////////////////////////////////////////////////////////


// [[Rcpp::export]]
Rcpp::List sinkhorn_vanilla_cpp(
    const arma::vec& a, const arma::vec& b,
    const arma::mat& C, double reg,
    bool withgrad = false,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {

  // init the class and start the computation
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_vanilla(a, b, C, reg);

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
      Rcpp::Named("P") = s.P,
      Rcpp::Named("grad_a") = s.grad_a,
      Rcpp::Named("u") = s.u,
      Rcpp::Named("v") = s.v,
      Rcpp::Named("loss") = s.loss,
      Rcpp::Named("iter") = s.iter,
      Rcpp::Named("err") = s.err,
      Rcpp::Named("return_status") = s.return_code
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("P") = s.P,
      // Rcpp::Named("grad_a") = s.grad_a,
      Rcpp::Named("u") = s.u,
      Rcpp::Named("v") = s.v,
      Rcpp::Named("loss") = s.loss,
      Rcpp::Named("iter") = s.iter,
      Rcpp::Named("err") = s.err,
      Rcpp::Named("return_status") = s.return_code
    );
  }
}


// [[Rcpp::export]]
Rcpp::List sinkhorn_log_cpp(
    const arma::vec& a, const arma::vec& b,
    const arma::mat& C, double reg,
    bool withgrad = false,
    const int& n_threads = 0,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {

  // init the class and start the computation
  Sinkhorn s(withgrad, maxiter, zerotol, verbose);
  s.compute_log(a, b, C, reg, n_threads);

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
      Rcpp::Named("P") = s.P,
      Rcpp::Named("grad_a") = s.grad_a,
      Rcpp::Named("f") = s.u,
      Rcpp::Named("g") = s.v,
      Rcpp::Named("loss") = s.loss,
      Rcpp::Named("iter") = s.iter,
      Rcpp::Named("err") = s.err,
      Rcpp::Named("return_status") = s.return_code
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("P") = s.P,
      // Rcpp::Named("grad_a") = s.grad_a,
      Rcpp::Named("f") = s.u,
      Rcpp::Named("g") = s.v,
      Rcpp::Named("loss") = s.loss,
      Rcpp::Named("iter") = s.iter,
      Rcpp::Named("err") = s.err,
      Rcpp::Named("return_status") = s.return_code
    );
  }
}




// [[cpp11::register]]
// writable::list sinkhorn_vanilla_cpp(
//     const doubles_matrix<>& a, const doubles_matrix<>& b,
//     const doubles_matrix<>& C, double reg,
//     bool withgrad = false, // bool reduce = false,
//     int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
// ) {
//   // convert the R vectors/matrices into arma ones
//   vec a_{as_Mat(a)};
//   vec b_{as_Mat(b)};
//   mat C_{as_Mat(C)};
//
//   Sinkhorn s(withgrad, maxiter, zerotol, verbose);
//   s.compute_vanilla(a_, b_, C_, reg);
//
//   // ctrack::result_print();
//
//   // output list
//   writable::list res;
//   res.push_back({"P"_nm = as_doubles_matrix(s.P)});
//   if (withgrad) {
//     res.push_back({"grad_a"_nm = s.grad_a});
//   }
//   // res.push_back({"grad_a"_nm = withgrad ? s.grad_a : R_NilValue});
//   res.push_back({"u"_nm = s.u});
//   res.push_back({"v"_nm = s.v});
//   res.push_back({"loss"_nm = s.loss});
//   res.push_back({"iter"_nm = s.iter});
//   res.push_back({"err"_nm = s.err});
//   res.push_back({"return_status"_nm = s.return_code});
//   return res;
// }
//
//
// [[cpp11::register]]
// writable::list sinkhorn_log_cpp(
//     const doubles_matrix<>& a, const doubles_matrix<>& b,
//     const doubles_matrix<>& C, double reg,
//     bool withgrad = false, // bool reduce = false,
//     int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
// ) {
//   // convert the R vectors/matrices into arma ones
//   vec a_{as_Mat(a)};
//   vec b_{as_Mat(b)};
//   mat C_{as_Mat(C)};
//
//   Sinkhorn s(withgrad, maxiter, zerotol, verbose);
//   s.compute_log(a_, b_, C_, reg);
//
//   // ctrack::result_print();
//
//   // output list
//   writable::list res;
//   res.push_back({"P"_nm = as_doubles_matrix(s.P)});
//   if (withgrad) {
//     res.push_back({"grad_a"_nm = s.grad_a});
//   }
//   // res.push_back({"grad_a"_nm = withgrad ? s.grad_a : R_NilValue});
//   res.push_back({"f"_nm = s.u});
//   res.push_back({"g"_nm = s.v});
//   res.push_back({"loss"_nm = s.loss});
//   res.push_back({"iter"_nm = s.iter});
//   res.push_back({"err"_nm = s.err});
//   res.push_back({"return_status"_nm = s.return_code});
//   return res;
// }
