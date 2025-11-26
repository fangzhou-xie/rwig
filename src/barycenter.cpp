
// this is the file defining the functions exporting to R side
// barycenter algos

#define ARMA_DONT_USE_OPENMP

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

#include "barycenter_impl.hpp"
#include "ctrack.hpp"

// using namespace arma;
// using namespace cpp11;
// using namespace cpp11::literals; // so we can use ""_nm syntax
// namespace writable = cpp11::writable; // writable list from cpp11

//////////////////////////////////////////////////////////////////////
// Interfaces for the R side
//////////////////////////////////////////////////////////////////////


// [[Rcpp::export]]
Rcpp::List barycenter_parallel_cpp(
    const arma::mat& A, const arma::mat& C,
    const arma::vec& w, double reg,
    const arma::vec& b_ext,
    bool withgrad = false,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {

  // init the class
  Barycenter bc(A.n_cols, withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C);
  bc.update_reg(reg);
  bc.update_A(A);
  bc.update_w(w);
  if (withgrad) {
    bc.update_b_ext(b_ext);
  }

  // start the computation
  bc.compute_parallel();

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
      Rcpp::Named("b") = bc.b,
      Rcpp::Named("grad_A") = bc.grad_A,
      Rcpp::Named("grad_w") = bc.grad_w,
      Rcpp::Named("loss") = bc.loss,
      Rcpp::Named("U") = bc.U,
      Rcpp::Named("V") = bc.V,
      Rcpp::Named("iter") = bc.iter,
      Rcpp::Named("err") = bc.err,
      Rcpp::Named("return_status") = bc.return_code
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("b") = bc.b,
      // Rcpp::Named("grad_A") = bc.grad_A,
      // Rcpp::Named("grad_w") = bc.grad_w,
      // Rcpp::Named("loss") = bc.loss,
      Rcpp::Named("U") = bc.U,
      Rcpp::Named("V") = bc.V,
      Rcpp::Named("iter") = bc.iter,
      Rcpp::Named("err") = bc.err,
      Rcpp::Named("return_status") = bc.return_code
    );
  }
}

// [[Rcpp::export]]
Rcpp::List barycenter_log_cpp(
    const arma::mat& A, const arma::mat& C,
    const arma::vec& w, double reg,
    const arma::vec& b_ext,
    bool withgrad = false,
    const int& n_threads = 0,
    int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
) {

  // init the class
  Barycenter bc(A.n_cols, withgrad, maxiter, zerotol, verbose);
  // update/load all the data
  bc.update_C(C);
  bc.update_reg(reg);
  bc.update_A(A);
  bc.update_w(w);
  if (withgrad) {
    bc.update_b_ext(b_ext);
  }

  // start the computation
  bc.compute_log(n_threads);

  // ctrack::result_print();

  if (withgrad) {
    return Rcpp::List::create(
      Rcpp::Named("b") = bc.b,
      Rcpp::Named("grad_A") = bc.grad_A,
      Rcpp::Named("grad_w") = bc.grad_w,
      Rcpp::Named("loss") = bc.loss,
      Rcpp::Named("F") = bc.U,
      Rcpp::Named("G") = bc.V,
      Rcpp::Named("iter") = bc.iter,
      Rcpp::Named("err") = bc.err,
      Rcpp::Named("return_status") = bc.return_code
    );
  } else {
    return Rcpp::List::create(
      Rcpp::Named("b") = bc.b,
      // Rcpp::Named("grad_A") = bc.grad_A,
      // Rcpp::Named("grad_w") = bc.grad_w,
      // Rcpp::Named("loss") = bc.loss,
      Rcpp::Named("F") = bc.U,
      Rcpp::Named("G") = bc.V,
      Rcpp::Named("iter") = bc.iter,
      Rcpp::Named("err") = bc.err,
      Rcpp::Named("return_status") = bc.return_code
    );
  }
}


// [[cpp11::register]]
// writable::list barycenter_parallel_cpp(
//     const doubles_matrix<>& A, const doubles_matrix<>& C,
//     const doubles_matrix<>& w, const doubles_matrix<>& b_ext, double reg,
//     bool withgrad = false,
//     int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
// ) {
//   // convert R matrices into arma ones
//   mat A_ = as_Mat(A);
//   mat C_ = as_Mat(C);
//   vec w_ = as_Mat(w);
//
//   // init the class
//   Barycenter bc(A_.n_cols, withgrad, maxiter, zerotol, verbose);
//   // update/load all the data
//   bc.update_C(C_);
//   bc.update_reg(reg);
//   bc.update_A(A_);
//   bc.update_w(w_);
//   if (withgrad) {
//     vec b_ext_ = as_Mat(b_ext); // convert the external data vector b only when needed
//     bc.update_b_ext(b_ext_);
//   }
//
//   // start the computation
//   bc.compute_parallel();
//
//   // ctrack::result_print();
//
//   // output list
//   writable::list res;
//   res.push_back({"b"_nm = bc.b});
//   if (withgrad) {
//     res.push_back({"grad_A"_nm = as_doubles_matrix(bc.grad_A)});
//     res.push_back({"grad_w"_nm = bc.grad_w});
//     res.push_back({"loss"_nm = bc.loss});
//   }
//   res.push_back({"U"_nm = as_doubles_matrix(bc.U)});
//   res.push_back({"V"_nm = as_doubles_matrix(bc.V)});
//   res.push_back({"iter"_nm = bc.iter});
//   res.push_back({"err"_nm = bc.err});
//   res.push_back({"return_status"_nm = bc.return_code});
//   return res;
// }
//
// [[cpp11::register]]
// writable::list barycenter_log_cpp(
//     const doubles_matrix<>& A, const doubles_matrix<>& C,
//     const doubles_matrix<>& w, const doubles_matrix<>& b_ext, double reg,
//     bool withgrad = false,
//     int maxiter = 1000, double zerotol = 1e-6, int verbose = 0
// ) {
//   // convert R matrices into arma ones
//   mat A_ = as_Mat(A);
//   mat C_ = as_Mat(C);
//   vec w_ = as_Mat(w);
//
//   // init the class
//   Barycenter bc(A_.n_cols, withgrad, maxiter, zerotol, verbose);
//   // update/load all the data
//   bc.update_C(C_);
//   bc.update_reg(reg);
//   bc.update_A(A_);
//   bc.update_w(w_);
//   if (withgrad) {
//     vec b_ext_ = as_Mat(b_ext); // convert the external data vector b only when needed
//     bc.update_b_ext(b_ext_);
//   }
//
//   // start the computation
//   bc.compute_log();
//
//   // ctrack::result_print();
//
//   // output list
//   writable::list res;
//   res.push_back({"b"_nm = bc.b});
//   if (withgrad) {
//     res.push_back({"grad_A"_nm = as_doubles_matrix(bc.grad_A)});
//     res.push_back({"grad_w"_nm = bc.grad_w});
//     res.push_back({"loss"_nm = bc.loss});
//   }
//   res.push_back({"F"_nm = as_doubles_matrix(bc.U)});
//   res.push_back({"G"_nm = as_doubles_matrix(bc.V)});
//   res.push_back({"iter"_nm = bc.iter});
//   res.push_back({"err"_nm = bc.err});
//   res.push_back({"return_status"_nm = bc.return_code});
//   return res;
// }
