
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

// // [[Rcpp::export]]
// arma::mat f1(arma::vec& a, arma::mat& C, double reg) {
//   arma::vec u = arma::vec(C.n_rows, arma::fill::ones);
//   arma::vec v = arma::vec(C.n_cols, arma::fill::ones);
//   arma::mat Ju = arma::mat(C.n_rows, C.n_rows, arma::fill::zeros);
//   arma::mat Jv = arma::mat(C.n_cols, C.n_rows, arma::fill::zeros);
//   arma::mat K = exp(-C/reg);
//   arma::vec Kv = K*v;
//   u = a / Kv;
//   Ju = diagmat(1 / Kv) - diagmat(a / pow(Kv, 2)) * K * Jv;
//   double aa = a(0);
//   return Ju;
// }
//
// // [[Rcpp::export]]
// arma::mat f2(arma::vec& a, arma::mat& C, double reg) {
//   arma::vec u = arma::vec(C.n_rows, arma::fill::ones);
//   arma::vec v = arma::vec(C.n_cols, arma::fill::ones);
//   arma::mat Ju = arma::mat(C.n_rows, C.n_rows, arma::fill::zeros);
//   arma::mat Jv = arma::mat(C.n_cols, C.n_rows, arma::fill::zeros);
//   arma::mat K = exp(-C/reg);
//   arma::vec Kv = K*v;
//   u = a / Kv;
//   // Ju = diagmat(1 / (K * v)) - diagmat(a / pow(K*v, 2)) * K * Jv;
//   for (size_t i = 0; i < C.n_rows; ++i) {
//     for (size_t m = 0; m < C.n_rows; ++m) {
//       if (i == m) {
//         Ju(i,m) = 1 / Kv(i)
//           - a(i) * accu(K.row(i).t() % Jv.col(m)) / (Kv(i) * Kv(i));
//       } else {
//         Ju(i,m) = - a(i) * accu(K.row(i).t() % Jv.col(m)) / (Kv(i) * Kv(i));
//       }
//     }
//   }
//   return Ju;
// }

arma::vec testfind(arma::vec& a) {
  return a.elem(arma::find(a == 0))
}
