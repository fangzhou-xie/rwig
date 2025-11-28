
// definition file for the utility functions

// #ifndef ARMA_DONT_USE_OPENMP
// #define ARMA_DONT_USE_OPENMP
// #endif

#include <vector> // std::vector
#include <string> // std::string

// #include <cpp11.hpp>
// #include <cpp11armadillo.hpp>

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]

// using namespace arma;

/////////////////////////////////////////////////////////////////////
// to be called in utils.R
/////////////////////////////////////////////////////////////////////

// Euclidean matrix based on embeddings
// [[Rcpp::export]]
arma::mat euclidean_cpp(const arma::mat& A) {
  // convert AR to A
  // mat A = as_Mat(AR);

  // create output matrix
  arma::mat euc = arma::mat(A.n_rows, A.n_rows, arma::fill::zeros);

  double c = 0.;
  for (size_t i = 0; i < A.n_rows; ++i) {
    for (size_t j = 0; j < A.n_rows; ++j) {
      if (i < j) {
        c = sqrt(accu(square(A.row(i) - A.row(j))));
        euc(i,j) = c;
        euc(j,i) = c;
      }
    }
  }
  // return as_doubles_matrix(euc);
  return euc;
}

// TODO: re-implement the `doc2dist` function directly using `cpp11::strings`
// [[Rcpp::export]]
arma::mat doc2dist_cpp(Rcpp::List docs, Rcpp::CharacterVector dict) {
  // docs: list of character vectors
  // dict: character vector of the dictionary

  // create output matrix
  arma::mat docmat(dict.size(), docs.size(), arma::fill::zeros);

  // loop the documents
  for (int j = 0; j < docs.size(); ++j) {

    // cpp11::strings docs_j = docs[j];
    std::vector<std::string> docs_j = docs[j];

    // loop the tokens inside doc
    for (long unsigned int k = 0; k < docs_j.size(); ++k) {
      // cpp11::r_string s = docs_j[k];
      std::string s = docs_j[k];
      auto it = std::find(dict.begin(), dict.end(), s);
      int idx = std::distance(dict.begin(), it);
      idx = (idx == dict.size()) ? dict.size() - 1 : idx;
      docmat(idx, j) += 1;
    } // END of loop tokens

    // scale the matrix so that each col sum to 1
    docmat.col(j) /= accu(docmat.col(j));
  } // END of loop documents

  // return as_doubles_matrix(docmat);
  return docmat;
}


// /////////////////////////////////////////////////////////////////////
// // Optimizers
// /////////////////////////////////////////////////////////////////////
//
// // optimizer: SGD
// void optimizer_sgd(mat &theta, mat &gtheta,
//                    mat &mtheta, mat &vtheta,
//                    int &t,
//                    const double eta = .001, const double gamma = .01,
//                    double beta1 = .9, double beta2 = .999,
//                    double eps = 1e-8) {
//   theta = theta - eta * gtheta;
//   t++;
// }
//
// // optimizer: Adam
// void optimizer_adam(mat &theta, mat &gtheta,
//                     mat &mtheta, mat &vtheta,
//                     int &t,
//                     const double eta = .001, const double gamma = .01,
//                     double beta1 = .9, double beta2 = .999,
//                     double eps = 1e-8) {
//   mtheta = beta1 * mtheta + (1 - beta1) * gtheta;
//   vtheta = beta2 * vtheta + (1 - beta2) * pow(gtheta, 2);
//   mat mhat = mtheta / (1 - pow(beta1, t));
//   mat vhat = vtheta / (1 - pow(beta2, t));
//   theta = theta - eta * (mhat / (sqrt(vhat) + eps));
//   t++;
// }
//
//
// // optimizer: AdamW
// void optimizer_adamw(mat &theta, mat &gtheta,
//                      mat &mtheta, mat &vtheta,
//                      int &t,
//                      const double eta = .001, const double gamma = .01,
//                      double beta1 = .9, double beta2 = .999,
//                      double eps = 1e-8) {
//   // gtheta = gtheta + lambda * theta;
//   mtheta = beta1 * mtheta + (1 - beta1) * gtheta;
//   vtheta = beta2 * vtheta + (1 - beta2) * pow(gtheta, 2);
//   mat mhat = mtheta / (1 - pow(beta1, t));
//   mat vhat = vtheta / (1 - pow(beta2, t));
//   // theta = theta - eta * (mhat / (sqrt(vhat) + eps)) - eta * gamma * theta;
//   theta = (1 - eta*gamma) * theta - eta * (mhat / (sqrt(vhat) + eps));
//   t++;
// }
