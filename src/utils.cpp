
// definition file for the utility functions


#include <vector> // std::vector
#include <string> // std::string
#include <unordered_map> // std::unordered_map

#include "common.hpp"

// using namespace arma;

/////////////////////////////////////////////////////////////////////
// to be called in utils.R
/////////////////////////////////////////////////////////////////////

// Euclidean matrix based on embeddings
// [[Rcpp::export]]
arma::mat euclidean_cpp(const arma::mat& A) {
  // Optimized vectorized implementation using the identity:
  // ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
  // This avoids nested loops and leverages BLAS operations

  size_t n = A.n_rows;

  // Compute squared norms for each row
  arma::vec sq_norms = arma::sum(arma::square(A), 1);

  // Compute distance matrix using broadcasting:
  // dist^2 = norms + norms' - 2*A*A'
  arma::mat euc = arma::repmat(sq_norms, 1, n) +
                  arma::repmat(sq_norms.t(), n, 1) -
                  2.0 * A * A.t();

  // Handle numerical errors (small negative values due to floating point)
  euc.transform([](double val) { return (val < 0.0) ? 0.0 : std::sqrt(val); });

  return euc;
}

// Optimized doc2dist using hash map for O(1) dictionary lookup
// [[Rcpp::export]]
arma::mat doc2dist_cpp(Rcpp::List docs, Rcpp::CharacterVector dict) {
  // docs: list of character vectors
  // dict: character vector of the dictionary

  // Build hash map for O(1) dictionary lookup
  std::unordered_map<std::string, int> dict_map;
  dict_map.reserve(dict.size());
  for (int i = 0; i < dict.size(); ++i) {
    dict_map[Rcpp::as<std::string>(dict[i])] = i;
  }

  // create output matrix
  arma::mat docmat(dict.size(), docs.size(), arma::fill::zeros);

  // loop the documents
  for (int j = 0; j < docs.size(); ++j) {
    std::vector<std::string> docs_j = docs[j];

    // loop the tokens inside doc
    for (size_t k = 0; k < docs_j.size(); ++k) {
      const std::string& s = docs_j[k];
      auto it = dict_map.find(s);

      // Only count tokens that are in the dictionary
      if (it != dict_map.end()) {
        docmat(it->second, j) += 1.0;
      }
    } // END of loop tokens

    // scale the matrix so that each col sum to 1
    double col_sum = arma::accu(docmat.col(j));
    if (col_sum > 0.0) {
      docmat.col(j) /= col_sum;
    }
  } // END of loop documents

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
