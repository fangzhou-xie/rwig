
// definition file for the utility functions

#include <cmath>
#include <string>        // std::string
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include "rcpp_glue.hpp"

/////////////////////////////////////////////////////////////////////
// to be called in utils.R
/////////////////////////////////////////////////////////////////////

// Euclidean distance matrix between the rows of A (embeddings)
// [[Rcpp::export]]
Rcpp::NumericMatrix euclidean_cpp(const SEXP &AR) {
  la::Mat A = la::mat_from_R(AR);
  const la::idx n = A.nrow(), d = A.ncol();

  // transpose once so that each embedding is contiguous (d x n)
  la::Mat At(d, n);
  for (la::idx j = 0; j < d; ++j)
    for (la::idx i = 0; i < n; ++i) At(j, i) = A(i, j);

  la::Mat euc(n, n);
  for (la::idx i = 0; i < n; ++i) {
    const double *ai = At.col(i);
    for (la::idx j = i + 1; j < n; ++j) {
      const double *aj = At.col(j);
      double c = 0.;
      for (la::idx k = 0; k < d; ++k) {
        const double diff = ai[k] - aj[k];
        c += diff * diff;
      }
      c = std::sqrt(c);
      euc(i, j) = c;
      euc(j, i) = c;
    }
  }
  return la::to_R(euc);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix doc2dist_cpp(Rcpp::List docs, Rcpp::CharacterVector dict) {
  // docs: list of character vectors
  // dict: character vector of the dictionary

  // token -> index lookup (first occurrence wins, as std::find did)
  std::unordered_map<std::string, int> lookup;
  lookup.reserve((std::size_t)dict.size());
  for (int k = 0; k < dict.size(); ++k) {
    lookup.emplace(std::string(dict[k]), k);
  }
  // tokens missing from the dictionary are counted under the last entry
  const int last = dict.size() - 1;

  // create output matrix
  la::Mat docmat((la::idx)dict.size(), (la::idx)docs.size());

  // loop the documents
  for (int j = 0; j < docs.size(); ++j) {
    std::vector<std::string> docs_j = docs[j];
    double *col = docmat.col(j);

    // loop the tokens inside doc
    for (const std::string &s : docs_j) {
      auto it = lookup.find(s);
      const int idx = (it == lookup.end()) ? last : it->second;
      col[idx] += 1;
    } // END of loop tokens

    // scale the matrix so that each col sum to 1
    double total = 0.;
    for (la::idx i = 0; i < docmat.nrow(); ++i) total += col[i];
    for (la::idx i = 0; i < docmat.nrow(); ++i) col[i] /= total;
  } // END of loop documents

  return la::to_R(docmat);
}
