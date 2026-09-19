
// definition file for the utility functions

#include <cmath>
#include <stdexcept>
#include <string>        // std::string
#include <unordered_map> // std::unordered_map
#include <vector>        // std::vector

#include "r_glue.hpp"

/////////////////////////////////////////////////////////////////////
// to be called in utils.R
/////////////////////////////////////////////////////////////////////

// Euclidean distance matrix between the rows of A (embeddings)
static SEXP euclidean_impl(SEXP AR) {
  la::Mat A = rr::mat_from_R(AR);
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
  return rr::to_R(euc);
}

extern "C" SEXP rwig_euclidean_cpp(SEXP AR) {
  return rr::call_guard([&]() -> SEXP { return euclidean_impl(AR); });
}

static SEXP doc2dist_impl(SEXP docs, SEXP dict) {
  // docs: list of character vectors
  // dict: character vector of the dictionary
  if (TYPEOF(docs) != VECSXP) throw std::runtime_error("docs must be a list");
  if (TYPEOF(dict) != STRSXP) throw std::runtime_error("dict must be a character vector");
  const int n_dict = (int)Rf_xlength(dict);
  const int n_docs = (int)Rf_xlength(docs);

  // token -> index lookup (first occurrence wins, as std::find did)
  std::unordered_map<std::string, int> lookup;
  lookup.reserve((std::size_t)n_dict);
  for (int k = 0; k < n_dict; ++k) {
    lookup.emplace(std::string(CHAR(STRING_ELT(dict, k))), k);
  }
  // tokens missing from the dictionary are counted under the last entry
  const int last = n_dict - 1;

  // create output matrix
  la::Mat docmat((la::idx)n_dict, (la::idx)n_docs);

  // loop the documents
  for (int j = 0; j < n_docs; ++j) {
    SEXP doc = VECTOR_ELT(docs, j);
    if (TYPEOF(doc) != STRSXP)
      throw std::runtime_error("each document must be a character vector");
    double *col = docmat.col(j);

    // loop the tokens inside doc
    const R_xlen_t n_tok = Rf_xlength(doc);
    for (R_xlen_t k = 0; k < n_tok; ++k) {
      auto it = lookup.find(std::string(CHAR(STRING_ELT(doc, k))));
      const int idx = (it == lookup.end()) ? last : it->second;
      col[idx] += 1;
    } // END of loop tokens

    // scale the matrix so that each col sum to 1
    double total = 0.;
    for (la::idx i = 0; i < docmat.nrow(); ++i) total += col[i];
    for (la::idx i = 0; i < docmat.nrow(); ++i) col[i] /= total;
  } // END of loop documents

  return rr::to_R(docmat);
}

extern "C" SEXP rwig_doc2dist_cpp(SEXP docs, SEXP dict) {
  return rr::call_guard([&]() -> SEXP { return doc2dist_impl(docs, dict); });
}
