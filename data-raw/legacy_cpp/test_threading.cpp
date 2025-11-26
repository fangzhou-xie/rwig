
// test the threading

#include <thread> // std::thread
#include <vector> // std::vector

#include <RcppArmadillo/Lightest>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// auto worker = [&results](int start, int end) {
//   for (int i = start; i < end; ++i) {
//     results[i] =
//   }
// }

// NOTE: threading is fast!!

// [[Rcpp::export]]
arma::vec test_threading(const arma::mat& X, const arma::vec& y) {
  // mat X_ = as_Mat(X);

  std::vector<std::thread> threads;
  std::vector<double> results(X.n_rows);

  int chunk_size = results.size() / 10;

  for (int t = 0; t < 10; ++t) {
    int start = t * chunk_size;
    int end = (t == 9) ? results.size() : (t + 1) * chunk_size;

    threads.emplace_back([&results, &X, &y, start, end]() {
      for (int i = start; i < end; ++i) {
        results[i] = accu(X.row(i) * y);
      }
    });
  }

  for (auto& t: threads) {
    t.join();
  }

  // arma::mat out{mat(arma::size(X), fill::zeros)};
  //
  // for (uword i = 0; i < X.n_cols; ++i) {
  //   out.col(i) = results[i];
  // }
  vec out = conv_to<vec>::from(results);
  return out;
}

// [[Rcpp::export]]
arma::vec test_threading2(const arma::mat& X, const arma::vec& y,
                         const int& n_t) {
  std::vector<std::thread> threads;
  arma::vec results(X.n_rows);
  int chunk_size = X.n_rows / n_t;

  for (int t = 0; t < n_t; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_t - 1)) ? X.n_rows : (t + 1) * chunk_size;

    threads.emplace_back([&results, &X, &y, start, end]() {
      for (int i = start; i < end; ++i) {
        results.row(i) = accu(X.row(i) * y);
      }
    });
  }

  for (auto& t : threads) { t.join(); }
  return results;
}


// [[Rcpp::export]]
arma::vec test_serial(const arma::mat& X, const arma::vec& y) {

  arma::vec out{arma::vec(X.n_rows, fill::zeros)};
  for (uword i = 0; i < X.n_rows; ++i) {
    out(i) = arma::accu(X.row(i) * y);
  }
  return out;
}
