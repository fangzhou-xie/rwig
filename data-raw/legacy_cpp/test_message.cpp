#include <string>

#include <Rcpp/Lightest>

// [[Rcpp::export]]
int test_msg() {
  std::string msg = "test message";
  Rcpp::message(Rf_mkString(msg.c_str()));

  Rcpp::message(Rf_mkString("test message"));
  return 0;
}
