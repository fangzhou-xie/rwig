
// test loading std::vector and array to cpp11::doubles

#include <vector>

#include <cpp11/doubles.hpp>

using namespace cpp11;

[[cpp11::register]]
// doubles f1() {
//   std::vector<double> s(10);
//   doubles r(s.begin(), s.end());
//   return r;
// }

// [[cpp11::register]]
// doubles f2() {
//   double s[10];
//   size_t size = sizeof(s) / sizeof(s);
//   doubles r(s, s + size);
//   return r;
// }
