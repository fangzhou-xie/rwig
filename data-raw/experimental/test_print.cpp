
#include <iostream>
#include <iomanip>

#include <cpp11.hpp>

using namespace cpp11;

// void check_mat(const doubles& x, const char * msg = "x") {
//   if (Rf_ncols(x) == 1) {
//     Rf_error("%s is not a matrix", msg);
//   }
// }

double at(const doubles& x, int i, int j) {
  // check_mat(x, "x");
  return x[i + j * Rf_nrows(x)]; // column-major
}

// std::ostream& operator<<(std::ostream& os, const doubles& x) {
//   for (int i{0}; i < Rf_nrows(x); ++i) {
//     os << "  ";
//     for (int j{0}; j < Rf_ncols(x); ++j) {
//       os << std::setw(6) << std::setprecision(3) << at(x, i, j) << "  ";
//     }
//     os << std::endl;
//   }
//   return os;
// }

void print_mat(const doubles& x) {
  for (int i{0}; i < Rf_nrows(x); ++i) {
    std::cout << " ";
    for (int j{0}; j < Rf_ncols(x); ++j) {
      std::cout << std::setw(6) << std::setprecision(3) <<
        at(x, i, j) << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n" << std::endl;
}


[[cpp11::register]]
int testprint(const doubles& x) {
  print_mat(x);
  return 0;
}
