

#ifndef BLAS_UTILS_H
#define BLAS_UTILS_H

#include <iostream>
#include <iomanip>

#include <cmath>    // exp, log, etc

#include <cpp11/integers.hpp>      // integers, aka r_vecotor<int>
#include <cpp11/doubles.hpp>       // doubles, aka r_vecotor<double>

#include "ctrack.hpp"
#include "fastexp.h"

using namespace cpp11;

using uint = unsigned int;

namespace blas {


/////////////////////////////////////////////////////////////////
// Utilities
/////////////////////////////////////////////////////////////////



template<typename T>
inline void print_vec(const r_vector<T>& x, const char * msg = "") {
  std::cout << msg << "\n ";
  for (int i{0}; i < x.size(); ++i) {
    std::cout << std::setw(6) << std::setprecision(3) << x[i] << "  ";
  }
  std::cout << "\n" << std::endl;
}

template<typename T>
inline void print_mat(const r_vector<T>& x, const char * msg = "") {
  std::cout << msg << std::endl;
  for (int i{0}; i < Rf_nrows(x); ++i) {
    std::cout << " ";
    for (int j{0}; j < Rf_ncols(x); ++j) {
      std::cout << std::setw(6) << std::setprecision(3) <<
        // at(x, i, j) << "  ";
        x[i + j * Rf_nrows(x)] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "\n" << std::endl;
}

// TODO: how to index the matrix/vector?
// will be used in the isSymmetric as well

// int nrow(const doubles& a);
// int ncol(const doubles& a);
// double at(const doubles& A, int i, int j);
// doubles t(const doubles& A);
// bool isSymmetric(const doubles& a);

// dims
inline int nrow(const doubles& x) { return Rf_nrows(x); }
inline int ncol(const doubles& x) { return Rf_ncols(x); }

// TODO: function to set/reset dim
inline void set_dim(writable::doubles& x, int m, int n) {CTRACK;
  if (n > 1) {
    x.attr(R_DimSymbol) = {m, n};
  }
}

inline void set_dim(writable::doubles& x, const doubles& a) {CTRACK;
  CTRACK;
  if (Rf_ncols(a) > 1) {
    x.attr(R_DimSymbol) = {Rf_nrows(a), ncol(a)};
  }
}

// TODO: also some errors on dimensions
inline void check_vec(const doubles& x, const char * msg = "x") {CTRACK;
  if (Rf_ncols(x) > 1)  {
    Rf_error("%s is not a vector", msg);
  }
}

inline void check_mat(const doubles& x, const char * msg = "x") {CTRACK;
  if (Rf_ncols(x) == 1) {
    Rf_error("%s is not a matrix", msg);
  }
}

inline void check_dim(const doubles& x, int m, int n, const char* msg) {CTRACK;
  int M = Rf_nrows(x);
  int N = Rf_ncols(x);
  if ((M != m) || (N != n)) {
    Rf_error(
      "%s is expected to be %d*%d, but receive %d*%d",
      msg, m, n, M, N
    );
  }
}

inline void check_dim(const doubles& x, const doubles& y) {CTRACK;
  int xr = Rf_nrows(x);
  int xc = Rf_ncols(x);
  int yr = Rf_nrows(y);
  int yc = Rf_ncols(y);
  if ((xr != yr) || (xc != yc)) {
    Rf_error("x (%d*%d) and y (%d*%d) are not same size", xr, xc, yr, yc);
  }
}

// init vec and mat
inline doubles init_mat(int m, int n, double fill) {CTRACK;
  int mn = m*n;
  writable::doubles r(mn);
  double * r_ = REAL(r.data());
  for (int i{0}; i < mn; ++i) {
    r_[i] = fill;
  }
  set_dim(r, m, n);
  return r;
}

inline doubles init_vec(int n, double fill) {CTRACK;
  writable::doubles r(n); // default to be column vector
  double * r_ = REAL(r.data());
  for (int i{0}; i < r.size(); ++i) {
    r_[i] = fill;
  }
  return r;
}

// init vecs of zero and one
inline doubles ones(int m, int n) { CTRACK;return init_mat(m, n, 1.); }
inline doubles zeros(int m, int n) { CTRACK;return init_mat(m, n, 0.); }
inline doubles ones(int n) { CTRACK;return init_vec(n, 1.); }
inline doubles zeros(int n) { CTRACK;return init_vec(n, 0.); }

// vec indexing
inline double at(const doubles& x, int i) {CTRACK;
  // check_vec(x, "at: x");
  double * x_ = REAL(x.data());
  return x_[i];
}

// mat indexing
inline double at(const doubles& x, int i, int j) {CTRACK;
  // check_mat(x, "at: x");
  double * x_ = REAL(x.data());
  return x_[i + j * Rf_nrows(x)]; // column-major
}

// set value at index
inline void set(writable::doubles& x, double a, int i) {CTRACK;
  // check_vec(x, "set: x");
  double * x_ = REAL(x.data());
  x_[i] = a;
}

inline void set(writable::doubles& x, const doubles& y,
                const integers& indx) {CTRACK;
  // check_vec(x, "set: x");
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());
  for (int i{0}; i < indx.size(); ++i) {
    x_[indx[i]] = y[i];
  }
}

inline void set(writable::doubles& x, double a, int i, int j) {CTRACK;
  // check_mat(x, "set: x");
  double * x_ = REAL(x.data());
  x_[i + j * Rf_nrows(x)] = a;
}

inline void set(writable::doubles& x, const doubles& y,
                const integers& indx, const integers& indy) {CTRACK;
  int Mx = Rf_nrows(x);
  int My = Rf_nrows(y);
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());
  for (int j{0}; j < indy.size(); ++j) {
    for (int i{0}; i < indx.size(); ++i) {
      x_[indx[i] + indy[j] * Mx] = y_[i + j * My];
    }
  }
}

// reset to 0
inline void reset(writable::doubles& x, double fill) {
  CTRACK;
  double * x_ = REAL(x.data());
  for (int i{0}; i < x.size(); ++i) {
    x_[i] = fill;
  }
}

// is Symmetric matrix?
inline bool is_symmetric(const doubles& x) {CTRACK;
  int M = Rf_nrows(x);
  int N = Rf_ncols(x);
  double * x_ = REAL(x.data());
  if (M != N) { return false; }
  for (int j{0}; j < N; ++j) {
    for (int i{0}; i < M; ++i) {
      if ((i != j) && (
        // at(x, i, j) != at(x, j, i)
        x_[i + j * M] != x_[j + i * M]
      )) { return false; }
    }
  }
  return true;
}

// bool masking like arma
inline integers find_value(const doubles& x, double v, bool equal) {CTRACK;
  // equal: bool, true -> find the value, false -> find NOT the value
  writable::integers ind;
  for (int i = 0; i < x.size(); ++i) {
    if (equal) {
      if (x[i] == v) { ind.push_back(i); }
    } else {
      if (x[i] != v) { ind.push_back(i); }
    }
  }
  return ind;
}

// subset vec
inline doubles select(const doubles& x, const integers& ind) {
  CTRACK;
  // check_vec(x, "select: x");
  writable::doubles out(ind.size());
  double * out_ = REAL(out.data());
  double * x_ = REAL(x.data());
  for (int i = 0; i < ind.size(); ++i) {
    out_[i] = x_[ind[i]];
  }
  return out;
}

// subset mat
inline doubles select(const doubles& x,
                      const integers& indx,
                      const integers& indy) {
  CTRACK;
  // indx: indices on the rows
  // indy: indices on the cols
  // check_mat(x, "select: x");
  int M = indx.size();
  int N = indy.size();
  writable::doubles out(M * N);
  set_dim(out, M, N);
  double * out_ = REAL(out.data());
  double * x_ = REAL(x.data());
  int Mx = Rf_nrows(x);
  for (int j{0}; j < N; ++j) {
    for (int i{0}; i < M; ++i) {
      // out[i + j * M] = at(x, i, j);
      out_[i + j * M] = x_[indx[i] + indy[j] * Mx];
    }
  }
  return out;
}

// diagonal
inline doubles diag(const doubles& x) {CTRACK;
  // check_vec(x, "diag: x");
  int n = x.size();
  writable::doubles out(n * n);
  set_dim(out, n, n);
  double * out_ = REAL(out.data());
  for (int j{0}; j < n; ++j) {
    for (int i{0}; i < n; ++i) {
      out_[i + j * n] = (i == j) ? x[i] : 0.;
    }
  }
  return out;
}

/////////////////////////////////////////////////////////////////
// WIG utils
/////////////////////////////////////////////////////////////////

inline doubles C2K(const doubles& C, const double reg) {
  CTRACK;
  int n = C.size();
  writable::doubles K(n);
  double * K_ = REAL(K);
  double * C_ = REAL(C);
  for (int i = 0; i < n; ++i) {
    // K_[i] = std::exp(- C_[i] / reg);
    K_[i] = fastExp(- C_[i] / reg);
  }
  set_dim(K, C);
  // print_mat(K);
  // print_vec(K);
  // stop("stop here");
  return K;
}

// y = a / x
inline void over(writable::doubles& y, const doubles& a, const doubles& x) {
  CTRACK;
  double * y_ = REAL(y.data());
  double * a_ = REAL(a.data());
  double * x_ = REAL(x.data());
  for (int i = 0; i < y.size(); ++i) {
    y_[i] = a_[i] / x_[i];
  }
}

// norm(_u % (_K*_v) - _a, 2)
inline double norm(const doubles& u, const doubles& kv, const doubles& a) {
  CTRACK;
  double out = 0.;
  // #pragma omp simd
  double * u_ = REAL(u.data());
  double * kv_ = REAL(kv.data());
  double * a_ = REAL(a.data());
  for (int i = 0; i < u.size(); ++i) {
    // out += std::pow(u[i] * kv[i] - a[i], 2);
    out += (u_[i] * kv_[i] - a_[i]) * (u_[i] * kv_[i] - a_[i]);
  }
  return std::sqrt(out);
}

inline doubles uKv(const doubles& u, const doubles& K, const doubles& v) {
  CTRACK;
  int M = u.size();
  int N = v.size();
  writable::doubles P(M*N);
  double * P_ = REAL(P.data());
  double * u_ = REAL(u.data());
  double * K_ = REAL(K.data());
  double * v_ = REAL(v.data());
  for (int j{0}; j < N; ++j) {
    for (int i{0}; i < M; ++i) {
      P_[i + j * N] = u_[i] * K_[i + j * N] * v_[j];
    }
  }
  set_dim(P, M, N);
  return P;
}

/////////////////////////////////////////////////////////////////
// Overloading Operators
/////////////////////////////////////////////////////////////////

// template<typename T, typename S>
// inline doubles operator/(const T& x, const S& y) {
//   // if x is double, then y must be vec
//   bool x_is_vec = std::is_same<T, doubles>::value();
//   bool y_is_vec = std::is_same<S, doubles>::value();
//   int n = x_is_vec ? x.size() : y.size();
//   // create output
//   writable::doubles r(n);
//   // set output dimension
//   x_is_vec ? set_dim(r, x) : set_dim(r, y);
//   // loop
//   for (int i = 0; i < n; ++i) {
//     if (x_is_vec && y_is_vec) {
//
//     }
//   }
// }

// <<
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

// /
inline doubles operator/(const doubles& x, const double a) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] / a;
  }
  return out;
}
inline doubles operator/(const double a, const doubles& x) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = a / x[i];
  }
  return out;
}
inline doubles operator/(const doubles& x, const doubles& y) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] / y[i];
  }
  return out;
}

// *
inline doubles operator*(const doubles& x, const double a) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] * a;
  }
  return out;
}
inline doubles operator*(const double a, const doubles& x) {
  return x * a;
}
inline doubles operator%(const doubles& x, const doubles& y) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] * y[i];
  }
  return out;
}

// +
inline doubles operator+(const doubles& x, const double a) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] + a;
  }
  return out;
}
inline doubles operator+(const double a, const doubles& x) {
  return x + a;
}
inline doubles operator+(const doubles& x, const doubles& y) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] + y[i];
  }
  return out;
}

// -
inline doubles operator-(const doubles& x, const double a) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] - a;
  }
  return out;
}
inline doubles operator-(const double a, const doubles& x) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = a - x[i];
  }
  return out;
}
inline doubles operator-(const doubles& x, const doubles& y) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = x[i] - y[i];
  }
  return out;
}
inline doubles operator-(const doubles& x) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = -x[i];
  }
  return out;
}

// ^
inline doubles operator^(const doubles& x, const double a) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = std::pow(x[i], a);
  }
  return out;
}
inline doubles operator^(const double a, const doubles& x) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = std::pow(a, x[i]);
  }
  return out;
}
inline doubles operator^(const doubles& x, const doubles& y) {
  writable::doubles out(x.size());
  set_dim(out, x);
  for (int i = 0; i < x.size(); ++i) {
    out[i] = std::pow(x[i], y[i]);
  }
  return out;
}


}


#endif // BLAS_UTILS_H
