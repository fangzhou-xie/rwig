
// kernels for the BLAS routines: wrapping lower-level BLAS

#ifndef BLAS_KERNELS_H
#define BLAS_KERNELS_H

// #include <cmath>    // exp, log, etc
// #include <iostream> // cout
// #include <iomanip>  // setprecision

#include <R_ext/BLAS.h>            // BLAS routines

// #include <cpp11/integers.hpp>      // integers, aka r_vecotor<int>
#include <cpp11/doubles.hpp>       // doubles, aka r_vecotor<double>

#include "ctrack.hpp"

using namespace cpp11;

// R_NilValue: NULL

// TODO: header only inline without c++ implementation file

// define everything under the `blas` namespace
namespace blas {


/////////////////////////////////////////////////////////////////
// Extending BLAS: Hadamard stuff
/////////////////////////////////////////////////////////////////

// following the naming convention of BLAS
// d: double
// hm: Hadamard
// pd, dv: product, division

// inline void dhmpd(writable::doubles& x, const doubles& y) {
//   check_dim(x, y);
//   double * x_ = safe[REAL](x);
//   for (int i = 0; i < x.size(); ++i) {
//     x_[i] = x_[i] * y[i];
//   }
// }
//
// inline void dhmdv(writable::doubles& x, const doubles& y) {
//   check_dim(x, y);
//   double * x_ = safe[REAL](x);
//   for (int i = 0; i < x.size(); ++i) {
//     x_[i] = x_[i] / y[i];
//   }
// }
//
// inline void dhmpw(writable::doubles& x, const doubles& y) {
//   check_dim(x, y);
//   double * x_ = safe[REAL](x);
//   for (int i = 0; i < x.size(); ++i) {
//     x_[i] = std::pow(x_[i], y[i]);
//   }
// }

inline doubles pow(const doubles& x, const double a) {CTRACK;
  writable::doubles x_(x);
  for (int i = 0; i < x.size(); ++i) {
    x_[i] = std::pow(x[i], a);
  }
  return x_;
}

inline doubles exp(const doubles& x) {CTRACK;
  writable::doubles x_(x);
  for (int i = 0; i < x.size(); ++i) {
    x_[i] = std::exp(x[i]);
  }
  return x_;
}

inline doubles log(const doubles& x) {CTRACK;
  writable::doubles x_(x);
  for (int i = 0; i < x.size(); ++i) {
    x_[i] = std::log(x[i]);
  }
  return x_;
}


/////////////////////////////////////////////////////////////////
// Level 1 BLAS: daxpy, ddot, dnrm2, dasum, idamax
/////////////////////////////////////////////////////////////////

inline void daxpy(
    writable::doubles& y, // output modify in-place
    const doubles& x,
    const double alpha = 1.
) {CTRACK;
  // check_vec(x, "daxpy: x");
  // check_vec(y, "daxpy: y");
  int N = y.size();
  int incx = 1;
  int incy = 1;
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());
  F77_CALL(daxpy)(&N,&alpha,x_,&incx,y_,&incy);
}

inline void dscal(
    writable::doubles& y, // output modify in-place
    const double alpha = 1.
) {CTRACK;
  // check_vec(y, "dscal: y");
  int N = y.size();
  int incy = 1;
  double * y_ = REAL(y.data());
  F77_CALL(dscal)(&N,&alpha,y_,&incy);
}

inline double ddot(const doubles& x, const doubles& y) {CTRACK;
  // check_vec(x, "ddot: x");
  // check_vec(y, "ddot: y");
  int n = x.size();
  int incx = 1;
  int incy = 1;
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());

  return F77_CALL(ddot)(&n,x_,&incx,y_,&incy);
}

inline double dnrm2(const doubles& x) {CTRACK;
  // check_vec(x, "dnrm2: x");
  int n = x.size();
  int incx = 1;
  double * x_ = REAL(x.data());

  return F77_CALL(dnrm2)(&n,x_,&incx);
}

inline double dasum(const doubles& x) {CTRACK;
  // check_vec(x, "dasum: x");
  int n = x.size();
  int incx = 1;
  double * x_ = REAL(x.data());

  return F77_CALL(dasum)(&n,x_,&incx);
}

inline double idamax(const doubles& x) {CTRACK;
  // check_vec(x, "idamax: x");
  int n = x.size();
  int incx = 1;
  double * x_ = REAL(x.data());

  return F77_CALL(idamax)(&n,x_,&incx);
}

/////////////////////////////////////////////////////////////////
// Level 2 BLAS: dgemv, dsymv, dtrmv, dtrsv
/////////////////////////////////////////////////////////////////

inline void dgemv(
    writable::doubles& y, // output modify in-place
    const doubles& A,
    const doubles& x,
    const bool trans = false,
    const double alpha = 1.,
    const double beta = 0.
) {CTRACK;
  const char * TA = trans ? "T" : "N";
  int M = Rf_nrows(A);
  int N = Rf_ncols(A);
  int LDA = M;
  int incx = 1;
  int incy = 1;
  double * A_ = REAL(A.data());
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());

  if (trans && (LDA < M)) { // LDA < M: invalid
    safe[Rf_error](
        "A has more cols (%d) than rows (%d): `trans` must be set to false",
        N, M
    );
  }
  // if ((trans ? N : M) != y.size()) {
  //   safe[Rf_error](
  //     "output y does not match the dimension with A*x"
  //   );
  // }

  F77_CALL(dgemv)(TA,&M,&N,&alpha,A_,&LDA,x_,&incx,&beta,y_,&incy,1);
}

inline void dsymv(
    writable::doubles& y, // output modify in-place
    const doubles& A,
    const doubles& x,
    const bool upper = true,
    const double alpha = 1.,
    const double beta = 0.
) {CTRACK;
  // default to using upper all the time
  const char * UL = upper ? "U" : "L";
  int N = Rf_nrows(A);
  int LDA = N;
  int incx = 1;
  int incy = 1;
  double * A_ = REAL(A.data());
  double * x_ = REAL(x.data());
  double * y_ = REAL(y.data());
  // TODO: need to make sure A is actually symmetric when transposing

  F77_CALL(dsymv)(UL,&N,&alpha,A_,&LDA,x_,&incx,&beta,y_,&incy,1);
}

inline void dtrmv(
    writable::doubles& x, // output modify in-place
    const doubles& A,
    const bool upper = true,
    const bool trans = false
    // const bool diag = false
) {CTRACK;
  const char * UL = upper ? "U" : "L";
  const char * TA = trans ? "T" : "N";
  // const char * DG = diag ? "U" : "N";
  const char * DG = "N";
  int N = Rf_nrows(A);   // Triangular matrices are square
  int LDA = N;                 // Leading dimension (typically = N)
  int incx = 1;                // Stride for x (usually 1)
  double* A_ = REAL(A.data());
  double* x_ = REAL(x.data());

  F77_CALL(dtrmv)(UL,TA,DG, &N, A_, &LDA, x_, &incx, 1, 1, 1);
}

inline void dtrsv(
    writable::doubles& x, // output modify in-place
    const doubles& A,
    const bool upper = true,
    const bool trans = false
    // const bool diag = false
) {CTRACK;
  const char * UL = upper ? "U" : "L";
  const char * TA = trans ? "T" : "N";
  // const char * DG = diag ? "U" : "N";
  const char * DG = "N";
  int N = Rf_nrows(A);   // Triangular matrices are square
  int LDA = N;                 // Leading dimension (typically = N)
  int incx = 1;                // Stride for x (usually 1)
  double* A_ = REAL(A.data());
  double* x_ = REAL(x.data());

  F77_CALL(dtrsv)(UL,TA,DG, &N, A_, &LDA, x_, &incx, 1, 1, 1);
}


/////////////////////////////////////////////////////////////////
// Level 3 BLAS: dgemm, dsymm, dtrmm, dtrsm
/////////////////////////////////////////////////////////////////

inline void dgemm(
    writable::doubles& C, // output modify in-place
    const doubles& A,
    const doubles& B,
    const bool transA = false,
    const bool transB = false,
    const double alpha = 1.0,
    const double beta = 0.0
) {CTRACK;
  const char * TA = transA ? "T" : "N";
  const char * TB = transB ? "T" : "N";

  // Get dimensions of A and B
  int A_rows = Rf_nrows(A);
  int A_cols = Rf_ncols(A);
  int B_rows = Rf_nrows(B);
  int B_cols = Rf_ncols(B);

  int M = transA ? A_cols : A_rows;
  int K = transA ? A_rows : A_cols;
  int K_B = transB ? B_cols : B_rows;
  int N = transB ? B_rows : B_cols;

  int LDA = A_rows;
  int LDB = B_rows;
  int LDC = M;
  double * A_ = REAL(A.data());
  double * B_ = REAL(B.data());
  double * C_ = REAL(C.data());

  // C must have M*N elements
  // if ((Rf_nrows(C) != M) || (Rf_ncols(C) != N)) {
  //   safe[Rf_error](
  //       "C is expected to be %d*%d, but only receive %d*%d",
  //       M, N, Rf_nrows(C), Rf_ncols(C)
  //   );
  // }

  // TODO: also had the same issue with `dgemv`?
  F77_CALL(dgemm)(TA,TB,&M,&N,&K,&alpha,A_,&LDA,B_,&LDB,&beta,C_,&LDC,1,1);
}

inline void dsymm(
    writable::doubles& C, // output modify in-place
    const doubles& A,
    const doubles& B,
    const bool left = true,
    const bool upper = true,
    const double alpha = 1.0,
    const double beta = 0.0
) {CTRACK;
  const char * LR = left ? "L" : "R";
  const char * UL = upper ? "U" : "L";

  int A_rows = Rf_nrows(A);
  int A_cols = Rf_ncols(A);
  int B_rows = Rf_nrows(B);
  int B_cols = Rf_ncols(B);

  int M = left ? A_rows : B_rows;
  int N = left ? B_cols : A_cols;

  int LDA = A_rows;
  int LDB = B_rows;
  int LDC = M;

  double * A_ = REAL(A.data());
  double * B_ = REAL(B.data());
  double * C_ = REAL(C.data());

  F77_CALL(dsymm)(LR,UL,&M,&N,&alpha,A_,&LDA,B_,&LDB,&beta,C_,&LDC,1,1);
}

inline void dtrmm(
    writable::doubles& B, // output modify in-place
    const doubles& A,
    const bool left = true,
    const bool upper = true,
    const bool trans = false,
    // const bool diag = false,
    const double alpha = 1.0
) {CTRACK;
  const char * LR = left ? "L" : "R";
  const char * UL = upper ? "U" : "L";
  const char * TA = trans ? "T" : "N";
  // const char * DG = diag ? "U" : "N";
  const char * DG = "N";

  int M = Rf_nrows(B);
  int N = Rf_ncols(B);

  int LDA = safe[Rf_nrows](A);;
  int LDB = M;

  double * A_ = REAL(A.data());
  double * B_ = REAL(B.data());

  F77_CALL(dtrmm)(LR,UL,TA,DG,&M,&N,&alpha,A_,&LDA,B_,&LDB,1,1,1,1);
}

inline void dtrsm(
    writable::doubles& B, // output modify in-place
    const doubles& A,
    const bool left = true,
    const bool upper = true,
    const bool trans = false,
    // const bool diag = false,
    const double alpha = 1.0
) {CTRACK;
  const char * LR = left ? "L" : "R";
  const char * UL = upper ? "U" : "L";
  const char * TA = trans ? "T" : "N";
  // const char * DG = diag ? "U" : "N";
  const char * DG = "N";

  int M = Rf_nrows(B);
  int N = Rf_ncols(B);

  int LDA = safe[Rf_nrows](A);;
  int LDB = M;

  double * A_ = REAL(A.data());
  double * B_ = REAL(B.data());

  F77_CALL(dtrsm)(LR,UL,TA,DG,&M,&N,&alpha,A_,&LDA,B_,&LDB,1,1,1,1);
}


/////////////////////////////////////////////////////////////////
// LAPACK
/////////////////////////////////////////////////////////////////


}


#endif // BLAS_KERNELS_H
