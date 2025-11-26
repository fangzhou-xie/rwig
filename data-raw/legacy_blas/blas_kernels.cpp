//
// // implementation file for the blas kernels
//
//
// #include <cpp11/doubles.hpp>      // doubles
// // #include <cpp11/function.hpp>     // package functions: used in the old isSymmetric
//
// #include "blas_kernels.hpp"         // blas kernels
//
// #include <Rinternals.h>           // R_DimSymbol, ...
// #include <R_ext/BLAS.h>           // BLAS routines
//
// using namespace cpp11;
//
// // R_NilValue: NULL
//
// namespace blas {
//
// /////////////////////////////////////////////////////////////////
// // Utilities
// /////////////////////////////////////////////////////////////////
//
//
// // void recover_dim(writable::doubles& r, const doubles& a) {
// //   // a: input
// //   // r: output
// //   // setup the by setting dimension: ncol and nrow
// //   r.attr(R_DimSymbol) = r_vector<int>({Rf_nrows(a), Rf_ncols(a)});
// //   // remove the dimension: the numeric vectors can resume their shape
// //   r.attr(R_DimSymbol) = R_NilValue;
// // }
//
// /////////////////////////////////////////////////////////////////
// // Extending BLAS: Hadamard stuff
// /////////////////////////////////////////////////////////////////
//
// doubles dhmpd(const doubles& x, const doubles& y) {
//   writable::doubles r(x.size());
//   for (int i =  0; i < x.size(); ++i) {
//     r[i] = x[i] * y[i];
//   }
//   if (Rf_ncols(x) > 1) { r.attr(R_DimSymbol) = {Rf_nrows(x), Rf_ncols(x)}; }
//   return r;
// }
//
// doubles dhmdv(const doubles& x, const doubles& y) {
//   writable::doubles r(x.size());
//   for (int i =  0; i < x.size(); ++i) {
//     r[i] = x[i] / y[i];
//   }
//   if (Rf_ncols(x) > 1) { r.attr(R_DimSymbol) = {Rf_nrows(x), Rf_ncols(x)}; }
//   return r;
// }
//
// /////////////////////////////////////////////////////////////////
// // Level 1 BLAS: daxpy, ddot, dnrm2, dasum, idamax
// /////////////////////////////////////////////////////////////////
//
// void daxpy(
//     writable::doubles& y, // output modify in-place
//     const doubles& x,
//     const double alpha = 1.
// ) {
//   int N = y.size();
//   int incx = 1;
//   int incy = 1;
//   double * x_ = safe[REAL](x);
//   double * y_ = safe[REAL](y);
//   F77_CALL(daxpy)(&N,&alpha,x_,&incx,y_,&incy);
// }
//
// void dscal(
//     writable::doubles& y, // output modify in-place
//     const double alpha = 1.
// ) {
//   int N = y.size();
//   int incy = 1;
//   double * y_ = safe[REAL](y);
//   F77_CALL(dscal)(&N,&alpha,y_,&incy);
// }
//
//
// // reduction: norms
// double ddot(
//     const doubles& x,
//     const doubles& y
// ) {
//   int n = x.size();
//   int incx = 1;
//   int incy = 1;
//   double * x_ = safe[REAL](x);
//   double * y_ = safe[REAL](y);
//
//   return F77_CALL(ddot)(&n,x_,&incx,y_,&incy);
// }
//
// double dnrm2(const doubles& x) {
//   int n = x.size();
//   int incx = 1;
//   double * x_ = safe[REAL](x);
//
//   return F77_CALL(dnrm2)(&n,x_,&incx);
// }
//
// double dasum(const doubles& x) {
//   int n = x.size();
//   int incx = 1;
//   double * x_ = safe[REAL](x);
//
//   return F77_CALL(dasum)(&n,x_,&incx);
// }
//
// double idamax(const doubles& x) {
//   int n = x.size();
//   int incx = 1;
//   double * x_ = safe[REAL](x);
//
//   return F77_CALL(idamax)(&n,x_,&incx);
// }
//
// /////////////////////////////////////////////////////////////////
// // Level 2 BLAS: dgemv, dsymv, dtrmv, dtrsv
// /////////////////////////////////////////////////////////////////
//
// // TODO: use emtpy vector to denote
// void dgemv(
//     const doubles& y, // output modify in-place
//     const doubles& A,
//     const doubles& x,
//     const bool trans,
//     const double alpha,
//     const double beta
// ) {
//   const char * TA = trans ? "T" : "N";
//   int M_s = safe[Rf_nrows](A);
//   int N_s = safe[Rf_ncols](A);
//   int M = trans ? N_s : M_s;
//   int N = trans ? M_s : N_s;
//   int LDA = M_s;
//   int incx = 1;
//   int incy = 1;
//   double * A_ = safe[REAL](A.data());
//   double * x_ = safe[REAL](x.data());
//   double * y_ = safe[REAL](y.data());
//
//   if (trans && (LDA < M)) { // LDA < M: invalid
//     safe[Rf_error](
//         "A has more cols (%d) than rows (%d): `trans` must be set to false",
//         N_s, M_s
//     );
//   }
//
//   F77_CALL(dgemv)(TA,&M,&N,&alpha,A_,&LDA,x_,&incx,&beta,y_,&incy,1);
// }
//
// void dsymv(
//     const doubles& y, // output modify in-place
//     const doubles& A,
//     const doubles& x,
//     const bool upper,
//     const double alpha,
//     const double beta
// ) {
//   // default to using upper all the time
//   const char * UL = upper ? "U" : "L";
//   int N = safe[Rf_nrows](A);
//   int LDA = N;
//   int incx = 1;
//   int incy = 1;
//   double * A_ = safe[REAL](A.data());
//   double * x_ = safe[REAL](x.data());
//   double * y_ = safe[REAL](y.data());
//   // TODO: need to make sure A is actually symmetric when transposing
//
//   F77_CALL(dsymv)(UL,&N,&alpha,A_,&LDA,x_,&incx,&beta,y_,&incy,1);
// }
//
// void dtrmv(
//     const doubles& x, // output modify in-place
//     const doubles& A,
//     const bool upper,
//     const bool trans,
//     const bool diag
// ) {
//   const char * UL = upper ? "U" : "L";
//   const char * TA = trans ? "T" : "N";
//   const char * DG = diag ? "U" : "N";
//   int N = safe[Rf_nrows](A);   // Triangular matrices are square
//   int LDA = N;                 // Leading dimension (typically = N)
//   int incx = 1;                // Stride for x (usually 1)
//   double* A_ = safe[REAL](A.data());
//   double* x_ = safe[REAL](x.data());
//
//   F77_CALL(dtrmv)(UL,TA,DG, &N, A_, &LDA, x_, &incx, 1, 1, 1);
// }
//
// void dtrsv(
//     const doubles& x, // output modify in-place
//     const doubles& A,
//     const bool upper,
//     const bool trans,
//     const bool diag
// ) {
//   const char * UL = upper ? "U" : "L";
//   const char * TA = trans ? "T" : "N";
//   const char * DG = diag ? "U" : "N";
//   int N = safe[Rf_nrows](A);   // Triangular matrices are square
//   int LDA = N;                 // Leading dimension (typically = N)
//   int incx = 1;                // Stride for x (usually 1)
//   double* A_ = safe[REAL](A.data());
//   double* x_ = safe[REAL](x.data());
//
//   F77_CALL(dtrsv)(UL,TA,DG, &N, A_, &LDA, x_, &incx, 1, 1, 1);
// }
//
//
// /////////////////////////////////////////////////////////////////
// // Level 3 BLAS: dgemm, dsymm, dtrmm, dtrsm
// /////////////////////////////////////////////////////////////////
//
// void dgemm(
//   writable::doubles& C,
//   const doubles& A,
//   const doubles& B,
//   const bool transA,
//   const bool transB,
//   const double alpha,
//   const double beta
// ) {
//   const char * TA = transA ? "T" : "N";
//   const char * TB = transB ? "T" : "N";
//
//   // Get dimensions of A and B
//   int A_rows = safe[Rf_nrows](A);
//   int A_cols = safe[Rf_ncols](A);
//   int B_rows = safe[Rf_nrows](B);
//   int B_cols = safe[Rf_ncols](B);
//
//   int M = transA ? A_cols : A_rows;
//   int K = transA ? A_rows : A_cols;
//   int K_B = transB ? B_cols : B_rows;
//   int N = transB ? B_rows : B_cols;
//
//   int LDA = A_rows;
//   int LDB = B_rows;
//   int LDC = M;
//   double * A_ = safe[REAL](A.data());
//   double * B_ = safe[REAL](B.data());
//   double * C_ = safe[REAL](C.data());
//   // C must have M*N elements
//   if ((Rf_nrows(C) != M) || (Rf_ncols(C) != N)) {
//     safe[Rf_error](
//         "C is expected to be %d*%d, but only receive %d*%d",
//         M, N, Rf_nrows(C), Rf_ncols(C)
//     );
//   }
//
//   // TODO: also had the same issue with `dgemv`?
//   F77_CALL(dgemm)(TA,TB,&M,&N,&K,&alpha,A_,&LDA,B_,&LDB,&beta,C_,&LDC,1,1);
// }
//
// void dsymm(
//   writable::doubles& C,
//   const doubles& A,
//   const doubles& B,
//   const bool left,
//   const bool upper,
//   const double alpha,
//   const double beta
// ) {
//   const char * LR = left ? "L" : "R";
//   const char * UL = upper ? "U" : "L";
//
//   int A_rows = safe[Rf_nrows](A);
//   int A_cols = safe[Rf_ncols](A);
//   int B_rows = safe[Rf_nrows](B);
//   int B_cols = safe[Rf_ncols](B);
//
//   int M = left ? A_rows : B_rows;
//   int N = left ? B_cols : A_cols;
//
//   int LDA = A_rows;
//   int LDB = B_rows;
//   int LDC = M;
//
//   double * A_ = safe[REAL](A.data());
//   double * B_ = safe[REAL](B.data());
//   double * C_ = safe[REAL](C.data());
//
//   F77_CALL(dsymm)(LR,UL,&M,&N,&alpha,A_,&LDA,B_,&LDB,&beta,C_,&LDC,1,1);
// }
//
// void dtrmm(
//   writable::doubles& B,
//   const doubles& A,
//   const bool left,
//   const bool upper,
//   const bool trans,
//   const bool diag,
//   const double alpha
// ) {
//   const char * LR = left ? "L" : "R";
//   const char * UL = upper ? "U" : "L";
//   const char * TA = trans ? "T" : "N";
//   const char * DG = diag ? "U" : "N";
//
//   int M = safe[Rf_nrows](B);
//   int N = safe[Rf_ncols](B);
//
//   int LDA = safe[Rf_nrows](A);;
//   int LDB = M;
//
//   double * A_ = safe[REAL](A.data());
//   double * B_ = safe[REAL](B.data());
//
//   F77_CALL(dtrmm)(LR,UL,TA,DG,&M,&N,&alpha,A_,&LDA,B_,&LDB,1,1,1,1);
// }
//
// void dtrsm(
//     writable::doubles& B,
//     const doubles& A,
//     const bool left,
//     const bool upper,
//     const bool trans,
//     const bool diag,
//     const double alpha
// ) {
//   const char * LR = left ? "L" : "R";
//   const char * UL = upper ? "U" : "L";
//   const char * TA = trans ? "T" : "N";
//   const char * DG = diag ? "U" : "N";
//
//   int M = safe[Rf_nrows](B);
//   int N = safe[Rf_ncols](B);
//
//   int LDA = safe[Rf_nrows](A);;
//   int LDB = M;
//
//   double * A_ = safe[REAL](A.data());
//   double * B_ = safe[REAL](B.data());
//
//   F77_CALL(dtrsm)(LR,UL,TA,DG,&M,&N,&alpha,A_,&LDA,B_,&LDB,1,1,1,1);
// }
//
// }
