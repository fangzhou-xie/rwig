// check cuda availability and versions

#include "common.hpp"

#include "check_cuda.hpp" // for checking cuda availability

// [[Rcpp::export]]
bool cuda_available_cpp() { return check_cuda::has_cuda; }

// #include "cuda_kernels.hpp"

// // [[Rcpp::export]]
// SEXP test_cuda_dgemm(SEXP A, SEXP B, const bool transA, const bool transB,
//                      const double alpha, const double beta) {

//   double *A_ptr = REAL(A);
//   double *B_ptr = REAL(B);

//   // Get dimensions of A and B
//   int A_rows = Rf_nrows(A);
//   int A_cols = Rf_ncols(A);
//   int B_rows = Rf_nrows(B);
//   int B_cols = Rf_ncols(B);

//   int M = transA ? A_cols : A_rows;
//   int K = transA ? A_rows : A_cols;
//   int K_B = transB ? B_cols : B_rows;
//   int N = transB ? B_rows : B_cols;

//   SEXP C_ = PROTECT(Rf_allocVector(REALSXP, M * N));
//   double *C_ptr = REAL(C_);

//   cuda_dgemm(C_ptr, alpha, A_ptr, transA, B_ptr, transB, M, N, K, beta);

//   // Set matrix dimensions
//   SEXP dims = PROTECT(Rf_allocVector(INTSXP, 2));
//   INTEGER(dims)[0] = M;
//   INTEGER(dims)[1] = N;
//   Rf_setAttrib(C_, R_DimSymbol, dims);

//   UNPROTECT(2);

//   return C_;
// }

// test the cuda kernel: getting data from R SEXP
// // [[Rcpp::export]]
// SEXP test_cuda_add(SEXP x, SEXP y) {
//   double *x_ptr = REAL(x);
//   double *y_ptr = REAL(y);
//   int n = LENGTH(x);
//   // create a copy of y pointer for the kernel

//   double *x_cuda;
//   double *y_cuda;
//   cudaMalloc(&x_cuda, n * sizeof(double));
//   cudaMalloc(&y_cuda, n * sizeof(double));
//   cudaMemcpy(x_cuda, x_ptr, n * sizeof(double), cudaMemcpyHostToDevice);
//   cudaMemcpy(y_cuda, y_ptr, n * sizeof(double), cudaMemcpyHostToDevice);

//   element_add(n, x_cuda, y_cuda);
//   cudaDeviceSynchronize();

//   // copy back
//   SEXP res_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *res_ptr = REAL(res_);
//   cudaMemcpy(res_ptr, y_cuda, n * sizeof(double), cudaMemcpyDeviceToHost);
//   UNPROTECT(1);

//   cudaFree(x_cuda);
//   cudaFree(y_cuda);
//   return res_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_add(SEXP x, SEXP y) {
//   double *x_ptr = REAL(x);
//   double *y_ptr = REAL(y);
//   int n = LENGTH(x);

//   SEXP res_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *res_ptr = REAL(res_);
//   element_add(res_ptr, n, x_ptr, y_ptr);
//   UNPROTECT(1);

//   return res_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_minus(SEXP x, SEXP y) {
//   double *x_ptr = REAL(x);
//   double *y_ptr = REAL(y);
//   int n = LENGTH(x);

//   SEXP res_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *res_ptr = REAL(res_);
//   element_minus(res_ptr, n, x_ptr, y_ptr);
//   UNPROTECT(1);

//   return res_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_dot(SEXP x, SEXP y) {
//   double *x_ptr = REAL(x);
//   double *y_ptr = REAL(y);
//   int n = LENGTH(x);

//   SEXP res_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *res_ptr = REAL(res_);
//   element_dot(res_ptr, n, x_ptr, y_ptr);
//   UNPROTECT(1);

//   return res_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_div(SEXP x, SEXP y) {
//   double *x_ptr = REAL(x);
//   double *y_ptr = REAL(y);
//   int n = LENGTH(x);

//   SEXP res_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *res_ptr = REAL(res_);
//   element_div(res_ptr, n, x_ptr, y_ptr);
//   UNPROTECT(1);

//   return res_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_dgemv(SEXP A, SEXP x, const bool trans = false,
//                      const double alpha = 1., const double beta = 0.) {
//   double *A_ptr = REAL(A);
//   double *x_ptr = REAL(x);
//   int M = Rf_nrows(A);
//   int N = Rf_ncols(A);

//   SEXP y_ = PROTECT(Rf_allocVector(REALSXP, trans ? N : M));
//   double *y_ptr = REAL(y_);
//   cuda_dgemv(y_ptr, A_ptr, x_ptr, M, N, trans, alpha, beta);
//   UNPROTECT(1);

//   return y_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_a_div_Kv(SEXP a, SEXP K, SEXP v) {
//   double *a_ptr = REAL(a);
//   double *K_ptr = REAL(K);
//   double *v_ptr = REAL(v);
//   int m = Rf_nrows(K);
//   int n = Rf_ncols(K);

//   SEXP u_ = PROTECT(Rf_allocVector(REALSXP, m));
//   double *u_ptr = REAL(u_);
//   cuda_a_div_Kv(u_ptr, m, n, a_ptr, K_ptr, v_ptr);
//   UNPROTECT(1);

//   return u_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_b_div_Ktu(SEXP b, SEXP K, SEXP u) {
//   double *b_ptr = REAL(b);
//   double *K_ptr = REAL(K);
//   double *u_ptr = REAL(u);
//   int m = Rf_nrows(K);
//   int n = Rf_ncols(K);

//   SEXP v_ = PROTECT(Rf_allocVector(REALSXP, n));
//   double *v_ptr = REAL(v_);
//   cuda_b_div_Ktu(v_ptr, m, n, b_ptr, K_ptr, u_ptr);
//   UNPROTECT(1);

//   return v_;
// }

// // [[Rcpp::export]]
// SEXP test_cuda_norm(SEXP x) {
//   double *x_ptr = REAL(x);
//   int n = LENGTH(x);

//   double res_ = element_norm(x_ptr, n);

//   return Rf_ScalarReal(res_);
// }

// // [[Rcpp::export]]
// SEXP test_cuda_diag_scale(SEXP u, SEXP K, SEXP v) {
//   double *u_ptr = REAL(u);
//   double *K_ptr = REAL(K);
//   double *v_ptr = REAL(v);
//   int m = Rf_nrows(K);
//   int n = Rf_ncols(K);

//   SEXP out_ = PROTECT(Rf_allocVector(REALSXP, m * n));
//   double *out_ptr = REAL(out_);
//   cuda_diag_scale(m, n, out_ptr, u_ptr, K_ptr, v_ptr);

//   // Set matrix dimensions
//   SEXP dims = PROTECT(Rf_allocVector(INTSXP, 2));
//   INTEGER(dims)[0] = m;
//   INTEGER(dims)[1] = n;
//   Rf_setAttrib(out_, R_DimSymbol, dims);
//   UNPROTECT(2);

//   return out_;
// }

// // [[Rcpp::export]]
// Rcpp::List test_cuda_sinkhorn_vanilla(SEXP a, SEXP b, SEXP C, const double
// reg,
//                                       const bool withgrad, const int
//                                       max_iter, const double zero_tol) {
//   double *a_ptr = REAL(a);
//   double *b_ptr = REAL(b);
//   double *C_ptr = REAL(C);
//   int m = Rf_nrows(C);
//   int n = Rf_ncols(C);

//   SEXP u_ = PROTECT(Rf_allocVector(REALSXP, m));
//   SEXP v_ = PROTECT(Rf_allocVector(REALSXP, n));
//   SEXP P_ = PROTECT(Rf_allocVector(REALSXP, m * n));
//   SEXP grad_a_ = PROTECT(Rf_allocVector(REALSXP, m));

//   double *u_ptr = REAL(u_);
//   double *v_ptr = REAL(v_);
//   double *P_ptr = REAL(P_);
//   double *grad_a_ptr = REAL(grad_a_);

//   double loss = 0.;
//   int iter = 0;
//   double err = 0.;

//   cuda_sinkhorn_vanilla(P_ptr, grad_a_ptr, u_ptr, v_ptr, &loss, &iter, &err,
//                         a_ptr, b_ptr, C_ptr, m, n, reg, withgrad, max_iter,
//                         zero_tol);

//   // Set matrix dimensions for P
//   SEXP dims = PROTECT(Rf_allocVector(INTSXP, 2));
//   INTEGER(dims)[0] = m;
//   INTEGER(dims)[1] = n;
//   Rf_setAttrib(P_, R_DimSymbol, dims);

//   UNPROTECT(5);

//   int return_code;
//   if (err <= zero_tol) {
//     return_code = 0;
//   } else if (iter == max_iter) {
//     return_code = 1;
//   } else {
//     return_code = 2;
//   }

//   if (withgrad) {
//     return Rcpp::List::create(
//         Rcpp::Named("P") = P_, Rcpp::Named("grad_a") = grad_a_,
//         Rcpp::Named("u") = u_, Rcpp::Named("v") = v_,
//         Rcpp::Named("loss") = loss, Rcpp::Named("iter") = iter,
//         Rcpp::Named("err") = err, Rcpp::Named("return_status") =
//         return_code);
//   } else {
//     return Rcpp::List::create(
//         Rcpp::Named("P") = P_, Rcpp::Named("u") = u_, Rcpp::Named("v") = v_,
//         Rcpp::Named("loss") = loss, Rcpp::Named("iter") = iter,
//         Rcpp::Named("err") = err, Rcpp::Named("return_status") =
//         return_code);
//   }
// }
