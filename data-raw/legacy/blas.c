
#include <R.h>
#include <Rinternals.h>

SEXP mat_mult_blas(SEXP A_, SEXP a_rows_, SEXP B_, SEXP b_cols_, SEXP k_) {

  char * TRANSA = "N";
  char * TRANSB = "N";

  double * A = REAL(A_);
  double * B = REAL(B_);

  int K = asInteger(k_);
  int M = asInteger(a_rows_);
  int N = asInteger(b_cols_);

  int LDA = K;
  int LDB = N;

  double ALPHA = 1.0;
  double BETA = 0.0;

  SEXP result_;
  PROTECT(result_ = allocMatrix(REALSXP, M, N));

  double * result = REAL(result_);
  F77_CALL(dgemm)(
    TRANSA, TRANSB, &M, &N, &K, &ALPHA,
    A, &LDA, B, &LDB, &BETA, result, &N);

  UNPROTECT(1);

  return(result_);
};
