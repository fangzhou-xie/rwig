// Minimal column-major dense linear algebra for rwig.
//
// Vec / Mat own their storage in std::vector<double>. There are no expression
// templates: element-wise work is written as explicit loops at the call site,
// and matrix products go straight to the BLAS/LAPACK that R itself links.
// Every BLAS wrapper takes raw pointers plus a leading dimension so that a
// row block or column block of a matrix can be passed by offsetting the
// pointer (used for threaded gemv on sub-blocks).

#ifndef RWIG_LINALG_H
#define RWIG_LINALG_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#ifndef FCONE
#define FCONE
#endif

namespace la {

using idx = std::size_t;

/*
  Containers
*/

class Vec {
public:
  Vec() = default;
  explicit Vec(idx n) : _d(n, 0.0) {}
  Vec(idx n, double value) : _d(n, value) {}
  Vec(const double *p, idx n) : _d(p, p + n) {}

  idx size() const { return _d.size(); }
  double *data() { return _d.data(); }
  const double *data() const { return _d.data(); }
  double &operator[](idx i) { return _d[i]; }
  double operator[](idx i) const { return _d[i]; }
  double &operator()(idx i) { return _d[i]; }
  double operator()(idx i) const { return _d[i]; }

  void resize(idx n) { _d.assign(n, 0.0); }
  void resize(idx n, double value) { _d.assign(n, value); }
  void fill(double value) { std::fill(_d.begin(), _d.end(), value); }
  void zeros() { fill(0.0); }
  void ones() { fill(1.0); }

  double sum() const {
    double s = 0.0;
    for (double x : _d) s += x;
    return s;
  }

private:
  std::vector<double> _d;
};

class Mat {
public:
  Mat() = default;
  Mat(idx nrow, idx ncol) : _nrow(nrow), _ncol(ncol), _d(nrow * ncol, 0.0) {}
  Mat(idx nrow, idx ncol, double value)
      : _nrow(nrow), _ncol(ncol), _d(nrow * ncol, value) {}
  Mat(const double *p, idx nrow, idx ncol)
      : _nrow(nrow), _ncol(ncol), _d(p, p + nrow * ncol) {}

  idx nrow() const { return _nrow; }
  idx ncol() const { return _ncol; }
  idx size() const { return _d.size(); }
  double *data() { return _d.data(); }
  const double *data() const { return _d.data(); }
  double *col(idx j) { return _d.data() + j * _nrow; }
  const double *col(idx j) const { return _d.data() + j * _nrow; }
  double &operator()(idx i, idx j) { return _d[i + j * _nrow]; }
  double operator()(idx i, idx j) const { return _d[i + j * _nrow]; }
  double &operator[](idx k) { return _d[k]; }
  double operator[](idx k) const { return _d[k]; }

  void resize(idx nrow, idx ncol) {
    _nrow = nrow;
    _ncol = ncol;
    _d.assign(nrow * ncol, 0.0);
  }
  void resize(idx nrow, idx ncol, double value) {
    _nrow = nrow;
    _ncol = ncol;
    _d.assign(nrow * ncol, value);
  }
  void fill(double value) { std::fill(_d.begin(), _d.end(), value); }
  void zeros() { fill(0.0); }
  void ones() { fill(1.0); }

  // is the matrix square and exactly symmetric?
  bool is_symmetric() const {
    if (_nrow != _ncol) return false;
    for (idx j = 0; j < _ncol; ++j)
      for (idx i = j + 1; i < _nrow; ++i)
        if ((*this)(i, j) != (*this)(j, i)) return false;
    return true;
  }
  // copy the upper triangle into the lower triangle (arma::symmatu)
  void symmetrize_upper() {
    for (idx j = 0; j < _ncol; ++j)
      for (idx i = j + 1; i < _nrow; ++i) (*this)(i, j) = (*this)(j, i);
  }

private:
  idx _nrow = 0, _ncol = 0;
  std::vector<double> _d;
};

/*
  BLAS level 1
*/

inline double dot(int n, const double *x, const double *y) {
  const int one = 1;
  return F77_CALL(ddot)(&n, x, &one, y, &one);
}

inline double nrm2(int n, const double *x) {
  const int one = 1;
  return F77_CALL(dnrm2)(&n, x, &one);
}

// y <- alpha * x + y
inline void axpy(int n, double alpha, const double *x, double *y) {
  const int one = 1;
  F77_CALL(daxpy)(&n, &alpha, x, &one, y, &one);
}

inline void scal(int n, double alpha, double *x) {
  const int one = 1;
  F77_CALL(dscal)(&n, &alpha, x, &one);
}

/*
  BLAS level 2
  A is m x n with leading dimension lda.
  y <- alpha * op(A) * x + beta * y, op(A) = A (trans = false) or A^T
*/

inline void gemv(bool trans, int m, int n, double alpha, const double *A,
                 int lda, const double *x, double beta, double *y) {
  const int one = 1;
  const char t = trans ? 'T' : 'N';
  F77_CALL(dgemv)(&t, &m, &n, &alpha, A, &lda, x, &one, &beta, y, &one FCONE);
}

inline void gemv(bool trans, const Mat &A, const double *x, double *y,
                 double alpha = 1.0, double beta = 0.0) {
  gemv(trans, (int)A.nrow(), (int)A.ncol(), alpha, A.data(), (int)A.nrow(), x,
       beta, y);
}

// symmetric A (n x n, upper triangle referenced): y <- alpha*A*x + beta*y
inline void symv(int n, double alpha, const double *A, int lda,
                 const double *x, double beta, double *y) {
  const int one = 1;
  const char uplo = 'U';
  F77_CALL(dsymv)(&uplo, &n, &alpha, A, &lda, x, &one, &beta, y, &one FCONE);
}

inline void symv(const Mat &A, const double *x, double *y, double alpha = 1.0,
                 double beta = 0.0) {
  symv((int)A.nrow(), alpha, A.data(), (int)A.nrow(), x, beta, y);
}

/*
  BLAS level 3
  C (m x n) <- alpha * op(A) * op(B) + beta * C
  op(A) is m x k, op(B) is k x n
*/

inline void gemm(bool transA, bool transB, int m, int n, int k, double alpha,
                 const double *A, int lda, const double *B, int ldb,
                 double beta, double *C, int ldc) {
  const char ta = transA ? 'T' : 'N';
  const char tb = transB ? 'T' : 'N';
  F77_CALL(dgemm)(&ta, &tb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C,
                  &ldc FCONE FCONE);
}

// C <- alpha * op(A) * B + beta * C with B, C full matrices (B not transposed)
inline void gemm(bool transA, const Mat &A, const Mat &B, Mat &C,
                 double alpha = 1.0, double beta = 0.0) {
  const int m = transA ? (int)A.ncol() : (int)A.nrow();
  const int k = transA ? (int)A.nrow() : (int)A.ncol();
  gemm(transA, false, m, (int)B.ncol(), k, alpha, A.data(), (int)A.nrow(),
       B.data(), (int)B.nrow(), beta, C.data(), (int)C.nrow());
}

// symmetric A (m x m, upper): C (m x n) <- alpha * A * B + beta * C
inline void symm(int m, int n, double alpha, const double *A, int lda,
                 const double *B, int ldb, double beta, double *C, int ldc) {
  const char side = 'L';
  const char uplo = 'U';
  F77_CALL(dsymm)(&side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C,
                  &ldc FCONE FCONE);
}

inline void symm(const Mat &A, const Mat &B, Mat &C, double alpha = 1.0,
                 double beta = 0.0) {
  symm((int)A.nrow(), (int)B.ncol(), alpha, A.data(), (int)A.nrow(), B.data(),
       (int)B.nrow(), beta, C.data(), (int)C.nrow());
}

/*
  LAPACK: thin SVD  A (m x n) = U diag(s) V^T
  U is m x r, s has r entries, Vt is r x n with r = min(m, n).
  Returns the LAPACK info code (0 on success). A is left untouched.
*/

inline int gesdd_thin(const Mat &A, Mat &U, Vec &s, Mat &Vt) {
  const int m = (int)A.nrow(), n = (int)A.ncol();
  const int r = std::min(m, n);
  Mat work_A(A); // dgesdd destroys its input
  U.resize(m, r);
  s.resize(r);
  Vt.resize(r, n);
  const char jobz = 'S';
  int info = 0;
  int lwork = -1;
  double wkopt = 0.0;
  std::vector<int> iwork(8 * (std::size_t)r);
  // workspace query
  F77_CALL(dgesdd)(&jobz, &m, &n, work_A.data(), &m, s.data(), U.data(), &m,
                   Vt.data(), &r, &wkopt, &lwork, iwork.data(), &info FCONE);
  if (info != 0) return info;
  lwork = (int)wkopt;
  std::vector<double> work((std::size_t)lwork);
  F77_CALL(dgesdd)(&jobz, &m, &n, work_A.data(), &m, s.data(), U.data(), &m,
                   Vt.data(), &r, work.data(), &lwork, iwork.data(),
                   &info FCONE);
  return info;
}

// largest singular value of A (arma::norm(A, 2) for a matrix)
inline double spectral_norm(const Mat &A) {
  const int m = (int)A.nrow(), n = (int)A.ncol();
  const int r = std::min(m, n);
  if (r == 0) return 0.0;
  Mat work_A(A);
  std::vector<double> s((std::size_t)r);
  const char jobz = 'N';
  int info = 0, lwork = -1;
  double wkopt = 0.0, dummy = 0.0;
  const int ldu = 1, ldvt = 1;
  std::vector<int> iwork(8 * (std::size_t)r);
  F77_CALL(dgesdd)(&jobz, &m, &n, work_A.data(), &m, s.data(), &dummy, &ldu,
                   &dummy, &ldvt, &wkopt, &lwork, iwork.data(), &info FCONE);
  lwork = (int)wkopt;
  std::vector<double> work((std::size_t)lwork);
  F77_CALL(dgesdd)(&jobz, &m, &n, work_A.data(), &m, s.data(), &dummy, &ldu,
                   &dummy, &ldvt, work.data(), &lwork, iwork.data(),
                   &info FCONE);
  return s[0];
}

} // namespace la

#endif // RWIG_LINALG_H
