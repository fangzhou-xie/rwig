// Fused kernels for the log-domain (log-stabilized) Sinkhorn / barycenter
// algorithms.
//
// All kernels work on the implicit matrix
//     R_ij = C_ij - f_i - g_j          (C is M x N, column-major, ld = M)
// without ever forming it. The "soft-min" of a row is
//     min_j R_ij - reg * log sum_j exp(-(R_ij - min_j R_ij) / reg)
// (Xie 2025); likewise for columns. The backward passes need the row- and
// column-softmax matrices X and W of -R / reg applied to a vector; those
// fill one caller-provided M x N buffer E and reduce it with BLAS gemv on
// row / column blocks, so a ThreadPool can split the work.

#ifndef RWIG_LOGDOMAIN_H
#define RWIG_LOGDOMAIN_H

#include <cmath>
#include <limits>

#include "linalg.hpp"
#include "thread_pool.hpp"

namespace logdom {

// problem description shared by all kernels
struct Problem {
  const double *C;
  int M, N;
  double reg;
};

/*
  serial kernels on a row range [i0, i1) or column range [j0, j1)
*/

// row minima of R(f, g)
inline void row_min(const Problem &p, const double *f, const double *g, int i0,
                    int i1, double *rowmin) {
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = i0; i < i1; ++i) rowmin[i] = inf;
  for (int j = 0; j < p.N; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < rowmin[i]) rowmin[i] = r;
    }
  }
}

// row soft-min: out[i] = rowmin_i - reg * log(rowsum_i); rowmin / rowsum are
// caller-provided scratch of length M and hold the raw min / sum on return
inline void row_lse(const Problem &p, const double *f, const double *g, int i0,
                    int i1, double *out, double *rowmin, double *rowsum) {
  row_min(p, f, g, i0, i1, rowmin);
  for (int i = i0; i < i1; ++i) rowsum[i] = 0.0;
  for (int j = 0; j < p.N; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      rowsum[i] += std::exp(-(r - rowmin[i]) / p.reg);
    }
  }
  for (int i = i0; i < i1; ++i)
    out[i] = rowmin[i] - p.reg * std::log(rowsum[i]);
}

// column soft-min: out[j] = colmin_j - reg * log(colsum_j)
inline void col_lse(const Problem &p, const double *f, const double *g, int j0,
                    int j1, double *out) {
  for (int j = j0; j < j1; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    const double gj = g[j];
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < p.M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < m) m = r;
    }
    double s = 0.0;
    for (int i = 0; i < p.M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      s += std::exp(-(r - m) / p.reg);
    }
    out[j] = m - p.reg * std::log(s);
  }
}

// E_ij = exp(-(R_ij - rowmin_i) / reg) for rows [i0, i1), accumulating rowsum
inline void fill_E_rowstab(const Problem &p, const double *f, const double *g,
                           const double *rowmin, int i0, int i1, double *E,
                           double *rowsum) {
  for (int i = i0; i < i1; ++i) rowsum[i] = 0.0;
  for (int j = 0; j < p.N; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    double *Ej = E + (std::size_t)j * p.M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      const double e = std::exp(-(r - rowmin[i]) / p.reg);
      Ej[i] = e;
      rowsum[i] += e;
    }
  }
}

// E_ij = exp(-(R_ij - colmin_j) / reg) for columns [j0, j1), with colmin, colsum
inline void fill_E_colstab(const Problem &p, const double *f, const double *g,
                           int j0, int j1, double *E, double *colmin,
                           double *colsum) {
  for (int j = j0; j < j1; ++j) {
    const double *Cj = p.C + (std::size_t)j * p.M;
    double *Ej = E + (std::size_t)j * p.M;
    const double gj = g[j];
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < p.M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < m) m = r;
    }
    double s = 0.0;
    for (int i = 0; i < p.M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      const double e = std::exp(-(r - m) / p.reg);
      Ej[i] = e;
      s += e;
    }
    colmin[j] = m;
    colsum[j] = s;
  }
}

/*
  pooled operations (rows or columns split across the pool's threads)
*/

// scratch shared by the pooled operations: E is M x N, the vectors M or N
struct Scratch {
  la::Mat E;
  la::Vec rowmin, rowsum, colmin, colsum, xM, xN;
  void resize(int M, int N, bool with_E) {
    if (with_E) E.resize(M, N);
    rowmin.resize(M);
    rowsum.resize(M);
    xM.resize(M);
    colmin.resize(N);
    colsum.resize(N);
    xN.resize(N);
  }
};

// out (M) = row soft-min of R(f, g)
inline void soft_min_rows(ThreadPool &pool, const Problem &p, const double *f,
                          const double *g, double *out, Scratch &s) {
  double *rowmin = s.rowmin.data(), *rowsum = s.rowsum.data();
  pool.parallel_for(p.M, [=](int i0, int i1) {
    row_lse(p, f, g, i0, i1, out, rowmin, rowsum);
  });
}

// out (N) = column soft-min of R(f, g)
inline void soft_min_cols(ThreadPool &pool, const Problem &p, const double *f,
                          const double *g, double *out) {
  pool.parallel_for(p.N, [=](int j0, int j1) { col_lse(p, f, g, j0, j1, out); });
}

// y (N) = alpha * X^T x with X the row-wise softmax of -R(f, g) / reg
inline void apply_XT(ThreadPool &pool, const Problem &p, const double *f,
                     const double *g, double alpha, const double *x, double *y,
                     Scratch &s) {
  const int M = p.M, N = p.N;
  double *E = s.E.data();
  double *rowmin = s.rowmin.data(), *rowsum = s.rowsum.data();
  pool.parallel_for(M, [=](int i0, int i1) { row_min(p, f, g, i0, i1, rowmin); });
  pool.parallel_for(M, [=](int i0, int i1) {
    fill_E_rowstab(p, f, g, rowmin, i0, i1, E, rowsum);
  });
  double *xs = s.xM.data();
  for (int i = 0; i < M; ++i) xs[i] = x[i] / rowsum[i];
  pool.parallel_for(N, [=](int j0, int j1) {
    la::gemv(true, M, j1 - j0, alpha, E + (std::size_t)j0 * M, M, xs, 0.0,
             y + j0);
  });
}

// x (M) = alpha * W y with W the column-wise softmax of -R(f, g) / reg.
// s.colmin / s.colsum hold the column minima and sums on return, so the
// column soft-min of the same R is colmin_j - reg * log(colsum_j).
inline void apply_W(ThreadPool &pool, const Problem &p, const double *f,
                    const double *g, double alpha, const double *y, double *x,
                    Scratch &s) {
  const int M = p.M, N = p.N;
  double *E = s.E.data();
  double *colmin = s.colmin.data(), *colsum = s.colsum.data();
  pool.parallel_for(N, [=](int j0, int j1) {
    fill_E_colstab(p, f, g, j0, j1, E, colmin, colsum);
  });
  double *ys = s.xN.data();
  for (int j = 0; j < N; ++j) ys[j] = y[j] / colsum[j];
  pool.parallel_for(M, [=](int i0, int i1) {
    la::gemv(false, i1 - i0, N, alpha, E + i0, M, ys, 0.0, x + i0);
  });
}

} // namespace logdom

#endif // RWIG_LOGDOMAIN_H
