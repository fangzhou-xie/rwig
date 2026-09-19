// Fused kernels for the log-domain (log-stabilized) Sinkhorn / barycenter
// algorithms.
//
// All kernels work on the implicit matrix
//     R_ij = C_ij - f_i - g_j          (C is M x N, column-major, ld = M)
// without ever forming it. Each kernel processes a contiguous range of rows
// or columns so that a ThreadPool can split the work; the serial version is
// the same function with the full range.
//
// Notation: "row lse" of row i is   min_j R_ij - reg * log sum_j exp(-(R_ij - min_j R_ij)/reg)
// which is the soft-min used in Xie (2025); likewise "col lse".

#ifndef RWIG_LOGDOMAIN_H
#define RWIG_LOGDOMAIN_H

#include <cmath>
#include <limits>

namespace logdom {

// column soft-min for columns [j0, j1): out[j] = cmin_j - reg*log(colsum_j)
// optionally also returns the raw min / sum (used by the backward pass)
inline void col_lse(const double *C, int M, int N, const double *f,
                    const double *g, double reg, int j0, int j1, double *out,
                    double *colmin = nullptr, double *colsum = nullptr) {
  (void)N;
  for (int j = j0; j < j1; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    const double gj = g[j];
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < m) m = r;
    }
    double s = 0.0;
    for (int i = 0; i < M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      s += std::exp(-(r - m) / reg);
    }
    out[j] = m - reg * std::log(s);
    if (colmin) colmin[j] = m;
    if (colsum) colsum[j] = s;
  }
}

// row soft-min for rows [i0, i1): out[i] = rmin_i - reg*log(rowsum_i)
// two column-major passes over the row block (min pass, then exp-sum pass).
// rowmin / rowsum are caller-provided scratch of length M (only [i0,i1) used)
// and hold the raw min / sum on return (used by the backward pass).
inline void row_lse(const double *C, int M, int N, const double *f,
                    const double *g, double reg, int i0, int i1, double *out,
                    double *rowmin, double *rowsum) {
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = i0; i < i1; ++i) rowmin[i] = inf;
  for (int j = 0; j < N; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < rowmin[i]) rowmin[i] = r;
    }
  }
  for (int i = i0; i < i1; ++i) rowsum[i] = 0.0;
  for (int j = 0; j < N; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      rowsum[i] += std::exp(-(r - rowmin[i]) / reg);
    }
  }
  for (int i = i0; i < i1; ++i) out[i] = rowmin[i] - reg * std::log(rowsum[i]);
}

// row minima only, rows [i0, i1)
inline void row_min(const double *C, int M, int N, const double *f,
                    const double *g, int i0, int i1, double *rowmin) {
  const double inf = std::numeric_limits<double>::infinity();
  for (int i = i0; i < i1; ++i) rowmin[i] = inf;
  for (int j = 0; j < N; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < rowmin[i]) rowmin[i] = r;
    }
  }
}

// column minima only, columns [j0, j1)
inline void col_min(const double *C, int M, int N, const double *f,
                    const double *g, int j0, int j1, double *colmin) {
  (void)N;
  for (int j = j0; j < j1; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    const double gj = g[j];
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < m) m = r;
    }
    colmin[j] = m;
  }
}

// backward helpers: fill E (M x N, same layout as C) with the row-stabilized
// exponentials E_ij = exp(-(R_ij - rowmin_i)/reg) for rows [i0, i1), and
// accumulate rowsum_i. rowmin must be complete for those rows.
inline void fill_E_rowstab(const double *C, int M, int N, const double *f,
                           const double *g, double reg, const double *rowmin,
                           int i0, int i1, double *E, double *rowsum) {
  for (int i = i0; i < i1; ++i) rowsum[i] = 0.0;
  for (int j = 0; j < N; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    double *Ej = E + (std::size_t)j * M;
    const double gj = g[j];
    for (int i = i0; i < i1; ++i) {
      const double r = Cj[i] - f[i] - gj;
      const double e = std::exp(-(r - rowmin[i]) / reg);
      Ej[i] = e;
      rowsum[i] += e;
    }
  }
}

// fill E with the column-stabilized exponentials for columns [j0, j1) and
// return colmin_j, colsum_j (computed here; no precomputed min needed)
inline void fill_E_colstab(const double *C, int M, int N, const double *f,
                           const double *g, double reg, int j0, int j1,
                           double *E, double *colmin, double *colsum) {
  (void)N;
  for (int j = j0; j < j1; ++j) {
    const double *Cj = C + (std::size_t)j * M;
    double *Ej = E + (std::size_t)j * M;
    const double gj = g[j];
    double m = std::numeric_limits<double>::infinity();
    for (int i = 0; i < M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      if (r < m) m = r;
    }
    double s = 0.0;
    for (int i = 0; i < M; ++i) {
      const double r = Cj[i] - f[i] - gj;
      const double e = std::exp(-(r - m) / reg);
      Ej[i] = e;
      s += e;
    }
    colmin[j] = m;
    colsum[j] = s;
  }
}

} // namespace logdom

#endif // RWIG_LOGDOMAIN_H
