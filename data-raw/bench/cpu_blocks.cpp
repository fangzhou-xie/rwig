// Per-iteration cost breakdown of rwig's CPU building blocks, using the
// package's own headers (no R runtime needed: only R's BLAS/LAPACK headers).
#include <chrono>
#include <cstdio>
#include <random>
#include "linalg.hpp"
#include "logdomain.hpp"
#include "thread_pool.hpp"
using clk = std::chrono::steady_clock;
template <typename F> double ms(F f, int reps = 3) { f(); auto t0 = clk::now(); for (int r = 0; r < reps; ++r) f(); return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps; }
int main() {
  std::mt19937 rng(1); std::uniform_real_distribution<double> U(0, 1);
  auto fill = [&](la::Mat &X) { for (la::idx k = 0; k < X.size(); ++k) X[k] = U(rng); };
  auto fillv = [&](la::Vec &x) { for (la::idx k = 0; k < x.size(); ++k) x[k] = U(rng); };

  // ---- log Sinkhorn building blocks, M = N = 2000, reg = .05 ----
  {
    const int M = 2000, N = 2000; const double reg = .05;
    la::Mat C(M, N); fill(C); la::Vec f(M), g(N), out(M), out2(N), x(M), y(N); fillv(f); fillv(g); fillv(x); fillv(y);
    logdom::Problem p{C.data(), M, N, reg}; logdom::Scratch s; s.resize(M, N, true);
    printf("log Sinkhorn blocks (M=N=%d): per call, ms\n", M);
    for (int nt : {0, 4, 8, 16, 24}) {
      ThreadPool pool(nt);
      double t_rows = ms([&] { logdom::soft_min_rows(pool, p, f.data(), g.data(), out.data(), s); });
      double t_cols = ms([&] { logdom::soft_min_cols(pool, p, f.data(), g.data(), out2.data()); });
      double t_xt = ms([&] { logdom::apply_XT(pool, p, f.data(), g.data(), -1.0, x.data(), y.data(), s); });
      double t_w = ms([&] { logdom::apply_W(pool, p, f.data(), g.data(), -1.0, y.data(), x.data(), s); });
      printf("  threads=%d: soft_min_rows %.1f  soft_min_cols %.1f  apply_XT %.1f  apply_W %.1f  => fwd iter (rows+2*cols) %.1f, bwd iter (XT+W) %.1f\n",
             nt, t_rows, t_cols, t_xt, t_w, t_rows + 2 * t_cols, t_xt + t_w);
    }
    // what is the floor? pure exp over M*N elements, and pure streaming read of C
    la::Mat E(M, N); double t_exp = ms([&] { for (la::idx k = 0; k < E.size(); ++k) E[k] = std::exp(-C[k] / reg); });
    double acc = 0; double t_read = ms([&] { double a = 0; for (la::idx k = 0; k < C.size(); ++k) a += C[k]; acc += a; });
    printf("  floors: exp() over M*N = %.1f ms (%.1f ns/exp), streaming read of C = %.1f ms%s\n", t_exp, 1e6 * t_exp / (M * N), t_read, acc > 1e300 ? "" : "");
    // gemv on E (used by apply_XT / apply_W)
    double t_gemv = ms([&] { la::gemv(true, E, x.data(), y.data()); });
    printf("  gemv 2000x2000 (reference BLAS) = %.1f ms\n", t_gemv);
  }
  // ---- log barycenter, N = 1000, S = 4: per topic soft-mins + spectral norm ----
  {
    const int M = 1000, N = 1000, S = 4; const double reg = .05;
    la::Mat C(M, N); fill(C); la::Mat F(M, S), G(N, S), out(M, S); fill(F); fill(G);
    logdom::Problem p{C.data(), M, N, reg}; logdom::Scratch s; s.resize(M, N, true); ThreadPool pool(0);
    double t_rows = ms([&] { for (int t = 0; t < S; ++t) logdom::soft_min_rows(pool, p, F.col(t), G.col(t), out.col(t), s); });
    la::Mat err(M, S); fill(err);
    double t_sn = ms([&] { la::spectral_norm(err); });
    printf("log barycenter (N=%d, S=%d): S soft_min_rows %.1f ms, spectral_norm(err M x S via dgesdd) %.2f ms per iteration\n", N, S, t_rows, t_sn);
  }
  // ---- WDL CPU, N = 500, S = 4, B = 32: the GEMMs vs the elementwise work ----
  {
    const int N = 500, S = 4, B = 32, SD = S * B;
    la::Mat C(N, N); fill(C); la::KernelOp K(C, 0.5); la::Mat X(N, SD), Y(N, SD); fill(X);
    double t_gemm = ms([&] { K.mul(false, X, Y, SD); });
    double t_pow = ms([&] { for (la::idx k = 0; k < X.size(); ++k) Y[k] = std::pow(X[k], 0.3); });
    double t_div = ms([&] { for (la::idx k = 0; k < X.size(); ++k) Y[k] = X[k] / (1 + Y[k]); });
    la::Mat X1(N, S), Y1(N, S); fill(X1);
    double t_gemm_inf = ms([&] { K.mul(false, X1, Y1, S); });
    printf("WDL CPU (N=%d, S=%d, B=%d): symm/gemm %dx%dx%d = %.2f ms (x4 per iteration), pow over N*S*B = %.2f ms, divide = %.2f ms; inference gemm %dx%dx%d = %.2f ms\n",
           N, S, B, N, N, SD, t_gemm, t_pow, t_div, N, N, S, t_gemm_inf);
    printf("  => one training iteration ~ %.1f ms of GEMM vs ~%.1f ms elementwise; %.1f GFLOP/s on this BLAS\n", 4 * t_gemm, 2 * t_pow + 6 * t_div, 2.0 * N * N * SD / t_gemm / 1e6);
  }
  return 0;
}
