# Benchmark results

Machine: 24-core Linux box, R 4.6.1 with the reference BLAS/LAPACK shipped by
Debian (`libblas.so.3`), g++ 13.3. `Rscript data-raw/bench/bench.R <label>`
appends to `results.csv` (gitignored); `bench::mark` median of 3 runs.

Problem sizes: sinkhorn M = N = 2000 (100 iterations, gradient on);
barycenter N = 1000, S = 4 (log: 50 iterations); wdl N = 500, M = 128
docs, S = 4, batch 32, 1 epoch, 100 inner iterations; euclidean n = 5000,
d = 10; doc2dist 200 docs x 50 tokens, 5000-word dictionary.

| case                 | 0.1.0 (RcppArmadillo) | 0.2.0 (direct BLAS) | speed-up |
|----------------------|----------------------:|--------------------:|---------:|
| sinkhorn log, serial |               27.43 s |             14.56 s |     1.9x |
| sinkhorn log, 8 thr  |                5.99 s |              3.60 s |     1.7x |
| sinkhorn vanilla     |                3.12 s |              1.47 s |     2.1x |
| barycenter log, ser. |                9.26 s |              5.58 s |     1.7x |
| barycenter log, 8 thr|                6.05 s |              2.34 s |     2.6x |
| barycenter parallel  |                1.67 s |              1.03 s |     1.6x |
| wdl (CPU)            |               42.98 s |             22.34 s |     1.9x |
| euclidean            |               0.327 s |             0.305 s |     1.1x |
| doc2dist             |               0.729 s |             0.005 s |    150x  |

Notes

- The log-domain kernels are now bound by `exp()` throughput (about five
  exponentials per matrix element per iteration including the backward
  pass); the remaining time is arithmetic, not memory traffic.
- `wdl` on the reference BLAS is GEMM-bound; an optimized BLAS (OpenBLAS,
  MKL) will help it far more than any further change in this package.
- Shared library size on this machine (Debian R flags include `-g`):
  CUDA build 15 MB -> 8.2 MB unstripped (860 KB -> 680 KB stripped); the
  CPU-only build (`RWIG_NO_CUDA=1`) is 7.4 MB unstripped, 284 KB stripped.
  The Armadillo template instantiations were the bulk of the old binary.
