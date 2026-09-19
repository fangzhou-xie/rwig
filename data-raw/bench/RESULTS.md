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
  After also dropping Rcpp (native .Call interface) the CUDA build is
  3.8 MB unstripped.

# Profiling (2026-09-19, same machine)

Tools: `data-raw/bench/cpu_blocks.cpp` times each per-iteration building block
with the package's own headers (build with
`g++ -std=c++17 -O2 -pthread -I inst/include -I $(R RHOME)/include cpu_blocks.cpp -llapack -lblas`);
`nsys profile --trace=cuda --stats=true Rscript <script>` for the GPU.

CPU, log-domain Sinkhorn (M = N = 2000, per iteration):

| threads | forward (rows + 2 cols soft-min) | backward (X^T, W products) |
|--------:|--------------------------------:|---------------------------:|
| 0       | 81 ms                           | 62 ms                      |
| 4       | 33 ms                           | 29 ms                      |
| 8       | 27 ms                           | 21 ms                      |
| 16      | 19 ms                           | 17 ms                      |
| 24      | 17 ms                           | 16 ms                      |

The floor is `exp()`: 5.3 ns per element on one core, and the kernels run at
about 80% of that floor. Five exponentials per matrix element per iteration
(three forward, two backward) is the whole cost. Scaling stops at about 5x on
12 physical cores (SMT does not help a compute-bound loop).

CPU, WDL (N = 500, S = 4, B = 32): one training iteration is ~63 ms of GEMM
(four 500x500x128 products at 4 GFLOP/s on the reference BLAS) against ~2 ms
of everything else. An optimized BLAS is the only lever that matters here.

GPU, WDL (RTX 3090 Ti, N = 1000, 512 docs, S = 4, B = 64, 20 fixed
iterations; wall 3.5 s, GPU busy 2.6 s):

| where                          | GPU time | share |
|--------------------------------|---------:|------:|
| training GEMMs (cutlass, FP64) |   0.77 s |   29% |
| inference: 2 x 1000x1000x4 GEMM per doc-iteration | 0.80 s | 31% |
| inference: `nip_rowprod_pow`   |   0.48 s |   19% |
| inference: `dnrm2` + host sync |   0.40 s |   15% |
| everything else                |   0.15 s |    6% |

Training GEMMs run at ~0.43 TFLOP/s, close to the card's FP64 peak (GeForce
runs FP64 at 1/64 rate). Inference is the bottleneck: 512 documents x ~18
iterations, each with two GEMV-shaped GEMMs, a latency-bound `pow` kernel
launched on 2 blocks, and a synchronizing norm. Batching inference over
documents (as training already is) removes ~9000 of the ~9200 syncs.

# After batching GPU inference and the log/exp product of powers (2026-09-19)

GPU WDL, RTX 3090 Ti, wall time:

| problem                                   | before  | after   | speed-up |
|-------------------------------------------|--------:|--------:|---------:|
| N = 2000, 4096 docs, S = 4, B = 64, L = 20 | 93.1 s | 27.8 s  |   3.35x |
| N = 1000, 512 docs, S = 4, B = 64, L = 20  |  3.10 s |  1.66 s |   1.9x  |

GPU time is now 94% cuBLAS/cutlass FP64 GEMMs (training and batched
inference), which run at the card's double-precision peak; the former
inference kernels (`nrm2`, GEMV-shaped GEMMs, `pow`) are gone from the top
of the profile. Batched inference reproduces the per-document CPU
barycenters to 4e-10.

CPU with OpenBLAS 0.3.26 (single BLAS thread, as the package sets at attach):

| case                 | reference BLAS | OpenBLAS |
|----------------------|---------------:|---------:|
| sinkhorn log, serial |        14.56 s |  14.18 s |
| sinkhorn vanilla     |         1.47 s |   0.61 s |
| barycenter parallel  |         1.03 s |   0.46 s |
| wdl (CPU)            |        22.34 s |   4.31 s |

The log-domain kernels are unchanged (exp-bound, no BLAS in the forward);
everything GEMM/GEMV-bound gained 2-5x from the BLAS swap alone.

# Stage 5: parallel barycenter backward reuses K V / K^T U from the forward

CPU, OpenBLAS, `with_grad = TRUE`, fixed iteration count (zero_tol = 0):

| problem              | recompute (5 GEMM/step) | stored history (2 GEMM/step) | speed-up |
|----------------------|------------------------:|-----------------------------:|---------:|
| N = 1000, S = 4, 100 it |               0.447 s |                      0.275 s |    1.6x |
| N = 3000, S = 4, 50 it  |               3.534 s |                      2.166 s |    1.6x |
| N = 1000, S = 32, 100 it |              1.080 s |                      0.693 s |    1.6x |

Cost: two more (L + 1) x N x S histories, i.e. the gradient path's memory
doubles (e.g. N = 3000, S = 4, 1000 iterations: ~190 MB -> ~380 MB).
Results are unchanged (golden gate 53/53).
