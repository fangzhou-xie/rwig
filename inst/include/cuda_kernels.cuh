// declare CUDA kernels here

#ifndef RWIG_CUDA_KERNELS_CUH
#define RWIG_CUDA_KERNELS_CUH

#include <cmath>
#include <cstdio> // for printf
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <R_ext/Print.h> // REprintf

const int BLOCK_SIZE = 512;

// CUDA error checking macro: reports and jumps to the caller's cleanup label
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      REprintf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
               cudaGetErrorString(err_));                                      \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

// grid size for a 1-D grid-stride kernel over n elements
inline int nblocks(int n) { return (n + BLOCK_SIZE - 1) / BLOCK_SIZE; }

// Block size for the one-block-per-column reduction kernels (softmax and
// its Jacobian). Those kernels need a power-of-two block; a block much wider
// than the column length only idles threads and adds sync rounds, so size
// it from the reduction length: the smallest power of two >= `length`,
// clamped to [32, BLOCK_SIZE] and to what the device allows.
inline int reduce_block(int length) {
  static int max_threads = 0; // queried once per process
  if (max_threads == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock,
                               dev) != cudaSuccess)
      max_threads = BLOCK_SIZE;
  }
  int bs = 32;
  while (bs < length && bs < BLOCK_SIZE) bs <<= 1;
  return bs < max_threads ? bs : max_threads;
}

// wrappers for the non-inplace kernels
void nip_minus(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_div(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_log(double *y, double *x, int n, cudaStream_t &stream);
void nip_dot_div(double *z, double *xbar, double *x, double *y, int n,
                 cudaStream_t &stream);
void nip_dot_minus(double *res, double *x, double *y, double *z, int n,
                   cudaStream_t &stream);
void nip_sumsq(double *loss, double *x, double *y, int n, cudaStream_t &stream);
void nip_b_div_KTU(double *V, double *b, double *KTU, int m, int n,
                   cudaStream_t &stream);
void nip_row_sum(double *x, double *A, int m, int n, cudaStream_t &stream);
void nip_diag_scale(double *P, double *u, double *K, double *v, int m, int n,
                    cudaStream_t &stream);
void nip_minus_2(double *z, double *x, double *y, int N, cudaStream_t &stream);
// b[i] = prod_s KTU[i, s] ^ w[s]  (KTU is m x n, left untouched)
void nip_row_prod_pow(double *b, double *KTU, double *w, int m, int n,
                      cudaStream_t &stream);

// wrappers for the inplace kernels: last pointer being replaced!
void ip_add(double *y, double *x, int n, cudaStream_t &stream);
void ip_dot(double *y, double *x, int n, cudaStream_t &stream);
void ip_fill_ones(double *x, int n, cudaStream_t &stream);
void ip_gibbs(double *x, int n, double reg, cudaStream_t &stream);
void ip_dot_div(double *y, double *xbar, double *x, int n,
                cudaStream_t &stream);
void ip_PbarK(double *PbarK, double *C, double *P, double *K, int m, int n,
              double reg, cudaStream_t &stream);
void ip_accu_abar(double *abar, double *ubar, double *Kv, int n,
                  cudaStream_t &stream);
void ip_sinkloss(double *loss, double *C, double *P, int m, int n, double reg,
                 cudaStream_t &stream);

// wrappers for BLAS functions
// BLAS-1: dscal, dnrm2, dasum
void dscal(double *y, int n, double alpha, cublasHandle_t &handle);
void dnrm2(double *result, double *x, int n, cublasHandle_t &handle);
void dasum(double *result, double *x, int n, cublasHandle_t &handle);
// BLAS-2: dgemv, dger
void dgemv(double *y, double alpha, double *A, int M, int N, bool transA,
           double *x, double beta, cublasHandle_t &handle);
void dger(double *A, int M, int N, double alpha, double *x, double *y,
          cublasHandle_t &handle);
// BLAS-3: dgemm
void dgemm(double *C, double alpha, double *A, bool transA, double *B,
           bool transB, int M, int N, int K, double beta,
           cublasHandle_t &handle);

// kernels for the Sinkhorn/Barycenter
void init_ones(double *x, int n, cudaStream_t &stream);
void update_K(double *K, int M, int N, double reg, cudaStream_t &stream);
void normalize(double *x, int n, cublasHandle_t &handle);

// optimizer step: opt = 0 (SGD), 1 (Adam), 2 (AdamW)
// d_m, d_v are momentum buffers (zeroed by caller, unused for SGD)
void optimizer_step(double *d_theta, double *d_g, double *d_m, double *d_v,
                    int opt, int n, double eta, double gamma, double beta1,
                    double beta2, double eps, int step, cudaStream_t &stream);

// internal barycenter interface (also used by cuda_wdl for inference)
// tmp_MS is an M*S scratch buffer used for the convergence check.
// The history buffers are only touched when withgrad is true.
void impl_barycenter(int &iter, double &err, double *U, double *V, double *b,
                     double *Ubar, double *Vbar, double *bbar, double *Abar,
                     double *wbar, double *Uhist, double *Vhist, double *bhist,
                     double *KVhist, double *KTUhist, double *A, double *w,
                     double *b_ext, double *K, double *KV, double *KTU,
                     double *tmp_MS, int M, int N, int S, const int max_iter,
                     const double zero_tol, bool withgrad, cudaStream_t &stream,
                     cublasHandle_t &handle);

#endif // RWIG_CUDA_KERNELS_CUH
