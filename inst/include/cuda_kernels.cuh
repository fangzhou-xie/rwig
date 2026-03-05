// declare CUDA kernels here

#ifndef RWIG_CUDA_KERNELS_CUH
#define RWIG_CUDA_KERNELS_CUH

// #include "check_cuda.hpp" // for checking cuda availability

// #ifdef HAVE_CUBLAS
// #ifdef HAVE_CUDA_RUNTIME

#include <cmath>
#include <cstdio> // for printf
#include <cublas_v2.h>
#include <cuda_runtime.h>

const int BLOCK_SIZE = 512;

// For matrices (column-major)
inline void print_device_matrix(double *d_arr, int nrow, int ncol,
                                const char *name, cudaStream_t &stream) {
  auto D2H = cudaMemcpyDeviceToHost;
  double *h_arr = (double *)malloc(nrow * ncol * sizeof(double));
  cudaMemcpyAsync(h_arr, d_arr, nrow * ncol * sizeof(double), D2H, stream);

  printf("%s (%d x %d):\n", name, nrow, ncol);
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      printf("%.6f ", h_arr[i + j * nrow]); // column-major
    }
    printf("\n");
  }
  printf("\n");

  free(h_arr);
}

// wrappers for the non-inplace kernels
void nip_add(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_minus(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_dot(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_div(double *z, double *x, double *y, int n, cudaStream_t &stream);
void nip_log(double *y, double *x, int n, cudaStream_t &stream);
void nip_dot_div(double *z, double *xbar, double *x, double *y, int n,
                 cudaStream_t &stream);
void nip_dot_minus(double *res, double *x, double *y, double *z, int n,
                   cudaStream_t &stream);
void nip_sumsq(double *loss, double *x, double *y, int n, cudaStream_t &stream);
// void nip_exp(double *y, int n, cudaStream_t &stream);
// void nip_KTU_w(double *KTUw, double *KTU, double *w, int m, int n,
//                cudaStream_t &stream);
void nip_b_div_KTU(double *V, double *b, double *KTU, int m, int n,
                   cudaStream_t &stream);
void nip_row_sum(double *x, double *A, int m, int n, cudaStream_t &stream);
void nip_row_prod(double *x, double *A, int m, int n, cudaStream_t &stream);
void nip_row_prod_shared(double *x, double *A, int m, int n,
                         cudaStream_t &stream);
void nip_diag_scale(double *P, double *u, double *K, double *v, int m, int n,
                    cudaStream_t &stream);
void nip_minus_2(double *z, double *x, double *y, int N, cudaStream_t &stream);

// wrappers for the inplace kernels: last pointer being replaced!
void ip_add(double *y, double *x, int n, cudaStream_t &stream);
void ip_dot(double *y, double *x, int n, cudaStream_t &stream);
void ip_div(double *y, double *x, int n, cudaStream_t &stream);
void ip_exp(double *x, int n, cudaStream_t &stream);
void ip_log(double *x, int n, cudaStream_t &stream);
void ip_fill_ones(double *x, int n, cudaStream_t &stream);
void ip_gibbs(double *x, int n, double reg, cudaStream_t &stream);
void ip_dot_minus_sum(double *res, double *x, double *y, double *z, int n,
                      cudaStream_t &stream);
void ip_dot_div(double *y, double *xbar, double *x, int n,
                cudaStream_t &stream);
void ip_KTU_w(double *KTU, double *w, int m, int n, cudaStream_t &stream);
void ip_PbarK(double *PbarK, double *C, double *P, double *K, int m, int n,
              double reg, cudaStream_t &stream);
void ip_accu_abar(double *abar, double *ubar, double *Kv, int n,
                  cudaStream_t &stream);
void ip_sinkloss(double *loss, double *C, double *P, int m, int n, double reg,
                 cudaStream_t &stream);

// wrappers for BLAS functions
// BLAS-1: daxpy, dscal, ddot, dnrm2, dasum
void daxpy(double *y, double alpha, double *x, int n, cublasHandle_t &handle);
void dscal(double *y, int n, double alpha, cublasHandle_t &handle);
void ddot(double *result, double *x, double *y, int n, cublasHandle_t &handle);
void dnrm2(double *result, double *x, int n, cublasHandle_t &handle);
void dasum(double *result, double *x, int n, cublasHandle_t &handle);
// BLAS-2: dgemv, dsymv, dger
void dgemv(double *y, double alpha, double *A, int M, int N, bool transA,
           double *x, double beta, cublasHandle_t &handle);
void dsymv(double *y, double alpha, double *A, int N, double *x, double beta,
           cublasHandle_t &handle);
void dger(double *A, int M, int N, double alpha, double *x, double *y,
          cublasHandle_t &handle);
void dsyr(double *A, int N, double alpha, double *x, cublasHandle_t &handle);
// BLAS-3: dgemm, dsymm
void dgemm(double *C, double alpha, double *A, bool transA, double *B,
           bool transB, int M, int N, int K, double beta,
           cublasHandle_t &handle);
void dsymm(double *C, double alpha, double *A, double *B, int M, int N,
           double beta, cublasHandle_t &handle);

// kernels for the Sinkhorn/Barycenter
void init_ones(double *x, int n, cudaStream_t &stream);
void update_K(double *K, int M, int N, double reg, cudaStream_t &stream);
void normalize(double *x, int n, cublasHandle_t &handle);

// optimizer step: opt = 0 (SGD), 1 (Adam), 2 (AdamW)
// d_m, d_v are momentum buffers (zeroed by caller, unused for SGD)
void optimizer_step(double *d_theta, double *d_g, double *d_m, double *d_v,
                    int opt, int n, double eta, double gamma, double beta1,
                    double beta2, double eps, int step, cudaStream_t &stream);

// TODO: replace this with the batched version
// internal barycenter interface (used by WDL)
void impl_barycenter(int &iter, double &err, double *U, double *V, double *b,
                     double *Ubar, double *Vbar, double *bbar, double *Abar,
                     double *wbar, double *Uhist, double *Vhist, double *bhist,
                     double *KVhist, double *KTUhist, double *A, double *w,
                     double *b_ext, double *K, double *KV, double *KTU, int M,
                     int N, int S, const int max_iter, const double zero_tol,
                     bool withgrad, cudaStream_t &stream,
                     cublasHandle_t &handle);

// #endif
// #endif

// void softmax(double *out, double *in, int nrows, int ncols);
// void softmax_jac(double *out, double *in, int nrows, int ncols);
// void diagmul(double *out, double *a, double *b, int n);

#endif // RWIG_CUDA_KERNELS_CUH
