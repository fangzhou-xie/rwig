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
void nip_sumsq(double *loss, double *x, double *y, int n, cudaStream_t &stream);
// void nip_exp(double *y, int n, cudaStream_t &stream);
// void nip_KTU_w(double *KTUw, double *KTU, double *w, int m, int n,
//                cudaStream_t &stream);
void nip_b_div_KTU(double *V, double *b, double *KTU, int m, int n,
                   cudaStream_t &stream);
void nip_row_prod(double *x, double *A, int m, int n, cudaStream_t &stream);
void nip_row_prod_shared(double *x, double *A, int m, int n,
                         cudaStream_t &stream);
void nip_diag_scale(double *P, double *u, double *K, double *v, int m, int n,
                    cudaStream_t &stream);
void nip_minus_2(double *z, double *x, double *y, int N, cudaStream_t &stream);

// wrappers for the inplace kernels: last pointer being replaced!
void ip_dot(double *y, double *x, int n, cudaStream_t &stream);
void ip_div(double *y, double *x, int n, cudaStream_t &stream);
void ip_exp(double *x, int n, cudaStream_t &stream);
void ip_log(double *x, int n, cudaStream_t &stream);
void ip_fill_ones(double *x, int n, cudaStream_t &stream);
void ip_gibbs(double *x, int n, double reg, cudaStream_t &stream);
void ip_dot_minus(double *res, double *x, double *y, double *z, int n,
                  cudaStream_t &stream);
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
// BLAS-1: dscal, ddot, dnrm2, dasum
void dscal(double *y, int n, double alpha, cublasHandle_t &handle);
void ddot(double *result, double *x, double *y, int n, cublasHandle_t &handle);
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

// #endif
// #endif

#endif // RWIG_CUDA_KERNELS_CUH
