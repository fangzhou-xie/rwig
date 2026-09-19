// implementations of cuda kernels

#include "cuda_kernels.cuh"

/*
  CUDA kernels
*/

__global__ void nip_minus(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] - y[i];
}

__global__ void nip_div(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] / y[i];
}

__global__ void nip_log(int n, double *y, double *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = log(x[i]);
}

// kernel: z = (vbar % v) / y
__global__ void nip_dot_div(int n, double *xbar, double *x, double *y,
                            double *z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = (xbar[i] * x[i]) / y[i];
}

// kernel: res = x .* y - z
__global__ void nip_dot_minus(int n, double *res, double *x, double *y,
                              double *z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    res[i] = x[i] * y[i] - z[i];
}

__global__ void nip_sumsq(int n, double *loss, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double sum = 0.;
  for (int i = index; i < n; i += stride)
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  // *loss = 0.;
  atomicAdd(loss, sum);
}

__global__ void nip_b_div_KTU(int m, int n, double *V, double *b, double *KTU) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int k = idx; k < m * n; k += stride) {
    int i = k % m; // row
    // int j = k / m; // col
    V[k] = b[i] / KTU[k];
  }
}

// row sum of column-major matrix: result[i] = sum_j A[i + j*m]
__global__ void nip_row_sum(int m, int n, double *A, double *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < m; i += stride) {
    double sum = 0.0;
    for (int j = 0; j < n; j++)
      sum += A[i + j * m];
    result[i] = sum;
  }
}

// b[i] = prod_j A[i, j] ^ w[j]   (A is m x n column-major, not modified)
__global__ void nip_rowprod_pow(int m, int n, double *b, const double *A,
                                const double *w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < m; i += stride) {
    double prod = 1.0;
    for (int j = 0; j < n; j++)
      prod *= pow(A[i + j * m], w[j]);
    b[i] = prod;
  }
}

__global__ void nip_diag_scale(int m, int n, double *out, const double *u,
                               const double *K, const double *v) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int k = idx; k < m * n; k += stride) {
    int i = k % m; // row
    int j = k / m; // col
    out[k] = u[i] * K[k] * v[j];
  }
}

__global__ void nip_minus_2(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = index; i < n; i += stride)
    z[i] = 2 * (x[i] - y[i]);
}

// all the `inplace_*`  functions are in-place
__global__ void ip_add(int n, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

__global__ void ip_dot(int n, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * y[i];
}

// kernel: fill x with ones
__global__ void ip_fill_ones(int n, double *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] = 1.0;
}

// kernel: K = exp(-C / reg), aka Gibbs Kernel
__global__ void ip_gibbs(int n, double *x, const double reg) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] = exp(-x[i] / reg);
}

// kernel: y = (vbar % v) / y
__global__ void ip_dot_div(int n, double *xbar, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = (xbar[i] * x[i]) / y[i];
}

// kernel: PbarK = (C + reg * log(P)) .* K
__global__ void ip_PbarK(int n, double *PbarK, double *C, double *P, double *K,
                         const double reg) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    PbarK[i] = (C[i] + reg * log(P[i])) * K[i];
}

// kernel for accumulating gradient w.r.t. a
__global__ void ip_accu_abar(int n, double *abar, double *ubar, double *Kv) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    abar[i] += ubar[i] / Kv[i];
}

// kernel: sinkhorn loss
__global__ void ip_sinkloss(int n, double *loss, double *C, double *P,
                            double reg) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double sum = 0.;
  for (int i = index; i < n; i += stride)
    sum += C[i] * P[i] + reg * P[i] * (log(P[i]) - 1.0);
  atomicAdd(loss, sum);
}

/*
  wrappers for the non-inplace kernels
*/

void nip_minus(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  nip_minus<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, z, x, y);
}

void nip_div(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  nip_div<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, z, x, y);
}

void nip_log(double *y, double *x, int n, cudaStream_t &stream) {
  nip_log<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, y, x);
}

void nip_dot_div(double *z, double *xbar, double *x, double *y, int n,
                 cudaStream_t &stream) {
  nip_dot_div<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, xbar, x, y, z);
}

void nip_dot_minus(double *res, double *x, double *y, double *z, int n,
                   cudaStream_t &stream) {
  nip_dot_minus<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, res, x, y, z);
}

void nip_sumsq(double *loss, double *x, double *y, int n,
               cudaStream_t &stream) {
  nip_sumsq<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, loss, x, y);
}

void nip_b_div_KTU(double *V, double *b, double *KTU, int m, int n,
                   cudaStream_t &stream) {
  nip_b_div_KTU<<<nblocks(m * n), BLOCK_SIZE, 0, stream>>>(m, n, V, b, KTU);
}

void nip_row_sum(double *x, double *A, int m, int n, cudaStream_t &stream) {
  nip_row_sum<<<nblocks(m), BLOCK_SIZE, 0, stream>>>(m, n, A, x);
}

void nip_row_prod_pow(double *b, double *KTU, double *w, int m, int n,
                      cudaStream_t &stream) {
  nip_rowprod_pow<<<nblocks(m), BLOCK_SIZE, 0, stream>>>(m, n, b, KTU, w);
}

void nip_diag_scale(double *P, double *u, double *K, double *v, int m, int n,
                    cudaStream_t &stream) {
  nip_diag_scale<<<nblocks(m * n), BLOCK_SIZE, 0, stream>>>(m, n, P, u, K, v);
}

void nip_minus_2(double *z, double *x, double *y, int N, cudaStream_t &stream) {
  nip_minus_2<<<nblocks(N), BLOCK_SIZE, 0, stream>>>(N, z, x, y);
}

/*
  wrappers for the inplace kernels
*/

void ip_add(double *y, double *x, int n, cudaStream_t &stream) {
  ip_add<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, x, y);
}

void ip_dot(double *y, double *x, int n, cudaStream_t &stream) {
  ip_dot<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, x, y);
}

void ip_fill_ones(double *x, int n, cudaStream_t &stream) {
  ip_fill_ones<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, x);
}

void ip_gibbs(double *x, int n, double reg, cudaStream_t &stream) {
  ip_gibbs<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, x, reg);
}

void ip_dot_div(double *y, double *xbar, double *x, int n,
                cudaStream_t &stream) {
  ip_dot_div<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, xbar, x, y);
}

void ip_PbarK(double *PbarK, double *C, double *P, double *K, int m, int n,
              double reg, cudaStream_t &stream) {
  ip_PbarK<<<nblocks(m * n), BLOCK_SIZE, 0, stream>>>(m * n, PbarK, C, P, K, reg);
}

void ip_accu_abar(double *abar, double *ubar, double *Kv, int n,
                  cudaStream_t &stream) {
  ip_accu_abar<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, abar, ubar, Kv);
}

void ip_sinkloss(double *loss, double *C, double *P, int m, int n, double reg,
                 cudaStream_t &stream) {
  ip_sinkloss<<<nblocks(m * n), BLOCK_SIZE, 0, stream>>>(m * n, loss, C, P, reg);
}

/*
  BLAS wrappers
*/

void dscal(double *y, int n, double alpha, cublasHandle_t &handle) {
  const int inc = 1;
  cublasDscal(handle, n, &alpha, y, inc);
}

void dnrm2(double *result, double *x, int n, cublasHandle_t &handle) {
  const int inc = 1;
  cublasDnrm2(handle, n, x, inc, result);
}

void dasum(double *result, double *x, int n, cublasHandle_t &handle) {
  const int inc = 1;
  cublasDasum(handle, n, x, inc, result);
}

// BLAS-2: dgemv
void dgemv(double *y, double alpha, double *A, int M, int N, bool transA,
           double *x, double beta, cublasHandle_t &handle) {
  const int inc = 1;
  cublasOperation_t TN = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasDgemv(handle, TN, M, N, &alpha, A, M, x, inc, &beta, y, inc);
}

void dger(double *A, int M, int N, double alpha, double *x, double *y,
          cublasHandle_t &handle) {
  const int inc = 1;
  cublasDger(handle, M, N, &alpha, x, inc, y, inc, A, M);
}

// BLAS-3: dgemm
void dgemm(double *C, double alpha, double *A, bool transA, double *B,
           bool transB, int M, int N, int K, double beta,
           cublasHandle_t &handle) {
  // const double beta = 0.0;
  // switch dimensions with transA and transB
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  // Leading dimensions are the physical row counts of the stored matrices
  int lda = transA ? K : M; // A is (M x K) if !transA, (K x M) if transA
  int ldb = transB ? N : K; // B is (K x N) if !transB, (N x K) if transB
  int ldc = M;              // C is always M x N

  cublasDgemm(handle, opA, opB, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

/*
  kernels shared by Sinkhorn and Barycenter
*/

void init_ones(double *x, int n, cudaStream_t &stream) {
  ip_fill_ones(x, n, stream);
}

void update_K(double *K, int M, int N, double reg, cudaStream_t &stream) {
  ip_gibbs(K, M * N, reg, stream);
}

void normalize(double *x, int n, cublasHandle_t &handle) {
  double norm;
  dasum(&norm, x, n, handle);
  double inv_norm = 1.0 / norm;
  dscal(x, n, inv_norm, handle);
}

