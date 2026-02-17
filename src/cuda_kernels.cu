// implementations of cuda kernels

// #include "check_cuda.hpp" // for checking cuda availability

// #ifdef HAVE_CUBLAS
// #ifdef HAVE_CUDA_RUNTIME

#include "cuda_kernels.cuh"

/*
  CUDA kernels
*/

__global__ void nip_add(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] + y[i];
}

__global__ void nip_minus(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] - y[i];
}

__global__ void nip_dot(int n, double *z, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    z[i] = x[i] * y[i];
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

__global__ void nip_sumsq(int n, double *loss, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double sum = 0.;
  for (int i = index; i < n; i += stride)
    sum += (x[i] - y[i]) * (x[i] - y[i]);
  // *loss = 0.;
  atomicAdd(loss, sum);
}

// __global__ void nip_KTU_w(int m, int n, double *KTU, double *w, double *KTUw)
// {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int stride = blockDim.x * gridDim.x;

//   for (int k = idx; k < m * n; k += stride) {
//     // int i = k % m; // row
//     int j = k / m; // col
//     KTUw[k] = pow(KTU[k], w[j]);
//   }
// }

__global__ void nip_b_div_KTU(int m, int n, double *V, double *b, double *KTU) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int k = idx; k < m * n; k += stride) {
    int i = k % m; // row
    // int j = k / m; // col
    V[k] = b[i] / KTU[k];
  }
}

__global__ void nip_rowprod(int m, int n, double *A, double *result) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m) {
    double prod = 1.0;
    for (int col = 0; col < n; col++) {
      // Column-major: A[row, col] = A[row + col * m]
      prod *= A[row + col * m];
    }
    result[row] = prod;
  }
}

__global__ void nip_rowprod_shared(int m, int n, double *A, double *result) {
  extern __shared__ double sdata[];

  int row = blockIdx.x;
  int tid = threadIdx.x;

  // Initialize shared memory to 1.0 (identity for multiplication)
  sdata[tid] = 1.0;

  // Each thread handles multiple columns
  for (int col = tid; col < n; col += blockDim.x) {
    sdata[tid] *= A[row + col * m]; // Column-major
  }
  __syncthreads();

  // Parallel reduction (multiplication)
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] *= sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    result[row] = sdata[0];
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
__global__ void ip_dot(int n, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] * y[i];
}

__global__ void ip_div(int n, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] / y[i];
}

__global__ void ip_exp(int n, double *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] = exp(x[i]);
}

__global__ void ip_log(int n, double *x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    x[i] = log(x[i]);
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

// kernel: res = x .* y - z
__global__ void ip_dot_minus(int n, double *res, double *x, double *y,
                             double *z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    res[i] = x[i] * y[i] - z[i];
}

// kernel: res = norm2(x .* y - z)
__global__ void ip_dot_minus_sum(int n, double *res, double *x, double *y,
                                 double *z) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double sum = 0.;
  for (int i = index; i < n; i += stride)
    sum += (x[i] * y[i] - z[i]) * (x[i] * y[i] - z[i]);
  atomicAdd(res, sum);
}

// kernel: y = (vbar % v) / y
__global__ void ip_dot_div(int n, double *xbar, double *x, double *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = (xbar[i] * x[i]) / y[i];
}

__global__ void ip_KTU_w(int m, int n, double *KTU, double *w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int k = idx; k < m * n; k += stride) {
    // int i = k % m; // row
    int j = k / m; // col
    KTU[k] = pow(KTU[k], w[j]);
  }
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
  *loss = 0.;
  atomicAdd(loss, sum);
}

/*
  wrappers for the non-inplace kernels
*/

void nip_add(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_add<<<numBlocks, blockSize, 0, stream>>>(n, z, x, y);
}

void nip_minus(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_minus<<<numBlocks, blockSize, 0, stream>>>(n, z, x, y);
}

void nip_dot(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_dot<<<numBlocks, blockSize, 0, stream>>>(n, z, x, y);
}

void nip_div(double *z, double *x, double *y, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_div<<<numBlocks, blockSize, 0, stream>>>(n, z, x, y);
}

void nip_log(double *y, double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_log<<<numBlocks, blockSize, 0, stream>>>(n, y, x);
}

void nip_dot_div(double *z, double *xbar, double *x, double *y, int n,
                 cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_dot_div<<<numBlocks, blockSize, 0, stream>>>(n, xbar, x, y, z);
}

void nip_sumsq(double *loss, double *x, double *y, int n,
               cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  nip_sumsq<<<numBlocks, blockSize, 0, stream>>>(n, loss, x, y);
}

// void nip_KTU_w(double *KTUw, double *KTU, double *w, int m, int n,
//                cudaStream_t &stream) {
//   int blockSize = BLOCK_SIZE;
//   int numBlocks = (m * n + blockSize - 1) / blockSize;

//   nip_KTU_w<<<numBlocks, blockSize, 0, stream>>>(m, n, KTU, w, KTUw);
// }

void nip_b_div_KTU(double *V, double *b, double *KTU, int m, int n,
                   cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  nip_b_div_KTU<<<numBlocks, blockSize, 0, stream>>>(m, n, V, b, KTU);
}

void nip_row_prod(double *x, double *A, int m, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  size_t sharedMemSize = blockSize * sizeof(double);

  nip_rowprod<<<m, blockSize, sharedMemSize, stream>>>(m, n, A, x);
}

void nip_row_prod_shared(double *x, double *A, int m, int n,
                         cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  size_t sharedMemSize = blockSize * sizeof(double);

  nip_rowprod_shared<<<m, blockSize, sharedMemSize, stream>>>(m, n, A, x);
}

void nip_diag_scale(double *P, double *u, double *K, double *v, int m, int n,
                    cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  nip_diag_scale<<<numBlocks, blockSize, 0, stream>>>(m, n, P, u, K, v);
}

void nip_minus_2(double *z, double *x, double *y, int N, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (N + blockSize - 1) / blockSize;

  nip_minus_2<<<numBlocks, blockSize, 0, stream>>>(N, z, x, y);
}

// void nip_outer(double *z, double *x, double *y, int m, int n,
//                cudaHandle_t &handle) {
//   cudaMemset(z, 0, m * n * sizeof(double));
//   dger(z, m, n, 1.0, x, y, handle);
// }

/*
  wrappers for the inplace kernels
*/

void ip_dot(double *y, double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_dot<<<numBlocks, blockSize, 0, stream>>>(n, x, y);
}

void ip_div(double *y, double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_div<<<numBlocks, blockSize, 0, stream>>>(n, x, y);
}

void ip_exp(double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_exp<<<numBlocks, blockSize, 0, stream>>>(n, x);
}

void ip_log(double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_log<<<numBlocks, blockSize, 0, stream>>>(n, x);
}

void ip_fill_ones(double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_fill_ones<<<numBlocks, blockSize, 0, stream>>>(n, x);
}

void ip_gibbs(double *x, int n, double reg, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_gibbs<<<numBlocks, blockSize, 0, stream>>>(n, x, reg);
}

void ip_dot_minus(double *res, double *x, double *y, double *z, int n,
                  cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_dot_minus<<<numBlocks, blockSize, 0, stream>>>(n, res, x, y, z);
}

void ip_dot_minus_sum(double *res, double *x, double *y, double *z, int n,
                      cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_dot_minus_sum<<<numBlocks, blockSize, 0, stream>>>(n, res, x, y, z);
}

void ip_dot_div(double *y, double *xbar, double *x, int n,
                cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_dot_div<<<numBlocks, blockSize, 0, stream>>>(n, xbar, x, y);
}

void ip_KTU_w(double *KTU, double *w, int m, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  ip_KTU_w<<<numBlocks, blockSize, 0, stream>>>(m, n, KTU, w);
}

void ip_PbarK(double *PbarK, double *C, double *P, double *K, int m, int n,
              double reg, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  ip_PbarK<<<numBlocks, blockSize, 0, stream>>>(m * n, PbarK, C, P, K, reg);
}

void ip_accu_abar(double *abar, double *ubar, double *Kv, int n,
                  cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_accu_abar<<<numBlocks, blockSize, 0, stream>>>(n, abar, ubar, Kv);
}

void ip_sinkloss(double *loss, double *C, double *P, int m, int n, double reg,
                 cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  ip_sinkloss<<<numBlocks, blockSize, 0, stream>>>(m * n, loss, C, P, reg);
}

/*
  BLAS wrappers
*/

// BLAS-1: dnrm2
void dscal(double *y, int n, double alpha, cublasHandle_t &handle) {
  const int inc = 1;
  cublasDscal(handle, n, &alpha, y, inc);
}

void ddot(double *result, double *x, double *y, int n, cublasHandle_t &handle) {
  const int inc = 1;
  cublasDdot(handle, n, x, inc, y, inc, result);
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

// #endif
// #endif
