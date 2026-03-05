// cuda kernels for WDL

#include <chrono>
#include <random>

#include <R.h>
#include <Rinternals.h>

#include "cuda_kernels.cuh"

#include "cuda_interface.cuh"

// check for pending R user interrupt without longjmp
static void check_interrupt_fn(void *dummy) { R_CheckUserInterrupt(); }
static bool check_interrupt() {
  return !R_ToplevelExec(check_interrupt_fn, NULL);
}

// CUDA error checking macro — reports and jumps to cleanup on failure
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      REprintf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                \
               cudaGetErrorString(err));                                       \
      goto cleanup;                                                            \
    }                                                                          \
  } while (0)

/*
  WDL kernels
*/

// elementwise diag(a) * b: out[i] = a[i] * b[i]
// launch with ceil(n / blockDim.x) blocks, blockDim.x threads
__global__ void diagmul(int n, double *out, double *a, double *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] * b[i];
}

// column-wise softmax: out[:,s] = softmax(in[:,s])
// launch with ncols blocks, BLOCK_SIZE threads, shared mem = BLOCK_SIZE *
// sizeof(double)
__global__ void softmax(int nrows, double *out, double *in) {
  int s = blockIdx.x;
  extern __shared__ double sdata[];

  double *in_col = in + s * nrows;
  double *out_col = out + s * nrows;

  // phase 1: find column max
  double local_max = -1e300;
  for (int i = threadIdx.x; i < nrows; i += blockDim.x)
    local_max = fmax(local_max, in_col[i]);
  sdata[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      sdata[threadIdx.x] =
          fmax(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
    __syncthreads();
  }
  double max_val = sdata[0];

  // phase 2: exp(x - max) and sum
  double local_sum = 0.0;
  for (int i = threadIdx.x; i < nrows; i += blockDim.x) {
    out_col[i] = exp(in_col[i] - max_val);
    local_sum += out_col[i];
  }
  sdata[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      sdata[threadIdx.x] += sdata[threadIdx.x + stride];
    __syncthreads();
  }
  double sum_val = sdata[0];

  // phase 3: normalize
  for (int i = threadIdx.x; i < nrows; i += blockDim.x)
    out_col[i] /= sum_val;
}

// batched tiled division: U[i, d*S+s] = A[i, s] / KV[i, d*S+s]
// A is N x S (shared across all D docs), KV is N x (S*D)
__global__ void batched_A_div_KV(int N, int S, int D, double *U, double *A,
                                 double *KV) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * S * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N;  // row
    int ds = k / N; // column in stacked matrix
    int s = ds % S; // topic index (tiles with period S)
    U[k] = A[i + s * N] / KV[k];
  }
}

// batched b broadcast division: V[i, d*S+s] = b[i, d] / KTU[i, d*S+s]
// b is N x D, KTU is N x (S*D)
__global__ void batched_b_div_KTU(int N, int S, int D, double *V, double *b,
                                  double *KTU) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * S * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N;  // row
    int ds = k / N; // column in stacked matrix
    int d = ds / S; // doc index
    V[k] = b[i + d * N] / KTU[k];
  }
}

// batched row product per doc: b[i, d] = prod_s KTU[i, d*S+s]
// KTU is N x (S*D) (already powered), b is N x D
// S is small so a serial loop per (i,d) pair is efficient
__global__ void batched_row_prod(int N, int S, int D, double *b, double *KTU) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N; // row
    int d = k / N; // doc
    double prod = 1.0;
    for (int s = 0; s < S; s++)
      prod *= KTU[i + (d * S + s) * N];
    b[i + d * N] = prod;
  }
}

// batched row sum per doc: bbar[i, d] = sum_s X[i, d*S+s]
// X is N x (S*D), bbar is N x D
// S is small so a serial loop per (i,d) pair is efficient
__global__ void batched_row_sum(int N, int S, int D, double *bbar, double *X) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N; // row
    int d = k / N; // doc
    double sum = 0.0;
    for (int s = 0; s < S; s++)
      sum += X[i + (d * S + s) * N];
    bbar[k] = sum;
  }
}

// batched outer product: out[i, d*S+s] = bbar[i, d] * w[s, d]
// bbar is N x D, w is S x D (column-major), out is N x (S*D)
__global__ void batched_bbar_wT(int N, int S, int D, double *out, double *bbar,
                                double *w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * S * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N;  // row
    int ds = k / N; // column in stacked matrix
    int s = ds % S; // topic index
    int d = ds / S; // doc index
    out[k] = bbar[i + d * N] * w[s + d * S];
  }
}

// batched minus outer product
__global__ void batched_div_bbar_wT(int N, int S, int D, double *Vbar,
                                    double *KTU, double *bbar, double *w) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = N * S * D;
  for (int k = idx; k < total; k += stride) {
    int i = k % N;  // row
    int ds = k / N; // column in stacked matrix
    int s = ds % S; // topic index
    int d = ds / S; // doc index
    KTU[k] = Vbar[k] / KTU[k] - bbar[i + d * N] * w[s + d * S];
  }
}

// replicate a column vector to all columns of a matrix (column-major)
// out[:,j] = col for all j in 0..ncols-1
__global__ void replicate_col(int nrows, int ncols, double *out, double *col) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int n = nrows * ncols;
  for (int k = idx; k < n; k += stride) {
    int r = k % nrows;
    out[k] = col[r];
  }
}

/*
  Wrapper functions for WDL kernels
*/

void diagmul(double *out, double *a, double *b, int n, cudaStream_t &stream) {
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  diagmul<<<numBlocks, BLOCK_SIZE, 0, stream>>>(n, out, a, b);
}

// wrapper for softmax
void softmax(double *out, double *in, int nrows, int ncols) {
  softmax<<<ncols, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(nrows, out, in);
}

// Jacobian of softmax is diag(a) - a*a^T
// alphabar = diag(a) * abar + (-1) * dot(a, abar) * a
void update_softmax_jac(double *alphabar, double *abar, double *a, int n,
                        cudaStream_t &stream, cublasHandle_t &handle) {
  // alphabar = diag(a) * abar
  diagmul(alphabar, a, abar, n, stream);
  // temp = dot(a, abar)
  double temp;
  ddot(&temp, a, abar, n, handle);
  // alphabar += (-1) * temp * a
  daxpy(alphabar, -temp, a, n, handle);
}

// batched outer product: out[i, d*S+s] = bbar[i,d] * w[s,d]
// bbar: N x D, w: S x D, out: N x (S*D)
void nip_bbarwT(double *out, double *bbar, double *w, int N, int S, int D,
                cudaStream_t &stream) {
  int total = N * S * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_bbar_wT<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, out, bbar, w);
}

void ip_div_bbarwT(double *KTU, double *Vbar, double *bbar, double *w, int N,
                   int S, int D, cudaStream_t &stream) {
  int total = N * S * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_div_bbar_wT<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, Vbar, KTU,
                                                            bbar, w);
}

// batched KV: KVB = K * VB
// K: N x N, VB: N x (S*D), KVB: N x (S*D)
void update_KVB(double *KVB, double *K, double *VB, int N, int S, int D,
                cublasHandle_t &handle) {
  dgemm(KVB, 1.0, K, false, VB, false, N, S * D, N, 0.0, handle);
}

// batched KTU: KTUB = K^T * UB
// K: N x N, UB: N x (S*D), KTUB: N x (S*D)
void update_KTUB(double *KTUB, double *K, double *UB, int N, int S, int D,
                 cublasHandle_t &handle) {
  dgemm(KTUB, 1.0, K, true, UB, false, N, S * D, N, 0.0, handle);
}

// batched U update: UB[i, d*S+s] = A[i, s] / KVB[i, d*S+s]
// A: N x S (tiled), KVB: N x (S*D), UB: N x (S*D)
void update_UB(double *UB, double *A, double *KVB, int N, int S, int D,
               cudaStream_t &stream) {
  int total = N * S * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_A_div_KV<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, UB, A, KVB);
}

// batched b update: power KTU in-place, then row product per doc
// KTUB is N x (S*D), wB is S x D, bB is N x D
// NOTE: this modifies KTUB in-place (power step) — save to history first!
void update_bB(double *bB, double *KTUB, double *wB, int N, int S, int D,
               cudaStream_t &stream) {
  // step 1: in-place power — KTUB[i, d*S+s] ^= wB[s, d]
  // reuse existing ip_KTU_w: w[j] where j is column index in N x (S*D)
  // j = d*S+s maps to wB[s + d*S] = wB[s, d] in column-major — correct!
  ip_KTU_w(KTUB, wB, N, S * D, stream);

  // step 2: row product per doc — bB[i, d] = prod_s KTUB[i, d*S+s]
  int total = N * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_row_prod<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, bB, KTUB);
}

// batched bbar at final iteration: bBbar[i,d] = 2*(bB[i,d] - b_ext[i,d])
// bB: N x D (barycenters at iter L), b_ext: N x D (target docs)
// both are contiguous N*D blocks, so reuse nip_minus_2 directly
void update_bBbar_L(double *bBbar, double *bB, double *bB_ext, int N, int D,
                    cudaStream_t &stream) {
  nip_minus_2(bBbar, bB, bB_ext, N * D, stream);
}

void update_UBbar_L(double *UBbar, double *bBbar, double *VBhist, int l,
                    double *wB, double *K, double *KTUB, int N, int S, int D,
                    cudaStream_t &stream, cublasHandle_t &handle) {
  // implement `bbarwT` kernel
  nip_bbarwT(KTUB, bBbar, wB, N, S, D, stream);
  ip_dot(KTUB, VBhist + (size_t)l * N * S * D, N * S * D, stream);
  dgemm(UBbar, 1.0, K, false, KTUB, false, N, S * D, N, 0.0, handle);
}

void update_VBbar_l(double *VBbar, double *UBbar, double *UBhist,
                    double *VBhist, double *KVBhist, int l, double *K,
                    double *KVB, int N, int S, int D, cudaStream_t &stream,
                    cublasHandle_t &handle) {
  nip_dot_div(KVB, UBbar, UBhist + (size_t)(l + 1) * N * S * D,
              KVBhist + (size_t)l * N * S * D, N * S * D, stream);
  dgemm(VBbar, -1.0, K, true, KVB, false, N, S * D, N, 0.0, handle);
}

// batched bbar at iteration l: bBbar[i,d] = sum_s (VBbar[i,d*S+s] /
// KTUBhist_l[i,d*S+s]) VBbar: N x (S*D), KTUBhist_l: N x (S*D), KTUB: N x (S*D)
// temp, bBbar: N x D
void update_bBbar_l(double *bBbar, double *VBbar, double *KTUBhist, int l,
                    double *KTUB, int N, int S, int D, cudaStream_t &stream) {
  // step 1: KTUB = VBbar / KTUBhist_l (elementwise)
  nip_div(KTUB, VBbar, KTUBhist + (size_t)l * N * S * D, N * S * D, stream);
  // step 2: bBbar[i,d] = sum_s KTUB[i, d*S+s] (batched row sum)
  int total = N * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_row_sum<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, bBbar, KTUB);
}

void update_UBbar_l(double *UBbar, double *VBbar, double *bBbar, double *VBhist,
                    double *KTUBhist, int l, double *wB, double *K,
                    double *KTUB, int N, int S, int D, cudaStream_t &stream,
                    cublasHandle_t &handle) {
  nip_div(KTUB, VBbar, KTUBhist + (size_t)l * N * S * D, N * S * D, stream);
  for (int d = 0; d < D; d++)
    dger(KTUB + d * N * S, N, S, -1.0, bBbar + d * N, wB + d * S, handle);
  ip_dot(KTUB, VBhist + (size_t)l * N * S * D, N * S * D, stream);
  dgemm(UBbar, -1.0, K, false, KTUB, false, N, S * D, N, 0.0, handle);
}

// batched V update: VB[i, d*S+s] = bB[i, d] / KTUB[i, d*S+s]
// bB: N x D, KTUB: N x (S*D), VB: N x (S*D)
void update_VB(double *VB, double *bB, double *KTUB, int N, int S, int D,
               cudaStream_t &stream) {
  int total = N * S * D;
  int numBlocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
  batched_b_div_KTU<<<numBlocks, BLOCK_SIZE, 0, stream>>>(N, S, D, VB, bB,
                                                          KTUB);
}

// batched Abar: ABbar (N x S*D) += UBbar / KVBhist[l-1]
void update_ABbar(double *ABbar, double *UBbar, double *KVBhist, int l,
                  double *KVB, int N, int S, int D, cudaStream_t &stream) {
  nip_div(KVB, UBbar, KVBhist + (size_t)(l - 1) * N * S * D, N * S * D, stream);
  ip_add(ABbar, KVB, N * S * D, stream);
}

// batched wbar: for each d, wbar_d += logKTUB_d^T * (bBbar_d * bBhist_d)
void update_wBbar(double *wbar, double *bBbar, double *bBhist, double *KTUBhist,
                  int l, double *KTUB, int N, int S, int D,
                  cudaStream_t &stream, cublasHandle_t &handle) {
  // KTUB = log(KTUBhist_l)
  nip_log(KTUB, KTUBhist + (size_t)l * N * S * D, N * S * D, stream);
  // bBbar *= bBhist_l
  ip_dot(bBbar, bBhist + (size_t)l * N * D, N * D, stream);
  // wbar_d += KTUB_d^T * bBbar_d for each d
  for (int d = 0; d < D; d++)
    dgemv(wbar + d * S, 1.0, KTUB + d * N * S, N, S, true, bBbar + d * N, 1.0,
          handle);
}

// column-wise softmax: out = softmax(in), each column independently
void softmax(double *out, double *in, int nrows, int ncols,
             cudaStream_t &stream) {
  size_t sharedMem = BLOCK_SIZE * sizeof(double);
  softmax<<<ncols, BLOCK_SIZE, sharedMem, stream>>>(nrows, out, in);
}

// broadcast: out (nrows x ncols) = col * ones^T
void replicate_col(double *out, double *col, int nrows, int ncols,
                   cudaStream_t &stream) {
  int n = nrows * ncols;
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  replicate_col<<<numBlocks, BLOCK_SIZE, 0, stream>>>(nrows, ncols, out, col);
}

/*
  wdl_batch: process one batch of D documents simultaneously

  Uses batched barycenter forward + backward with fixed iterations.
  All D documents share the same A (topics) and K (kernel) but have
  different weights wB (S x D) and targets bB_ext (N x D).

  Dimensions (WDL convention):
    N = vocab size, M = number of docs, S = number of topics
    D = n_docs in this batch (<= B)
  The barycenter uses a square cost matrix (N x N).
*/

void wdl_batch(
    // gradient outputs (zeroed internally)
    double *g_Alpha,  // N x S
    double *g_lambda, // S
    double *g_Lambda, // S x M
    // model data (read-only in batch)
    double *A, // N x S (topics, after softmax)
    double *W, // S x M (weights, after softmax)
    double *Y, // N x M (documents)
    double *K, // N x N (Gibbs kernel)
    // batched scratch (sized for B docs)
    double *UB, double *VB, double *bB, // N*(S*B), N*(S*B), N*B
    double *KVB, double *KTUB,          // N*(S*B), N*(S*B)
    // batched history buffers
    double *UB_hist, double *VB_hist, double *bB_hist, double *KVB_hist,
    double *KTUB_hist,
    // adjoint buffers
    double *UBbar, double *VBbar, double *bBbar, // N*(S*B), N*(S*B), N*B
    double *ABbar, double *wBbar,                // N*S, S*B
    // params
    int batch_id, int B, int M, int N, int S, int max_iter,
    cudaStream_t &stream, cublasHandle_t &handle) {

  auto D2D = cudaMemcpyDeviceToDevice;

  // number of docs in this batch
  int D = (batch_id == (M / B)) ? (M % B) : B;
  if (D <= 0)
    return;

  // contiguous column pointers into W and Y for this batch (no copy needed)
  double *wB = W + batch_id * B * S;     // S x D
  double *bB_ext = Y + batch_id * B * N; // N x D

  // zero gradient accumulators
  cudaMemsetAsync(g_Alpha, 0, sizeof(double) * N * S, stream);
  cudaMemsetAsync(g_lambda, 0, sizeof(double) * S, stream);

  // zero adjoint buffers
  cudaMemsetAsync(ABbar, 0, sizeof(double) * N * S * D, stream);
  cudaMemsetAsync(wBbar, 0, sizeof(double) * S * D, stream);
  cudaMemsetAsync(UBbar, 0, sizeof(double) * N * S * D, stream);
  cudaMemsetAsync(VBbar, 0, sizeof(double) * N * S * D, stream);
  cudaMemsetAsync(bBbar, 0, sizeof(double) * N * D, stream);

  // init scaling vectors to ones
  init_ones(UB, N * S * D, stream);
  init_ones(VB, N * S * D, stream);

  // save initial history
  cudaMemcpyAsync(UB_hist, UB, sizeof(double) * N * S * D, D2D, stream);
  cudaMemcpyAsync(VB_hist, VB, sizeof(double) * N * S * D, D2D, stream);

  // ---- FORWARD: fixed max_iter iterations ----
  for (int l = 0; l < max_iter; ++l) {
    // KV = K * V
    update_KVB(KVB, K, VB, N, S, D, handle);
    cudaMemcpyAsync(KVB_hist + (size_t)l * N * S * D, KVB,
                    sizeof(double) * N * S * D, D2D, stream);

    // U = A / KV
    update_UB(UB, A, KVB, N, S, D, stream);
    cudaMemcpyAsync(UB_hist + (size_t)(l + 1) * N * S * D, UB,
                    sizeof(double) * N * S * D, D2D, stream);

    // KTU = K^T * U
    update_KTUB(KTUB, K, UB, N, S, D, handle);
    cudaMemcpyAsync(KTUB_hist + (size_t)(l + 1) * N * S * D, KTUB,
                    sizeof(double) * N * S * D, D2D, stream);

    // b = row_prod(KTU^w)  — modifies KTUB in-place (power step)
    update_bB(bB, KTUB, wB, N, S, D, stream);
    cudaMemcpyAsync(bB_hist + (size_t)(l + 1) * N * D, bB,
                    sizeof(double) * N * D, D2D, stream);

    // V = b / KTU  — use pre-power KTU from history
    update_VB(VB, bB, KTUB_hist + (size_t)(l + 1) * N * S * D, N, S, D, stream);
    cudaMemcpyAsync(VB_hist + (size_t)(l + 1) * N * S * D, VB,
                    sizeof(double) * N * S * D, D2D, stream);
  }

  // ---- BACKWARD: l = max_iter down to 1 ----
  for (int l = max_iter; l > 0; --l) {
    if (l == max_iter) {
      // bBbar = 2*(bBhist[l] - bB_ext)
      update_bBbar_L(bBbar, bB_hist + (size_t)l * N * D, bB_ext, N, D, stream);
      // UBbar = K * (bBbar*wB^T % VBhist[l])
      update_UBbar_L(UBbar, bBbar, VB_hist, l, wB, K, KTUB, N, S, D, stream,
                     handle);
    } else {
      // VBbar = -K^T * (UBbar * UBhist[l+1] / KVBhist[l])
      update_VBbar_l(VBbar, UBbar, UB_hist, VB_hist, KVB_hist, l, K, KVB, N, S,
                     D, stream, handle);
      // bBbar = row_sum(VBbar / KTUBhist[l])
      update_bBbar_l(bBbar, VBbar, KTUB_hist, l, KTUB, N, S, D, stream);
      // UBbar = -K * (VBbar/KTUBhist[l] - bBbar*wB^T) % VBhist[l]
      update_UBbar_l(UBbar, VBbar, bBbar, VB_hist, KTUB_hist, l, wB, K, KTUB, N,
                     S, D, stream, handle);
    }

    // ABbar += UBbar / KVBhist[l-1]  (accumulates over D docs into N x S)
    update_ABbar(ABbar, UBbar, KVB_hist, l, KVB, N, S, D, stream);
    // wBbar: log(KTUBhist[l])^T * (bBbar * bBhist[l])
    update_wBbar(wBbar, bBbar, bB_hist, KTUB_hist, l, KTUB, N, S, D, stream,
                 handle);
  }

  // ---- GRADIENT ACCUMULATION ----
  // softmax Jacobian: chain rule through softmax(Alpha) → A
  // UB is used as intermediate; loop through D * S columns
  for (int j = 0; j < D * S; j++) {
    int s = j % S;
    update_softmax_jac(UB + j * N, ABbar + j * N, A + s * N, N, stream, handle);
  }

  // softmax Jacobian: chain rule through softmax(Lambda) → W
  // use bBbar as intermediate; loop through D docs
  for (int d = 0; d < D; d++) {
    update_softmax_jac(bBbar + d * S, wBbar + d * S, wB + d * S, S, stream,
                       handle);
  }

  // reduce UB from N x (S*D) to N x S by summing D blocks into block 0
  for (int d = 1; d < D; d++)
    ip_add(UB, UB + d * N * S, N * S, stream);

  // reduce bBbar from S x D to S by summing D blocks into block 0
  for (int d = 1; d < D; d++)
    ip_add(bBbar, bBbar + d * S, S, stream);

  // g_Alpha = UB / D
  double inv_D = 1.0 / D;
  cudaMemcpyAsync(g_Alpha, UB, sizeof(double) * N * S, D2D, stream);
  dscal(g_Alpha, N * S, inv_D, handle);

  // g_lambda = bBbar / D
  cudaMemcpyAsync(g_lambda, bBbar, sizeof(double) * S, D2D, stream);
  dscal(g_lambda, S, inv_D, handle);

  // broadcast: g_Lambda (S x M) = g_lambda * ones^T
  replicate_col(g_Lambda, g_lambda, S, M, stream);
}

/*
  cuda_wdl: Wasserstein Dictionary Learning on GPU

  Mirrors _compute_serial() in wdl_impl.cpp.
  Manages all device memory, runs the full training loop + inference.

  Dimensions (WDL convention):
    N = vocab size, M = number of docs, S = number of topics
  The barycenter uses a square cost matrix (N x N).
*/

void cuda_wdl(
    // outputs (host, pre-allocated by caller)
    double *A,    // N x S (topics)
    double *W,    // S x M (weights)
    double *Yhat, // N x M (predicted barycenters)
    // inputs (host)
    const double *Y, // N x M (documents)
    const double *C, // N x N (cost matrix)
    // dimensions
    const int N, const int M, const int S,
    // barycenter params
    const double reg, const int max_iter, const double zero_tol,
    // training params
    const int B, const int E,
    // optimizer params: opt = 0 (SGD), 1 (Adam), 2 (AdamW)
    const int opt, const double eta, const double gamma, const double beta1,
    const double beta2, const double eps, const int seed, const bool verbose) {

  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: declare device pointers */
  // data
  double *d_Y = nullptr, *d_K = nullptr;
  // latent vars
  double *d_Alpha = nullptr, *d_Lambda = nullptr;
  double *d_A = nullptr, *d_W = nullptr;
  // gradients
  double *d_g_Alpha = nullptr, *d_g_lambda = nullptr, *d_g_Lambda = nullptr;
  // optimizer momentum buffers
  double *d_m_Alpha = nullptr, *d_v_Alpha = nullptr;
  double *d_m_Lambda = nullptr, *d_v_Lambda = nullptr;
  // batched barycenter scratch (sized for B docs)
  double *d_UB = nullptr, *d_VB = nullptr, *d_bB = nullptr;
  double *d_KVB = nullptr, *d_KTUB = nullptr;
  // batched history buffers
  double *d_UB_hist = nullptr, *d_VB_hist = nullptr, *d_bB_hist = nullptr;
  double *d_KVB_hist = nullptr, *d_KTUB_hist = nullptr;
  // batched adjoint buffers
  double *d_UBbar = nullptr, *d_VBbar = nullptr, *d_bBbar = nullptr;
  double *d_ABbar = nullptr, *d_wBbar = nullptr;
  // inference output
  double *d_Yhat = nullptr;

  // variables declared here so goto cleanup doesn't bypass initialization
  std::mt19937 rng(seed);
  std::normal_distribution<double> randn(0.0, 1.0);
  double *h_init = nullptr;
  int L = max_iter + 1; // history depth
  int batches = (M % B) ? (M / B + 1) : (M / B);
  int step = 0;

  /* step 3: allocate device memory */
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0);

  // data
  CUDA_CHECK(cudaMallocAsync((void **)&d_Y, sizeof(double) * N * M, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_K, sizeof(double) * N * N, stream));
  // latent vars
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_Alpha, sizeof(double) * N * S, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_Lambda, sizeof(double) * S * M, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_A, sizeof(double) * N * S, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_W, sizeof(double) * S * M, stream));
  // gradients
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_g_Alpha, sizeof(double) * N * S, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_g_lambda, sizeof(double) * S, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_g_Lambda, sizeof(double) * S * M, stream));
  // optimizer momentum buffers (zeroed)
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_m_Alpha, sizeof(double) * N * S, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_v_Alpha, sizeof(double) * N * S, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_m_Lambda, sizeof(double) * S * M, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_v_Lambda, sizeof(double) * S * M, stream));
  cudaMemsetAsync(d_m_Alpha, 0, sizeof(double) * N * S, stream);
  cudaMemsetAsync(d_v_Alpha, 0, sizeof(double) * N * S, stream);
  cudaMemsetAsync(d_m_Lambda, 0, sizeof(double) * S * M, stream);
  cudaMemsetAsync(d_v_Lambda, 0, sizeof(double) * S * M, stream);
  // batched barycenter scratch (sized for B docs)
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_UB, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_VB, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_bB, sizeof(double) * N * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_KVB, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_KTUB, sizeof(double) * N * S * B, stream));
  // batched history buffers
  CUDA_CHECK(cudaMallocAsync((void **)&d_UB_hist,
                             sizeof(double) * L * N * S * B, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_VB_hist,
                             sizeof(double) * L * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_bB_hist, sizeof(double) * L * N * B, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_KVB_hist,
                             sizeof(double) * L * N * S * B, stream));
  CUDA_CHECK(cudaMallocAsync((void **)&d_KTUB_hist,
                             sizeof(double) * L * N * S * B, stream));
  // batched adjoint buffers
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_UBbar, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_VBbar, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_bBbar, sizeof(double) * N * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_ABbar, sizeof(double) * N * S * B, stream));
  CUDA_CHECK(
      cudaMallocAsync((void **)&d_wBbar, sizeof(double) * S * B, stream));
  // inference output
  CUDA_CHECK(cudaMallocAsync((void **)&d_Yhat, sizeof(double) * N * M, stream));

  /* step 4: initialize data on device */

  // copy Y to device
  cudaMemcpyAsync(d_Y, Y, sizeof(double) * N * M, H2D, stream);

  // compute Gibbs kernel K from C
  cudaMemcpyAsync(d_K, C, sizeof(double) * N * N, H2D, stream);
  update_K(d_K, N, N, reg, stream);

  // init Alpha and Lambda with random Gaussian values
  h_init = (double *)malloc(sizeof(double) * N * S);
  for (int i = 0; i < N * S; ++i)
    h_init[i] = randn(rng);
  cudaMemcpyAsync(d_Alpha, h_init, sizeof(double) * N * S, H2D, stream);
  cudaStreamSynchronize(stream);
  free(h_init);

  h_init = (double *)malloc(sizeof(double) * S * M);
  for (int i = 0; i < S * M; ++i)
    h_init[i] = randn(rng);
  cudaMemcpyAsync(d_Lambda, h_init, sizeof(double) * S * M, H2D, stream);
  cudaStreamSynchronize(stream);
  free(h_init);

  // initial softmax: Alpha → A, Lambda → W
  softmax(d_A, d_Alpha, N, S, stream);
  softmax(d_W, d_Lambda, S, M, stream);

  cudaStreamSynchronize(stream);

  /* step 5: training loop */

  if (verbose) {
    Rprintf(
        "Initializing WDL model with %d vocabs, %d docs, and %d topics...\n", N,
        M, S);
    Rprintf("Training WDL model with %d epochs, %d batches\n", E, batches);
  }

  for (int e = 0; e < E; ++e) {
    for (int batch_id = 0; batch_id < batches; ++batch_id) {

      // check for user interrupt
      cudaStreamSynchronize(stream);
      if (check_interrupt()) {
        if (verbose)
          Rprintf("User interrupt detected, cleaning up...\n");
        goto cleanup;
      }

      auto t1 = std::chrono::steady_clock::now();

      if (verbose) {
        Rprintf("Epoch %d of %d, batch %d of %d\n", e + 1, E, batch_id + 1,
                batches);
      }

      // compute batch-averaged gradients
      wdl_batch(d_g_Alpha, d_g_lambda, d_g_Lambda, d_A, d_W, d_Y, d_K, d_UB,
                d_VB, d_bB, d_KVB, d_KTUB, d_UB_hist, d_VB_hist, d_bB_hist,
                d_KVB_hist, d_KTUB_hist, d_UBbar, d_VBbar, d_bBbar, d_ABbar,
                d_wBbar, batch_id, B, M, N, S, max_iter, stream, handle);

      // optimizer step
      step++;
      optimizer_step(d_Alpha, d_g_Alpha, d_m_Alpha, d_v_Alpha, opt, N * S, eta,
                     gamma, beta1, beta2, eps, step, stream);
      optimizer_step(d_Lambda, d_g_Lambda, d_m_Lambda, d_v_Lambda, opt, S * M,
                     eta, gamma, beta1, beta2, eps, step, stream);

      // update A and W via softmax
      softmax(d_A, d_Alpha, N, S, stream);
      softmax(d_W, d_Lambda, S, M, stream);

      if (verbose) {
        cudaStreamSynchronize(stream);
        auto t2 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t2 - t1).count();
        Rprintf("  batch time: %.2f sec\n", sec);
      }
    }
  }

  /* step 6: inference — compute predicted barycenters */
  if (verbose)
    Rprintf("Inference on the dataset\n");

  for (int m = 0; m < M; ++m) {
    double *d_w_m = d_W + m * S;

    // reset scaling vectors (reuse batched buffers for single doc)
    init_ones(d_UB, N * S, stream);
    init_ones(d_VB, N * S, stream);

    int iter = 0;
    double err = 1000.0;

    // forward only (withgrad = false) — history/adjoint buffers not accessed
    impl_barycenter(iter, err, d_UB, d_VB, d_bB, d_UBbar, d_VBbar, d_bBbar,
                    d_ABbar, d_wBbar, d_UB_hist, d_VB_hist, d_bB_hist,
                    d_KVB_hist, d_KTUB_hist, d_A, d_w_m, d_Y + m * N, d_K,
                    d_KVB, d_KTUB, N, N, S, max_iter, zero_tol, false, stream,
                    handle);

    // store barycenter into Yhat column
    cudaMemcpyAsync(d_Yhat + m * N, d_bB, sizeof(double) * N, D2D, stream);

    if (verbose && ((m + 1) % B == 0 || m == M - 1))
      Rprintf("  Inference: %d of %d docs done\n", m + 1, M);
  }

  /* step 7: copy results to host */
  cudaMemcpyAsync(A, d_A, sizeof(double) * N * S, D2H, stream);
  cudaMemcpyAsync(W, d_W, sizeof(double) * S * M, D2H, stream);
  cudaMemcpyAsync(Yhat, d_Yhat, sizeof(double) * N * M, D2H, stream);
  cudaStreamSynchronize(stream);

cleanup:
  /* step 8: free all device memory */
  cudaFreeAsync(d_Y, stream);
  cudaFreeAsync(d_K, stream);
  cudaFreeAsync(d_Alpha, stream);
  cudaFreeAsync(d_Lambda, stream);
  cudaFreeAsync(d_A, stream);
  cudaFreeAsync(d_W, stream);
  cudaFreeAsync(d_g_Alpha, stream);
  cudaFreeAsync(d_g_lambda, stream);
  cudaFreeAsync(d_g_Lambda, stream);
  cudaFreeAsync(d_m_Alpha, stream);
  cudaFreeAsync(d_v_Alpha, stream);
  cudaFreeAsync(d_m_Lambda, stream);
  cudaFreeAsync(d_v_Lambda, stream);
  cudaFreeAsync(d_UB, stream);
  cudaFreeAsync(d_VB, stream);
  cudaFreeAsync(d_bB, stream);
  cudaFreeAsync(d_KVB, stream);
  cudaFreeAsync(d_KTUB, stream);
  cudaFreeAsync(d_UB_hist, stream);
  cudaFreeAsync(d_VB_hist, stream);
  cudaFreeAsync(d_bB_hist, stream);
  cudaFreeAsync(d_KVB_hist, stream);
  cudaFreeAsync(d_KTUB_hist, stream);
  cudaFreeAsync(d_UBbar, stream);
  cudaFreeAsync(d_VBbar, stream);
  cudaFreeAsync(d_bBbar, stream);
  cudaFreeAsync(d_ABbar, stream);
  cudaFreeAsync(d_wBbar, stream);
  cudaFreeAsync(d_Yhat, stream);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}
