// implementation of the CUDA Barycenter interface
// `cuda_barycenter_parallel`

// #include "check_cuda.hpp" // for checking cuda availability

#ifdef HAVE_CUBLAS
#ifdef HAVE_CUDA_RUNTIME

#include "cuda_kernels.cuh"

#include "cuda_interface.cuh"

/*
  helper functions: now using the wrappers in cuda_kernels.cuh
*/

void update_KV(double *KV, double *K, double *V, int M, int N, int S,
               cublasHandle_t &handle) {
  // KV = K * V
  dgemm(KV, 1.0, K, false, V, false, M, S, N, 0.0, handle);
}

void update_U(double *U, double *A, double *KV, int M, int S,
              cudaStream_t &stream) {
  // U = A / KV
  nip_div(U, A, KV, M * S, stream);
}

void update_KTU(double *KTU, double *K, double *U, int N, int S, int M,
                cublasHandle_t &handle) {
  // KTU = K^T * U
  dgemm(KTU, 1.0, K, true, U, false, N, S, M, 0.0, handle);
}

void update_b(double *b, double *U, double *K, double *w, double *KTU, int M,
              int N, int S, cudaStream_t &stream, cublasHandle_t &handle) {
  // raise power
  ip_KTU_w(KTU, w, N, S, stream);
  // nip_KTU_w(tmp_NS, KTU, w, N, S, stream);
  // nip_row_prod(b, tmp_NS, N, S, stream);
  nip_row_prod_shared(b, KTU, N, S, stream);
  // cudaStreamSynchronize(stream);
}

void update_V(double *V, double *b, double *KTUhist, int l, int N, int S,
              cudaStream_t &stream) {
  // V = b / KTU
  nip_b_div_KTU(V, b, KTUhist + (l + 1) * N * S, N, S, stream);
}

void update_bbar_L(double *bbar, double *bhist, int l, double *b_ext, int N,
                   cudaStream_t &stream) {
  // nip_minus(bbar, bhist + l * N, b_ext, N, stream);
  // dscal(bbar, N, 2.0, handle);
  nip_minus_2(bbar, bhist + l * N, b_ext, N, stream);
}

void update_Ubar_L(double *Ubar, double *bbar, double *Vhist, int l, double *w,
                   double *K, double *KTU, int M, int N, int S,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  // bbar * w^T
  cudaMemset(KTU, 0, N * S * sizeof(double));
  dger(KTU, N, S, 1.0, bbar, w, handle);
  // tmp_NS = V_L % tmp_NS
  ip_dot(KTU, Vhist + l * N * S, N * S, stream);
  // Ubar = K * tmp_NS
  dgemm(Ubar, 1.0, K, false, KTU, false, M, S, N, 0.0, handle);
}

void update_Vbar_l(double *Vbar, double *Ubar, double *Uhist, double *Vhist,
                   double *KVhist, int l, double *K, double *KV, int M, int N,
                   int S, cudaStream_t &stream, cublasHandle_t &handle) {
  // KV already computed in `update_Abar`
  // dgemm(KV, 1.0, K, false, Vhist + l * N * S, false, M, S, N, 0.0, handle);
  // KV = Ubar / KV
  nip_dot_div(KV, Ubar, Uhist + (l + 1) * M * S, KVhist + l * M * S, M * S,
              stream);
  // ip_dot_div(KV, Ubar, Uhist + (l + 1) * M * S, M * S, stream);
  // Vbar = -K^T * KV
  dgemm(Vbar, -1.0, K, true, KV, false, N, S, M, 0.0, handle);
}

void update_bbar_l(double *bbar, double *Vbar, double *Uhist, double *KTUhist,
                   int l, double *K, double *KTU, double *ones_S, int M, int N,
                   int S, cudaStream_t &stream, cublasHandle_t &handle) {
  // KTU
  // dgemm(KTU, 1.0, K, true, Uhist + l * M * S, false, N, S, M, 0.0, handle);
  // tmp_NS = Vbar / KTU
  nip_div(KTU, Vbar, KTUhist + l * N * S, N * S, stream);
  // TODO: replace dgemv with custom kernel for row sum (remove ones_S)
  // bbar = sum_rows(tmp_NS) => tmp_NS * ones_S
  dgemv(bbar, 1.0, KTU, N, S, false, ones_S, 0.0, handle);
}

void update_Ubar_l(double *Ubar, double *Vbar, double *bbar, double *Vhist,
                   double *KTUhist, int l, double *w, double *K, double *KTU,
                   int M, int N, int S, cudaStream_t &stream,
                   cublasHandle_t &handle) {
  // KTU already computed in `update_bbar_l`
  // tmp_NS = Vbar / KTU
  nip_div(KTU, Vbar, KTUhist + l * N * S, N * S, stream);
  // tmp_NS = tmp_NS - bbar * w^T
  dger(KTU, N, S, -1.0, bbar, w, handle);
  // tmp_NS = tmp_NS % V_l
  ip_dot(KTU, Vhist + l * N * S, N * S, stream);
  // Ubar = -K * tmp_NS
  dgemm(Ubar, -1.0, K, false, KTU, false, M, S, N, 0.0, handle);
}

void update_Abar(double *Abar, double *Ubar, double *Vhist, double *KVhist,
                 int l, double *K, double *KV, int M, int N, int S,
                 cudaStream_t &stream, cublasHandle_t &handle) {
  // update KV
  // dgemm(KV, 1.0, K, false, Vhist + (l - 1) * N * S, false, M, S, N, 0.0,
  //       handle);
  // print_device_matrix(KV, M, S, "KV", stream);
  ip_accu_abar(Abar, Ubar, KVhist + (l - 1) * M * S, M * S, stream);
}

void update_wbar(double *wbar, double *bbar, double *bhist, double *KTUhist,
                 int l, double *KTU, int N, int S, cudaStream_t &stream,
                 cublasHandle_t &handle) {
  // tmp_NS = log(KTU)
  nip_log(KTU, KTUhist + l * N * S, N * S, stream);
  ip_dot(bbar, bhist + l * N, N, stream);
  // wbar = (logKTU)^T * bbar + wbar
  dgemv(wbar, 1.0, KTU, N, S, true, bbar, 1.0, handle);
}

void update_err(double &err, double *U, double *KV, double *A, int M, int S,
                cudaStream_t &stream, cublasHandle_t &handle) {
  // compute difference
  // ip_dot_minus(tmp_MS, U, KV, A, M * S, stream);
  // compute norm
  // dnrm2(&err, tmp_MS, M * S, handle);

  auto D2H = cudaMemcpyDeviceToHost;
  double *d_norm = nullptr;
  double h_norm = 0.;

  cudaMallocAsync((void **)&d_norm, sizeof(double), stream);
  cudaMemsetAsync(d_norm, 0, sizeof(double), stream);
  ip_dot_minus_sum(d_norm, U, KV, A, M * S, stream);
  cudaMemcpyAsync(&h_norm, d_norm, sizeof(double), D2H, stream);
  // cudaStreamSynchronize(stream);
  cudaFreeAsync(d_norm, stream);

  err = sqrt(h_norm);
}

void update_loss(double *loss, double *b, double *b_ext, int N,
                 cudaStream_t &stream) {
  nip_sumsq(loss, b, b_ext, N, stream);
}

// forward for barycenter
void forward(int &iter, double &err, double *U, double *V, double *b,
             double *Uhist, double *Vhist, double *bhist, double *KVhist,
             double *KTUhist, double *A, double *w, double *K, double *KV,
             double *KTU, int M, int N, int S, const int max_iter,
             const double zero_tol, bool withgrad, cudaStream_t &stream,
             cublasHandle_t &handle) {
  // forward pass
  auto D2D = cudaMemcpyDeviceToDevice;

  // pre-compute KV
  update_KV(KV, K, V, M, N, S, handle);
  if (withgrad) {
    cudaMemcpyAsync(KVhist, KV, sizeof(double) * M * S, D2D, stream);
  }

  while ((iter < max_iter) && (err > zero_tol)) {
    update_U(U, A, KV, M, S, stream);
    if (withgrad) {
      cudaMemcpyAsync(Uhist + (iter + 1) * M * S, U, sizeof(double) * M * S,
                      D2D, stream);
    }

    update_KTU(KTU, K, U, N, S, M, handle);
    if (withgrad) {
      cudaMemcpyAsync(KTUhist + (iter + 1) * N * S, KTU, sizeof(double) * N * S,
                      D2D, stream);
    }
    update_b(b, U, K, w, KTU, M, N, S, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(bhist + (iter + 1) * N, b, sizeof(double) * N, D2D,
                      stream);
    }

    update_V(V, b, KTUhist, iter, N, S, stream);
    if (withgrad) {
      cudaMemcpyAsync(Vhist + (iter + 1) * N * S, V, sizeof(double) * N * S,
                      D2D, stream);
    }

    update_KV(KV, K, V, M, N, S, handle);
    if (withgrad) {
      cudaMemcpyAsync(KVhist + (iter + 1) * M * S, KV, sizeof(double) * M * S,
                      D2D, stream);
    }
    update_err(err, U, KV, A, M, S, stream, handle);
    iter++;
  }
  // TODO: rescale b sum to 1
}

void backward(int &iter, double *Ubar, double *Vbar, double *bbar, double *Abar,
              double *wbar, double *Uhist, double *Vhist, double *bhist,
              double *KVhist, double *KTUhist, double *w, double *b_ext,
              double *K, double *KV, double *KTU, double *ones_S, int M, int N,
              int S, cudaStream_t &stream, cublasHandle_t &handle) {
  // backward pass
  for (int l = iter; l > 0; --l) {
    if (l == iter) {
      update_bbar_L(bbar, bhist, l, b_ext, N, stream);
      update_Ubar_L(Ubar, bbar, Vhist, l, w, K, KTU, M, N, S, stream, handle);
      // compute KTU for `update_wbar`
      // dgemm(KTU, 1.0, K, true, Uhist + l * M * S, false, N, S, M, 0.0,
      // handle);
    } else {
      update_Vbar_l(Vbar, Ubar, Uhist, Vhist, KVhist, l, K, KV, M, N, S, stream,
                    handle);
      update_bbar_l(bbar, Vbar, Uhist, KTUhist, l, K, KTU, ones_S, M, N, S,
                    stream, handle);
      update_Ubar_l(Ubar, Vbar, bbar, Vhist, KTUhist, l, w, K, KTU, M, N, S,
                    stream, handle);
    }

    update_Abar(Abar, Ubar, Vhist, KVhist, l, K, KV, M, N, S, stream, handle);
    update_wbar(wbar, bbar, bhist, KTUhist, l, KTU, N, S, stream, handle);
  }
}

// output for barycenter: U, V, b, grad_A, grad_w
// output for WDL: b, grad_A, grad_w
// NOTE: maybe two different functions?

void impl_barycenter(int &iter, double &err, double *U, double *V, double *b,
                     double *Ubar, double *Vbar, double *bbar, double *Abar,
                     double *wbar, double *Uhist, double *Vhist, double *bhist,
                     double *KVhist, double *KTUhist, double *A, double *w,
                     double *b_ext, double *K, double *KV, double *KTU,
                     double *ones_S, int M, int N, int S, const int max_iter,
                     const double zero_tol, bool withgrad, cudaStream_t &stream,
                     cublasHandle_t &handle) {
  forward(iter, err, U, V, b, Uhist, Vhist, bhist, KVhist, KTUhist, A, w, K, KV,
          KTU, M, N, S, max_iter, zero_tol, withgrad, stream, handle);
  if (withgrad) {
    backward(iter, Ubar, Vbar, bbar, Abar, wbar, Uhist, Vhist, bhist, KVhist,
             KTUhist, w, b_ext, K, KV, KTU, ones_S, M, N, S, stream, handle);
  }
  // rescale at the end
  normalize(b, N, handle);
}

void cuda_barycenter_parallel(double *U, double *V, double *b, double *grad_A,
                              double *grad_w, double *loss, int *iter_out,
                              double *err_out, const double *A, const double *w,
                              const double *C, const double *b_ext, const int M,
                              const int N, const int S, const double reg,
                              const bool withgrad, const int max_iter,
                              const double zero_tol) {
  // takes device pointers as input, to be exported
  // output: U, V, b, iter, err, loss, grad_A, grad_w
  // input: A, w, K
  // params: M, N, S, max_iter, zero_tol
  // flags: withgrad

  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;

  int iter = 0;
  double err = 1000.;

  /* step 1: create the pointers */
  double *d_A = nullptr, *d_C = nullptr;     // data
  double *d_w = nullptr, *d_b_ext = nullptr; // data
  double *d_K = nullptr;                     // kernel matrix
  double *d_U = nullptr, *d_V = nullptr;     // scaling vectors
  double *d_b = nullptr;                     // barycenter
  double *d_KV = nullptr, *d_KTU = nullptr;  // temp matrices
  double *d_ones_S = nullptr;                // ones_S
  double *d_loss = nullptr;                  // loss

  // history of U, V, b
  double *d_U_hist = nullptr, *d_V_hist = nullptr, *d_b_hist = nullptr;
  // keep track of the KV and KTU too
  double *d_KV_hist = nullptr, *d_KTU_hist = nullptr;
  // adjoints
  double *d_Ubar = nullptr, *d_Vbar = nullptr, *d_bbar = nullptr;
  double *d_Abar = nullptr, *d_wbar = nullptr;

  /* step 2: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 3: allocate memory for the variables */
  // Create a memory pool (once at initialization)
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0); // TODO: check device id?
  cudaMallocAsync((void **)&d_A, sizeof(double) * M * S, stream);
  cudaMallocAsync((void **)&d_C, sizeof(double) * M * N, stream);
  cudaMallocAsync((void **)&d_w, sizeof(double) * S, stream);
  cudaMallocAsync((void **)&d_b_ext, sizeof(double) * N, stream);
  cudaMallocAsync((void **)&d_K, sizeof(double) * M * N, stream);
  cudaMallocAsync((void **)&d_U, sizeof(double) * M * S, stream);
  cudaMallocAsync((void **)&d_V, sizeof(double) * N * S, stream);
  cudaMallocAsync((void **)&d_b, sizeof(double) * N, stream);
  cudaMallocAsync((void **)&d_KV, sizeof(double) * M * S, stream);
  cudaMallocAsync((void **)&d_KTU, sizeof(double) * N * S, stream);
  cudaMallocAsync((void **)&d_ones_S, sizeof(double) * S, stream);
  cudaMallocAsync((void **)&d_loss, sizeof(double), stream);
  cudaMemsetAsync(d_loss, 0, sizeof(double), stream);

  if (withgrad) {
    cudaMallocAsync((void **)&d_U_hist, sizeof(double) * (max_iter + 1) * M * S,
                    stream);
    cudaMallocAsync((void **)&d_V_hist, sizeof(double) * (max_iter + 1) * N * S,
                    stream);
    cudaMallocAsync((void **)&d_b_hist, sizeof(double) * (max_iter + 1) * N,
                    stream);
    cudaMallocAsync((void **)&d_KV_hist,
                    sizeof(double) * (max_iter + 1) * M * S, stream);
    cudaMallocAsync((void **)&d_KTU_hist,
                    sizeof(double) * (max_iter + 1) * N * S, stream);
    cudaMallocAsync((void **)&d_Ubar, sizeof(double) * M * S, stream);
    cudaMallocAsync((void **)&d_Vbar, sizeof(double) * N * S, stream);
    cudaMallocAsync((void **)&d_bbar, sizeof(double) * N, stream);
    cudaMallocAsync((void **)&d_Abar, sizeof(double) * M * S, stream);
    cudaMallocAsync((void **)&d_wbar, sizeof(double) * S, stream);

    cudaMemsetAsync(d_Ubar, 0, sizeof(double) * M * S, stream);
    cudaMemsetAsync(d_Vbar, 0, sizeof(double) * N * S, stream);
    cudaMemsetAsync(d_bbar, 0, sizeof(double) * N, stream);
    cudaMemsetAsync(d_Abar, 0, sizeof(double) * M * S, stream);
    cudaMemsetAsync(d_wbar, 0, sizeof(double) * S, stream);
  }

  /* step 4: copy data to device */
  cudaMemcpyAsync(d_A, A, sizeof(double) * M * S, H2D, stream);
  cudaMemcpyAsync(d_C, C, sizeof(double) * M * N, H2D, stream);
  cudaMemcpyAsync(d_w, w, sizeof(double) * S, H2D, stream);
  cudaMemcpyAsync(d_b_ext, b_ext, sizeof(double) * N, H2D, stream);

  // compute the Gibbs kernel K
  cudaMemcpyAsync(d_K, C, sizeof(double) * M * N, H2D, stream);
  update_K(d_K, M, N, reg, stream);
  // print_device_matrix(d_K, M, N, "K");

  // init U, V to ones
  init_ones(d_U, M * S, stream);
  init_ones(d_V, N * S, stream);
  init_ones(d_ones_S, S, stream);
  if (withgrad) {
    cudaMemcpyAsync(d_U_hist, d_U, sizeof(double) * M * S, D2D, stream);
    cudaMemcpyAsync(d_V_hist, d_V, sizeof(double) * N * S, D2D, stream);
  }

  /* step 5: computation*/
  impl_barycenter(iter, err, d_U, d_V, d_b, d_Ubar, d_Vbar, d_bbar, d_Abar,
                  d_wbar, d_U_hist, d_V_hist, d_b_hist, d_KV_hist, d_KTU_hist,
                  d_A, d_w, d_b_ext, d_K, d_KV, d_KTU, d_ones_S, M, N, S,
                  max_iter, zero_tol, withgrad, stream, handle);
  update_loss(d_loss, d_b, d_b_ext, N, stream);

  /* step 6: copy data back to host */
  cudaMemcpyAsync(U, d_U, sizeof(double) * M * S, D2H, stream);
  cudaMemcpyAsync(V, d_V, sizeof(double) * N * S, D2H, stream);
  cudaMemcpyAsync(b, d_b, sizeof(double) * N, D2H, stream);
  cudaMemcpyAsync(loss, d_loss, sizeof(double), D2H, stream);
  if (withgrad) {
    cudaMemcpyAsync(grad_A, d_Abar, sizeof(double) * M * S, D2H, stream);
    cudaMemcpyAsync(grad_w, d_wbar, sizeof(double) * S, D2H, stream);
  }
  // wait for all to finish
  cudaStreamSynchronize(stream);
  // other info
  *iter_out = iter;
  *err_out = err;

  /* step 7: free resources */
  cudaFreeAsync(d_A, stream);
  cudaFreeAsync(d_C, stream);
  cudaFreeAsync(d_w, stream);
  cudaFreeAsync(d_b_ext, stream);
  cudaFreeAsync(d_K, stream);
  cudaFreeAsync(d_U, stream);
  cudaFreeAsync(d_V, stream);
  cudaFreeAsync(d_b, stream);
  cudaFreeAsync(d_KV, stream);
  cudaFreeAsync(d_KTU, stream);
  cudaFreeAsync(d_ones_S, stream);
  if (withgrad) {
    cudaFreeAsync(d_U_hist, stream);
    cudaFreeAsync(d_V_hist, stream);
    cudaFreeAsync(d_b_hist, stream);
    cudaFreeAsync(d_KV_hist, stream);
    cudaFreeAsync(d_KTU_hist, stream);
    cudaFreeAsync(d_Ubar, stream);
    cudaFreeAsync(d_Vbar, stream);
    cudaFreeAsync(d_bbar, stream);
    cudaFreeAsync(d_Abar, stream);
    cudaFreeAsync(d_wbar, stream);
  }
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

#endif
#endif
