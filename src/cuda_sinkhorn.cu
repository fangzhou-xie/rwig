// implementation of the CUDA Sinkhorn interface
// `cuda_sinkhorn_vanilla`

// #include "check_cuda.hpp" // for checking cuda availability

#ifdef HAVE_CUBLAS
#ifdef HAVE_CUDA_RUNTIME

#include "cuda_kernels.cuh"

#include "cuda_interface.cuh"

/*
  helper functions: now using the wrappers in cuda_kernels.cuh
*/

void update_Kv(double *Kv, double *K, double *v, int M, int N,
               cublasHandle_t &handle) {
  dgemv(Kv, 1.0, K, M, N, false, v, 0.0, handle);
}

void update_u(double *u, double *a, double *Kv, int M, cudaStream_t &stream) {
  nip_div(u, a, Kv, M, stream);
}

void update_KTu_v(double *v, double *u, double *b, double *K, double *KTu,
                  int M, int N, cudaStream_t &stream, cublasHandle_t &handle) {
  dgemv(KTu, 1.0, K, M, N, true, u, 0.0, handle);
  nip_div(v, b, KTu, N, stream);
}

void update_err(double &err, double *u, double *v, double *Kv, double *KTu,
                double *a, double *b, double *tmp_M, double *tmp_N, int M,
                int N, cudaStream_t &stream, cublasHandle_t &handle) {
  auto D2H = cudaMemcpyDeviceToHost;
  double *d_norm_M = nullptr, *d_norm_N = nullptr;
  double norm_M, norm_N;

  cudaMallocAsync((void **)&d_norm_M, sizeof(double), stream);
  cudaMemsetAsync(d_norm_M, 0, sizeof(double), stream);

  cudaMallocAsync((void **)&d_norm_N, sizeof(double), stream);
  cudaMemsetAsync(d_norm_N, 0, sizeof(double), stream);

  ip_dot_minus_sum(d_norm_M, u, Kv, a, M, stream);
  ip_dot_minus_sum(d_norm_N, v, KTu, b, N, stream);

  cudaMemcpyAsync(&norm_M, d_norm_M, sizeof(double), D2H, stream);
  cudaMemcpyAsync(&norm_N, d_norm_N, sizeof(double), D2H, stream);
  cudaStreamSynchronize(stream);

  cudaFreeAsync(d_norm_M, stream);
  cudaFreeAsync(d_norm_N, stream);

  // compute the differences
  // ip_dot_minus(tmp_M, u, Kv, a, M, stream);
  // ip_dot_minus(tmp_N, v, KTu, b, N, stream);
  // compute norm
  // dnrm2(&norm_M, tmp_M, M, handle);
  // dnrm2(&norm_N, tmp_N, N, handle);

  // sum the two norms
  err = sqrt(norm_M) + sqrt(norm_N);
}

void update_P(double *P, double *u, double *K, double *v, int M, int N,
              cudaStream_t &stream) {
  nip_diag_scale(P, u, K, v, M, N, stream);
}

void update_loss(double *loss, double *C, double *P, int M, int N, double reg,
                 cudaStream_t &stream) {
  ip_sinkloss(loss, C, P, M, N, reg, stream);
}

void update_PbarK(double *PbarK, double *C, double *P, double *K, int M, int N,
                  double reg, cudaStream_t &stream) {
  ip_PbarK(PbarK, C, P, K, M, N, reg, stream);
}

void update_vbar_L(double *vbar, double *PbarK, double *u, int M, int N,
                   cublasHandle_t &handle) {
  dgemv(vbar, 1.0, PbarK, M, N, true, u, 0.0, handle);
}

void update_ubar_L(double *ubar, double *vbar, double *u, double *v, double *K,
                   double *PbarK, double *KTu, int M, int N,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  // KTu = K^T * u_L
  dgemv(KTu, 1.0, K, M, N, true, u, 0.0, handle);
  // tmp_N = (vbar % v_L) / KTu
  ip_dot_div(KTu, vbar, v, N, stream);
  // ubar_L = PbarK * vbar_L
  dgemv(ubar, 1.0, PbarK, M, N, false, v, 0.0, handle);
  // ubar_L = - K * KTu + ubar_L
  dgemv(ubar, -1.0, K, M, N, false, KTu, 1.0, handle);
}

void update_vbar_l(double *vbar, double *ubar, double *u_hist, double *v_hist,
                   int l, double *K, double *Kv, int M, int N,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  ip_dot_div(Kv, ubar, u_hist + (l + 1) * M, M, stream);
  dgemv(vbar, -1.0, K, M, N, true, Kv, 0.0, handle);
}

void update_ubar_l(double *ubar, double *vbar, double *u_hist, double *v_hist,
                   int l, double *K, double *KTu, int M, int N,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  // update KTu
  dgemv(KTu, 1.0, K, M, N, true, u_hist + l * M, 0.0, handle);
  ip_dot_div(KTu, vbar, v_hist + l * N, N, stream);
  // ubar = - K * KTu
  dgemv(ubar, -1.0, K, M, N, false, KTu, 0.0, handle);
}

void update_abar(double *abar, double *ubar, double *v_hist, int l, double *K,
                 double *Kv, int M, int N, cudaStream_t &stream,
                 cublasHandle_t &handle) {
  dgemv(Kv, 1.0, K, M, N, false, v_hist + (l - 1) * N, 0.0, handle);
  ip_accu_abar(abar, ubar, Kv, M, stream);
}

void forward(int &iter, double &err, double *u, double *v, double *u_hist,
             double *v_hist, double *a, double *b, double *K, int M, int N,
             double *Kv, double *KTu, double *tmp_M, double *tmp_N,
             int max_iter, double zero_tol, bool withgrad, cudaStream_t &stream,
             cublasHandle_t &handle) {
  auto D2D = cudaMemcpyDeviceToDevice;

  // pre-copmpute Kv
  update_Kv(Kv, K, v, M, N, handle);

  while ((iter < max_iter) && (err > zero_tol)) {
    // update u
    update_u(u, a, Kv, M, stream);
    if (withgrad) {
      cudaMemcpyAsync(u_hist + (iter + 1) * M, u, sizeof(double) * M, D2D,
                      stream);
    }

    // update KTu and v
    update_KTu_v(v, u, b, K, KTu, M, N, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(v_hist + (iter + 1) * N, v, sizeof(double) * N, D2D,
                      stream);
    }

    // update Kv
    update_Kv(Kv, K, v, M, N, handle);
    // update error
    update_err(err, u, v, Kv, KTu, a, b, tmp_M, tmp_N, M, N, stream, handle);
    iter++;
  }
}

void backward(double *ubar, double *vbar, double *abar, int &iter, double *u,
              double *v, double *u_hist, double *v_hist, double *PbarK,
              double *C, double *P, double *K, double *Kv, double *KTu, int M,
              int N, double reg, cudaStream_t &stream, cublasHandle_t &handle) {
  // backward pass
  update_PbarK(PbarK, C, P, K, M, N, reg, stream);

  for (int l = iter; l > 0; --l) {
    if (l == iter) {
      // update vbar_L
      update_vbar_L(vbar, PbarK, u, M, N, handle);
      // update ubar_L
      update_ubar_L(ubar, vbar, u, v, K, PbarK, KTu, M, N, stream, handle);
    } else {
      // update vbar_l
      update_vbar_l(vbar, ubar, u_hist, v_hist, l, K, Kv, M, N, stream, handle);
      // update ubar_l
      update_ubar_l(ubar, vbar, u_hist, v_hist, l, K, KTu, M, N, stream,
                    handle);
    }
    // accumulate abar
    update_abar(abar, ubar, v_hist, l, K, Kv, M, N, stream, handle);
  }
}

void cuda_sinkhorn_vanilla(double *P, double *grad_a, double *u, double *v,
                           double *loss, int *iter_out, double *err_out,
                           const double *a, const double *b, const double *C,
                           const int M, const int N, const double reg,
                           const bool withgrad, const int max_iter,
                           const double zero_tol) {
  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;

  int iter = 0;
  double err = 1000.;

  // data: a, b
  double *d_a = nullptr, *d_b = nullptr;
  double *d_K = nullptr, *d_C = nullptr;
  // scaling vectors
  double *d_u = nullptr, *d_v = nullptr;
  // output P
  double *d_P = nullptr;
  // temp vectors
  double *d_Kv = nullptr, *d_KTu = nullptr;
  // loss
  double *d_loss = nullptr;

  // history of u and v
  double *d_u_hist = nullptr, *d_v_hist = nullptr;
  // adjoints
  double *d_ubar = nullptr, *d_vbar = nullptr, *d_abar = nullptr;
  double *d_PbarK = nullptr;
  // tmp vecs
  double *d_tmp_M = nullptr, *d_tmp_N = nullptr;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: allocate memory for the variables */
  // Create a memory pool (once at initialization)
  cudaMemPool_t pool;
  cudaDeviceGetDefaultMemPool(&pool, 0); // TODO: check device id?

  cudaMallocAsync((void **)&d_a, sizeof(double) * M, stream);
  cudaMallocAsync((void **)&d_b, sizeof(double) * N, stream);
  cudaMallocAsync((void **)&d_K, sizeof(double) * M * N, stream);
  cudaMallocAsync((void **)&d_C, sizeof(double) * M * N, stream);
  cudaMallocAsync((void **)&d_u, sizeof(double) * M, stream);
  cudaMallocAsync((void **)&d_v, sizeof(double) * N, stream);
  cudaMallocAsync((void **)&d_P, sizeof(double) * M * N, stream);
  cudaMallocAsync((void **)&d_Kv, sizeof(double) * M, stream);
  cudaMallocAsync((void **)&d_KTu, sizeof(double) * N, stream);
  cudaMallocAsync((void **)&d_loss, sizeof(double), stream);
  cudaMemsetAsync(d_loss, 0, sizeof(double), stream);

  cudaMallocAsync((void **)&d_tmp_M, sizeof(double) * M, stream);
  cudaMallocAsync((void **)&d_tmp_N, sizeof(double) * N, stream);

  if (withgrad) {
    cudaMallocAsync((void **)&d_u_hist, sizeof(double) * M * (max_iter + 1),
                    stream);
    cudaMallocAsync((void **)&d_v_hist, sizeof(double) * N * (max_iter + 1),
                    stream);
    cudaMallocAsync((void **)&d_PbarK, sizeof(double) * M * N, stream);
    cudaMallocAsync((void **)&d_ubar, sizeof(double) * M, stream);
    cudaMallocAsync((void **)&d_vbar, sizeof(double) * N, stream);
    cudaMallocAsync((void **)&d_abar, sizeof(double) * M, stream);

    cudaMemsetAsync(d_ubar, 0, sizeof(double) * M, stream);
    cudaMemsetAsync(d_vbar, 0, sizeof(double) * N, stream);
    cudaMemsetAsync(d_abar, 0, sizeof(double) * M, stream);
  }

  /* step 3: copy data to device */
  cudaMemcpyAsync(d_a, a, sizeof(double) * M, H2D, stream);
  cudaMemcpyAsync(d_b, b, sizeof(double) * N, H2D, stream);
  cudaMemcpyAsync(d_C, C, sizeof(double) * M * N, H2D, stream);

  // compute the Gibbs kernel K
  cudaMemcpyAsync(d_K, C, sizeof(double) * M * N, H2D, stream);
  update_K(d_K, M, N, reg, stream);

  // init u, v to ones
  init_ones(d_u, M, stream);
  init_ones(d_v, N, stream);
  if (withgrad) {
    cudaMemcpyAsync(d_u_hist + iter * M, d_u, sizeof(double) * M, D2D, stream);
    cudaMemcpyAsync(d_v_hist + iter * N, d_v, sizeof(double) * N, D2D, stream);
  }

  cudaStreamSynchronize(stream);

  /* step 4: forward pass */
  forward(iter, err, d_u, d_v, d_u_hist, d_v_hist, d_a, d_b, d_K, M, N, d_Kv,
          d_KTu, d_tmp_M, d_tmp_N, max_iter, zero_tol, withgrad, stream,
          handle);

  // compute optimal coupling P
  update_P(d_P, d_u, d_K, d_v, M, N, stream);

  /* step 5: backward pass */
  if (withgrad) {
    backward(d_ubar, d_vbar, d_abar, iter, d_u, d_v, d_u_hist, d_v_hist,
             d_PbarK, d_C, d_P, d_K, d_Kv, d_KTu, M, N, reg, stream, handle);
  }

  update_loss(d_loss, d_C, d_P, M, N, reg, stream);

  /* step 6: copy data back to host */
  cudaMemcpyAsync(u, d_u, sizeof(double) * M, D2H, stream);
  cudaMemcpyAsync(v, d_v, sizeof(double) * N, D2H, stream);
  cudaMemcpyAsync(P, d_P, sizeof(double) * M * N, D2H, stream);
  cudaMemcpyAsync(loss, d_loss, sizeof(double), D2H, stream);
  if (withgrad) {
    // copy d_abar to grad_a
    cudaMemcpyAsync(grad_a, d_abar, sizeof(double) * M, D2H, stream);
  }
  cudaStreamSynchronize(stream);
  // other info
  *iter_out = iter;
  *err_out = err;

  /* step 7: free resources */
  cudaFreeAsync(d_a, stream);
  cudaFreeAsync(d_b, stream);
  cudaFreeAsync(d_K, stream);
  cudaFreeAsync(d_C, stream);
  cudaFreeAsync(d_u, stream);
  cudaFreeAsync(d_v, stream);
  cudaFreeAsync(d_P, stream);
  cudaFreeAsync(d_Kv, stream);
  cudaFreeAsync(d_KTu, stream);
  cudaFreeAsync(d_loss, stream);
  cudaFreeAsync(d_tmp_M, stream);
  cudaFreeAsync(d_tmp_N, stream);
  if (withgrad) {
    cudaFreeAsync(d_u_hist, stream);
    cudaFreeAsync(d_v_hist, stream);
    cudaFreeAsync(d_PbarK, stream);
    cudaFreeAsync(d_ubar, stream);
    cudaFreeAsync(d_vbar, stream);
    cudaFreeAsync(d_abar, stream);
  }
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

#endif
#endif
