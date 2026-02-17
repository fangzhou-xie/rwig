/*
  Element-wise operation wrappers
*/

template <void (*KernelFunc)(int, double *, double *, double *)>
void element_op(double *z, int n, double *x, double *y) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  double *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, n * sizeof(double));
  cudaMalloc(&d_y, n * sizeof(double));
  cudaMalloc(&d_z, n * sizeof(double));

  cudaMemcpyAsync(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice, stream);

  KernelFunc<<<numBlocks, blockSize, 0, stream>>>(n, d_z, d_x, d_y);

  cudaMemcpyAsync(z, d_z, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaStreamDestroy(stream);
}

// Convenience aliases
void element_add(double *z, int n, double *x, double *y) {
  element_op<nip_add>(z, n, x, y);
}

void element_minus(double *z, int n, double *x, double *y) {
  element_op<nip_minus>(z, n, x, y);
}

void element_dot(double *z, int n, double *x, double *y) {
  element_op<nip_dot>(z, n, x, y);
}

void element_div(double *z, int n, double *x, double *y) {
  element_op<nip_div>(z, n, x, y);
}

void element_exp(double *y, int n) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  double *d_y;
  cudaMalloc(&d_y, n * sizeof(double));

  cudaMemcpyAsync(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice, stream);

  inplace_exp<<<numBlocks, blockSize, 0, stream>>>(n, d_y);

  cudaMemcpyAsync(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_y);
  cudaStreamDestroy(stream);
}

double element_norm(double *x, int n) {
  cublasHandle_t handle;
  cudaStream_t stream;
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  double *d_x;
  cudaMalloc(&d_x, n * sizeof(double));
  double result;

  cudaMemcpyAsync(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice, stream);

  cublasDnrm2(handle, n, d_x, 1, &result);
  cudaMemcpyAsync(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_x);
  cudaStreamDestroy(stream);

  return result;
}

void cuda_diag_scale(int m, int n, double *out, const double *u,
                     const double *K, const double *v) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m * n + blockSize - 1) / blockSize;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  double *d_u, *d_K, *d_v, *d_out;
  cudaMalloc(&d_u, m * sizeof(double));
  cudaMalloc(&d_K, m * n * sizeof(double));
  cudaMalloc(&d_v, n * sizeof(double));
  cudaMalloc(&d_out, m * n * sizeof(double));

  cudaMemcpyAsync(d_u, u, m * sizeof(double), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_K, K, m * n * sizeof(double), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_v, v, n * sizeof(double), cudaMemcpyHostToDevice, stream);

  noninplace_diag_scale<<<numBlocks, blockSize, 0, stream>>>(m, n, d_out, d_u,
                                                             d_K, d_v);

  cudaMemcpyAsync(out, d_out, m * n * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  cudaFree(d_u);
  cudaFree(d_K);
  cudaFree(d_v);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
}

void cuda_dgemv(double *y, const double *A, const double *x, const int M,
                const int N, const bool trans, const double alpha,
                const double beta) {
  cublasHandle_t handle;
  cudaStream_t stream;
  cublasOperation_t transa = trans ? CUBLAS_OP_T : CUBLAS_OP_N;

  const int lda = M;
  const int incx = 1;
  const int incy = 1;

  // if transA is true, A is N x M, y is N x 1, x is M x 1
  // else A is M x N, y is M x 1, x is N x 1
  const int x_size = trans ? M : N;
  const int y_size = trans ? N : M;

  double *d_A = nullptr;
  double *d_x = nullptr;
  double *d_y = nullptr;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_x), sizeof(double) * x_size);
  cudaMalloc(reinterpret_cast<void **>(&d_y), sizeof(double) * y_size);

  cudaMemcpyAsync(d_A, A, sizeof(double) * M * N, cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_x, x, sizeof(double) * x_size, cudaMemcpyHostToDevice,
                  stream);

  /* step 3: compute */
  cublasDgemv(handle, transa, M, N, &alpha, d_A, lda, d_x, incx, &beta, d_y,
              incy);

  /* step 4: copy data to host */
  cudaMemcpyAsync(y, d_y, sizeof(double) * y_size, cudaMemcpyDeviceToHost,
                  stream);

  cudaStreamSynchronize(stream);

  /* step 5: cleanup */
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

void cuda_dgemm(double *C, double alpha, double *A, bool transA, double *B,
                bool transB, int M, int N, int K, double beta) {

  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  // auto D2D = cudaMemcpyDeviceToDevice;

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);
  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * M * K);
  cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * N * K);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * M * N);
  cudaMemcpyAsync(d_A, A, sizeof(double) * M * K, H2D, stream);
  cudaMemcpyAsync(d_B, B, sizeof(double) * N * K, H2D, stream);
  // cudaMemcpyAsync(d_C, C, sizeof(double) * M * N, H2D, stream);
  dgemm(d_C, alpha, d_A, transA, d_B, transB, M, N, K, beta, handle);
  cudaMemcpyAsync(C, d_C, sizeof(double) * M * N, D2H, stream);
  cudaStreamSynchronize(stream);

  /* step 3: free device memory and destroy handle and stream */
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);
  cudaStreamDestroy(stream);
}

void cuda_a_div_Kv(double *u, const int m, const int n, const double *a,
                   const double *K, const double *v) {
  cublasHandle_t handle;
  cudaStream_t stream;

  const int incx = 1;
  const int incy = 1;

  double *d_K = nullptr;
  double *d_a = nullptr;
  double *d_v = nullptr;
  double *d_u = nullptr;
  const double alpha = 1.0;
  const double beta = 0.0;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * m * n);
  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&d_u), sizeof(double) * m);

  cudaMemcpyAsync(d_K, K, sizeof(double) * m * n, cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_a, a, sizeof(double) * m, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_v, v, sizeof(double) * n, cudaMemcpyHostToDevice, stream);

  /* step 3: compute */
  // First compute u = K * v
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_K, m, d_v, incx, &beta, d_u,
              incy);

  // Then compute u = a ./ u
  int blockSize = BLOCK_SIZE;
  int numBlocks = (m + blockSize - 1) / blockSize;
  inplace_div<<<numBlocks, blockSize, 0, stream>>>(m, d_a, d_u);

  /* step 4: copy data to host */
  cudaMemcpyAsync(u, d_u, sizeof(double) * m, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  /* step 5: cleanup */
  cudaFree(d_K);
  cudaFree(d_a);
  cudaFree(d_v);
  cudaFree(d_u);

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

void cuda_b_div_Ktu(double *v, const int m, const int n, const double *b,
                    const double *K, const double *u) {
  cublasHandle_t handle;
  cudaStream_t stream;

  const int incx = 1;
  const int incy = 1;

  double *d_K = nullptr;
  double *d_b = nullptr;
  double *d_u = nullptr;
  double *d_v = nullptr;
  const double alpha = 1.0;
  const double beta = 0.0;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * m * n);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&d_u), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(double) * n);

  cudaMemcpyAsync(d_K, K, sizeof(double) * m * n, cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_b, b, sizeof(double) * n, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_u, u, sizeof(double) * m, cudaMemcpyHostToDevice, stream);

  /* step 3: compute */
  // First compute v = K^T * u
  cublasDgemv(handle, CUBLAS_OP_T, m, n, &alpha, d_K, m, d_u, incx, &beta, d_v,
              incy);

  // Then compute v = b ./ v
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;
  inplace_div<<<numBlocks, blockSize, 0, stream>>>(n, d_b, d_v);

  /* step 4: copy data to host */
  cudaMemcpyAsync(v, d_v, sizeof(double) * n, cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  /* step 5: cleanup */
  cudaFree(d_K);
  cudaFree(d_b);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

//
void cuda_sinkhorn_vanilla(double *P, double *grad_a, double *u, double *v,
                           double *loss, int *iter_out, double *err_out,
                           const double *a, const double *b, const double *C,
                           const int m, const int n, const double reg,
                           const bool withgrad, const int max_iter,
                           const double zero_tol) {

  // CUDA version ignores the verbose
  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;
  auto opN = CUBLAS_OP_N;
  auto opT = CUBLAS_OP_T;

  int blockSize = BLOCK_SIZE;
  int numBlocks_m = (m + blockSize - 1) / blockSize;
  int numBlocks_n = (n + blockSize - 1) / blockSize;
  int numBlocks_mn = (m * n + blockSize - 1) / blockSize;

  const int incx = 1;
  const int incy = 1;
  const double alpha = 1.0;
  const double alpha_minus = -1.0;
  const double beta = 0.0;

  // data: a, b
  double *d_a = nullptr, *d_b = nullptr;
  double *d_K = nullptr, *d_C = nullptr;
  // scaling vectors
  double *d_u = nullptr, *d_v = nullptr;
  // output P
  double *d_P = nullptr;

  // temp vectors
  double *d_Kv = nullptr, *d_KTu = nullptr;
  double *tmp_vec_m = nullptr, *tmp_vec_n = nullptr;
  // history of u and v
  double *d_u_hist = nullptr, *d_v_hist = nullptr;
  // adjoints
  double *d_ubar = nullptr, *d_vbar = nullptr, *d_abar = nullptr;
  double *d_PbarK = nullptr;
  double *d_loss = nullptr;
  // double *d_grad_a = nullptr;

  int iter = 0;
  double err = 1000.;
  double err_m, err_n;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(double) * m * n);

  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * m * n);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * m * n);
  cudaMalloc(reinterpret_cast<void **>(&d_u), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&d_Kv), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&d_KTu), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&tmp_vec_m), sizeof(double) * m);
  cudaMalloc(reinterpret_cast<void **>(&tmp_vec_n), sizeof(double) * n);
  cudaMalloc(reinterpret_cast<void **>(&d_loss), sizeof(double));
  cudaMemset(d_loss, 0, sizeof(double));

  if (withgrad) {
    cudaMalloc(reinterpret_cast<void **>(&d_u_hist),
               sizeof(double) * m * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_v_hist),
               sizeof(double) * n * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_PbarK), sizeof(double) * m * n);
    cudaMalloc(reinterpret_cast<void **>(&d_ubar), sizeof(double) * m);
    cudaMalloc(reinterpret_cast<void **>(&d_vbar), sizeof(double) * n);
    cudaMalloc(reinterpret_cast<void **>(&d_abar), sizeof(double) * m);

    cudaMemset(d_ubar, 0, sizeof(double) * m);
    cudaMemset(d_vbar, 0, sizeof(double) * n);
    cudaMemset(d_abar, 0, sizeof(double) * m);
  }

  cudaMemcpyAsync(d_C, C, sizeof(double) * m * n, H2D, stream);
  cudaMemcpyAsync(d_K, C, sizeof(double) * m * n, H2D, stream);

  cudaMemcpyAsync(d_a, a, sizeof(double) * m, H2D, stream);
  cudaMemcpyAsync(d_b, b, sizeof(double) * n, H2D, stream);

  // compute K = exp(-C / reg)
  inplace_gibbs<<<numBlocks_mn, blockSize, 0, stream>>>(m * n, d_K, reg);

  // initialize u and v to ones
  inplace_fill_ones<<<numBlocks_m, blockSize, 0, stream>>>(m, d_u);
  inplace_fill_ones<<<numBlocks_n, blockSize, 0, stream>>>(n, d_v);
  if (withgrad) {
    cudaMemcpyAsync(d_u_hist + iter * m, d_u, sizeof(double) * m, D2D, stream);
    cudaMemcpyAsync(d_v_hist + iter * n, d_v, sizeof(double) * n, D2D, stream);
  }

  // pre-compute Kv: d_Kv = d_K * d_v
  cublasDgemv(handle, opN, m, n, &alpha, d_K, m, d_v, incx, &beta, d_Kv, incy);

  /* step 3: forward pass */
  while ((iter < max_iter) && (err >= zero_tol)) {
    // division of u = a ./ (K * v): d_u = d_a / d_u
    noninplace_div<<<numBlocks_m, blockSize, 0, stream>>>(m, d_u, d_a, d_Kv);
    if (withgrad) {
      cudaMemcpyAsync(d_u_hist + (iter + 1) * m, d_u, sizeof(double) * m, D2D,
                      stream);
    }

    // v = b ./ (K^T * u): save d_KTu for norm
    // d_KTu = d_K^T * d_u
    cublasDgemv(handle, opT, m, n, &alpha, d_K, m, d_u, incx, &beta, d_KTu,
                incy);
    // d_v = d_b ./ d_KTu
    noninplace_div<<<numBlocks_n, blockSize, 0, stream>>>(n, d_v, d_b, d_KTu);
    if (withgrad) {
      cudaMemcpyAsync(d_v_hist + (iter + 1) * n, d_v, sizeof(double) * n, D2D,
                      stream);
    }

    // check convergence
    // d_Kv = d_K * d_v
    cublasDgemv(handle, opN, m, n, &alpha, d_K, m, d_v, incx, &beta, d_Kv,
                incy);
    inplace_err_vec<<<numBlocks_m, blockSize, 0, stream>>>(m, tmp_vec_m, d_u,
                                                           d_Kv, d_a);
    inplace_err_vec<<<numBlocks_n, blockSize, 0, stream>>>(n, tmp_vec_n, d_v,
                                                           d_KTu, d_b);
    // compute norm
    cublasDnrm2(handle, m, tmp_vec_m, 1, &err_m);
    cublasDnrm2(handle, n, tmp_vec_n, 1, &err_n);

    err = err_m + err_n;
    iter++;
  }

  // P = diag(u) * K * diag(v)
  noninplace_diag_scale<<<numBlocks_mn, blockSize, 0, stream>>>(m, n, d_P, d_u,
                                                                d_K, d_v);

  /* step 4: backward pass*/
  if (withgrad) {
    inplace_PbarK<<<numBlocks_mn, blockSize, 0, stream>>>(m * n, d_PbarK, d_C,
                                                          d_P, d_K, reg);
    for (int l = iter; l > 0; --l) {
      if (l == iter) {
        // update d_vbar: vbar = PbarK^T * u
        cublasDgemv(handle, opT, m, n, &alpha, d_PbarK, m, d_u_hist + l * m,
                    incx, &beta, d_vbar, incy);

        // update d_ubar: update KTu
        cublasDgemv(handle, opT, m, n, &alpha, d_K, m, d_u_hist + l * m, incx,
                    &beta, d_KTu, incy);
        noninplace_vbarv_div_KTu<<<numBlocks_n, blockSize, 0, stream>>>(
            n, tmp_vec_n, d_vbar, d_v, d_KTu);
        // d_ubar = PbarK * v
        cublasDgemv(handle, opN, m, n, &alpha, d_PbarK, m, d_v, incx, &beta,
                    d_ubar, incy);
        // final d_ubar
        cublasDgemv(handle, opN, m, n, &alpha_minus, d_K, m, tmp_vec_n, incx,
                    &alpha, d_ubar, incy);
      } else {
        // update d_vbar: update Kv
        cublasDgemv(handle, opN, m, n, &alpha, d_K, m, d_v_hist + l * n, incx,
                    &beta, d_Kv, incy);
        noninplace_vbarv_div_KTu<<<numBlocks_m, blockSize, 0, stream>>>(
            m, tmp_vec_m, d_ubar, d_u_hist + (l + 1) * m, d_Kv);
        cublasDgemv(handle, opT, m, n, &alpha_minus, d_K, m, tmp_vec_m, incx,
                    &beta, d_vbar, incy);

        // update d_ubar: update KTu
        cublasDgemv(handle, opT, m, n, &alpha, d_K, m, d_u_hist + l * m, incx,
                    &beta, d_KTu, incy);
        noninplace_vbarv_div_KTu<<<numBlocks_n, blockSize, 0, stream>>>(
            n, tmp_vec_n, d_vbar, d_v_hist + l * n, d_KTu);
        cublasDgemv(handle, opN, m, n, &alpha_minus, d_K, m, tmp_vec_n, incx,
                    &beta, d_ubar, incy);
      }

      // accumulate gradient
      // update Kv = K * v
      cublasDgemv(handle, opN, m, n, &alpha, d_K, m, d_v_hist + (l - 1) * n,
                  incx, &beta, d_Kv, incy);
      inplace_abar<<<numBlocks_m, blockSize, 0, stream>>>(m, d_abar, d_ubar,
                                                          d_Kv);
    }
  }

  // compute loss
  inplace_sinkloss<<<numBlocks_mn, blockSize, 0, stream>>>(m * n, d_loss, d_C,
                                                           d_P, reg);
  /* step 5: copy data to host */
  cudaMemcpyAsync(u, d_u, sizeof(double) * m, D2H, stream);
  cudaMemcpyAsync(v, d_v, sizeof(double) * n, D2H, stream);
  cudaMemcpyAsync(P, d_P, sizeof(double) * m * n, D2H, stream);
  cudaMemcpyAsync(loss, d_loss, sizeof(double), D2H, stream);
  if (withgrad) {
    // copy d_abar to grad_a
    cudaMemcpyAsync(grad_a, d_abar, sizeof(double) * m, D2H, stream);
  }
  cudaStreamSynchronize(stream);

  // print u
  // std::cout << "u: " << "\n";
  // for (int i = 0; i < m; ++i) {
  //   std::cout << u[i] << " ";
  // }
  // std::cout << "\n";

  // report other info
  *iter_out = iter;
  *err_out = err;

  /* step 6: cleanup */
  cudaFree(d_K);
  cudaFree(d_C);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_Kv);
  cudaFree(d_KTu);
  cudaFree(tmp_vec_m);
  cudaFree(tmp_vec_n);
  cudaFree(d_P);
  cudaFree(d_loss);
  if (withgrad) {
    cudaFree(d_u_hist);
    cudaFree(d_v_hist);
    cudaFree(d_PbarK);
    cudaFree(d_ubar);
    cudaFree(d_vbar);
    cudaFree(d_abar);
  }

  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

void cuda_barycenter_parallel(double *b, double *grad_A, double *grad_w,
                              double *U, double *V, double *loss, int *iter_out,
                              double *err_out, const double *A, const double *C,
                              const double *w, const double *b_ext, const int M,
                              const int N, const int S, const double reg,
                              const bool withgrad, const int max_iter,
                              const double zero_tol) {
  // CUDA version ignores the verbose
  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;
  auto opN = CUBLAS_OP_N;
  auto opT = CUBLAS_OP_T;

  int blockSize = BLOCK_SIZE;
  int numBlocks_M = (M + blockSize - 1) / blockSize;
  int numBlocks_N = (N + blockSize - 1) / blockSize;
  int numBlocks_MS = (M * S + blockSize - 1) / blockSize;
  int numBlocks_NS = (N * S + blockSize - 1) / blockSize;
  int numBlocks_MN = (M * N + blockSize - 1) / blockSize;

  const int incx = 1;
  const int incy = 1;
  const double alpha = 1.0;
  const double alpha_minus = -1.0;
  const double beta = 0.0;

  // data: A, w, b_ext
  double *d_A = nullptr, *d_w = nullptr;
  double *d_K = nullptr, *d_C = nullptr;
  // scaling matrices
  double *d_U = nullptr, *d_V = nullptr;
  // temp matrices
  double *d_KV = nullptr, *d_KTU = nullptr;
  double *tmp_vec_MS = nullptr, *tmp_vec_NS = nullptr;
  // history of U and V
  double *d_U_hist = nullptr, *d_V_hist = nullptr;
  // adjoints
  double *d_Ubar = nullptr, *d_Vbar = nullptr, *d_bbar = nullptr;
  double *d_loss = nullptr;

  int iter = 0;
  double err = 1000.;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: copy data to device */
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_w), sizeof(double) * S);
  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&d_KV), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_KTU), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&tmp_vec_MS), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&tmp_vec_NS), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&d_loss), sizeof(double));
  cudaMemset(d_loss, 0, sizeof(double));

  if (withgrad) {
    cudaMalloc(reinterpret_cast<void **>(&d_U_hist),
               sizeof(double) * M * S * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_V_hist),
               sizeof(double) * N * S * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_Ubar), sizeof(double) * M * S);
    cudaMalloc(reinterpret_cast<void **>(&d_Vbar), sizeof(double) * N * S);
    cudaMalloc(reinterpret_cast<void **>(&d_bbar), sizeof(double) * M);

    cudaMemset(d_Ubar, 0, sizeof(double) * M * S);
    cudaMemset(d_Vbar, 0, sizeof(double) * N * S);
    cudaMemset(d_bbar, 0, sizeof(double) * M);
  }

  cudaMemcpyAsync(d_C, C, sizeof(double) * M * N, H2D, stream);
  cudaMemcpyAsync(d_K, C, sizeof(double) * M * N, H2D, stream);

  cudaMemcpyAsync(d_A, A, sizeof(double) * M * S, H2D, stream);
  cudaMemcpyAsync(d_w, w, sizeof(double) * S, H2D, stream);

  // compute K = exp(-C / reg)
  inplace_gibbs < <<numBlocks_mn, blockSize, 0, stream>>(M * N, d_K, reg);

  // initialize U and V to ones
  inplace_fill_ones<<<numBlocks_MS, blockSize, 0, stream>>>(M * S, d_U);
  inplace_fill_ones<<<numBlocks_NS, blockSize, 0, stream>>>(N * S, d_V);
  if (withgrad) {
    cudaMemcpyAsync(d_U_hist + iter * M * S, d_U, sizeof(double) * M * S, D2D,
                    stream);
    cudaMemcpyAsync(d_V_hist + iter * N * S, d_V, sizeof(double) * N * S, D2D,
                    stream);
  }

  /* step 3: forward pass */
  // pre-compute KV: d_KV = d_K * d_V
  cublasDgemm(handle, opN, opN, M, S, N, &alpha, d_K, M, d_V, N, &beta, d_KV,
              M);

  while ((iter < max_iter) && (err >= zero_tol)) {
    // division of U = A ./ (K * V): d_U = d_A / d_KV
    noninplace_div<<<numBlocks_MS, blockSize, 0, stream>>>(M * S, d_U, d_A,
                                                           d_KV);
    if (withgrad) {
      cudaMemcpyAsync(d_U_hist + (iter + 1) * M * S, d_U,
                      sizeof(double) * M * S, D2D, stream);
    }
    // TODO: how to compute barycenter

    // V = b_ext ./ (K^T * U): save d_KTU for norm
    // d_KTU = d_K^T * d_U
    cublasDgemm(handle, opT, opN, N, S, M, &alpha, d_K, M, d_U, M, &beta, d_KTU,
                N);
    // d_V = d_b_ext ./ d_KTU
    noninplace_div<<<numBlocks_NS, blockSize, 0, stream>>>(N * S, d_V, b_ext,
                                                           d_KTU);
    if (withgrad) {
      cudaMemcpyAsync(d_V_hist + (iter + 1) * N * S, d_V,
                      sizeof(double) * N * S, D2D, stream);
    }

    // check convergence
    // d_KV = d_K * d_V
    cublasDgemm(handle, opN, opN, M, S, N, &alpha, d_K, M, d_V, N, &beta, d_KV,
                M);
    inplace_err_vec<<<numBlocks_MS, blockSize, 0, stream>>>(M * S, tmp_vec_MS,
                                                            d_U, d_KV, d_A);

    cublasDnrm2(handle, M * S, tmp_vec_MS, 1, &err);
    iter++;
  }

  /* step 4: backward pass*/
  // TODO
}

/*
  Sinkhorn kernels
*/

void init_ones(double *x, int n, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (n + blockSize - 1) / blockSize;

  ip_fill_ones<<<numBlocks, blockSize, 0, stream>>>(n, x);
}

void update_K(double *K, int M, int N, double reg, cudaStream_t &stream) {
  // K = exp(-C / reg)
  int blockSize = BLOCK_SIZE;
  int numBlocks = (M * N + blockSize - 1) / blockSize;

  ip_gibbs<<<numBlocks, blockSize, 0, stream>>>(M * N, K, reg);
}

void update_PbarK(double *PbarK, int M, int N, double *C, double *P, double *K,
                  const double reg, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks = (M * N + blockSize - 1) / blockSize;

  ip_PbarK<<<numBlocks, blockSize, 0, stream>>>(M * N, PbarK, C, P, K, reg);
}

void update_K(double *K, int M, int N, double reg, int numblocks, int blocksize,
              cudaStream_t &stream) {
  // K = exp(-C / reg)
  ip_gibbs<<<numblocks, blocksize, 0, stream>>>(M * N, K, reg);
}

void update_Kv(double *Kv, double *K, int M, int N, double *v,
               cudaStream_t &stream, cublasHandle_t &handle) {
  // Kv = K * v
  dgemv(Kv, 1.0, K, M, N, false, v, 0.0, handle);
}

void update_u(double *u, double *a, double *Kv, int M, cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_M = (M + blockSize - 1) / blockSize;
  // u = a / Kv
  nip_div<<<numBlocks_M, blockSize, 0, stream>>>(M, u, a, Kv);
}

// update v two steps in one go
void update_KTu_v(double *v, double *b, double *K, int M, int N, double *u,
                  double *KTu, cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_N = (N + blockSize - 1) / blockSize;
  // KTu = K^T * u
  dgemv(KTu, 1.0, K, M, N, true, u, 0.0, handle);
  // v = b / KTu
  nip_div<<<numBlocks_N, blockSize, 0, stream>>>(N, v, b, KTu);
}

// update error
void update_err(double &err, double *u, double *v, double *Kv, double *KTu,
                double *a, double *b, double *tmp_M, double *tmp_N, int M,
                int N, cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_M = (M + blockSize - 1) / blockSize;
  int numBlocks_N = (N + blockSize - 1) / blockSize;

  ip_dot_minus<<<numBlocks_M, blockSize, 0, stream>>>(M, tmp_M, u, Kv, a);
  ip_dot_minus<<<numBlocks_N, blockSize, 0, stream>>>(N, tmp_N, v, KTu, b);

  // compute norm
  double e_m, e_n;
  dnrm2(e_m, tmp_M, M, handle);
  dnrm2(e_n, tmp_N, N, handle);

  err = e_m + e_n;
}

void update_vbar_L(double *vbar, double *PbarK, int M, int N, double *u,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  // vbar_L = PbarK^T * u_L
  dgemv(vbar, 1.0, PbarK, M, N, true, u, 0.0, handle);
}

void update_ubar_L(double *ubar, double *vbar, double *PbarK, double *K, int M,
                   int N, double *u, double *v, double *KTu,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_N = (N + blockSize - 1) / blockSize;

  // KTu = K^T * u_L
  dgemv(KTu, 1.0, K, M, N, true, u, 0.0, handle);
  // tmp_vec_N = (vbar % v_L) / KTu
  ip_dot_div<<<numBlocks_N, blockSize, 0, stream>>>(N, vbar, v, KTu);
  // ubar_L = PbarK * vbar_L
  dgemv(ubar, 1.0, PbarK, M, N, false, v, 0.0, handle);
  // ubar_L = - K * KTu + ubar_L
  dgemv(ubar, -1.0, K, M, N, false, KTu, 1.0, handle);
}

void update_vbar_l(double *vbar, double *ubar, double *K, int M, int N,
                   double *Kv, double *u_hist, double *v_hist, int l,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_M = (M + blockSize - 1) / blockSize;
  // update Kv (not needed, as this is computed by previous abar)
  // dgemv(Kv, 1.0, K, M, N, false, v_hist + l * N, 0.0, handle);
  ip_dot_div<<<numBlocks_M, blockSize, 0, stream>>>(M, ubar,
                                                    u_hist + (l + 1) * M, Kv);
  // vbar = - K^T * Kv
  dgemv(vbar, -1.0, K, M, N, true, Kv, 0.0, handle);
}

void update_ubar_l(double *ubar, double *vbar, double *K, int M, int N,
                   double *KTu, double *u_hist, double *v_hist, int l,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_N = (N + blockSize - 1) / blockSize;
  // update KTu
  dgemv(KTu, 1.0, K, M, N, true, u_hist + l * M, 0.0, handle);
  ip_dot_div<<<numBlocks_N, blockSize, 0, stream>>>(N, vbar, v_hist + l * N,
                                                    KTu);
  // ubar = - K * KTu
  dgemv(ubar, -1.0, K, M, N, false, KTu, 0.0, handle);
}

void update_abar(double *abar, double *ubar, double *K, int M, int N,
                 double *Kv, double *v_hist, int l, cudaStream_t &stream,
                 cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_M = (M + blockSize - 1) / blockSize;
  // abar += ubar / Kv
  // update Kv
  dgemv(Kv, 1.0, K, M, N, false, v_hist + (l - 1) * N, 0.0, handle);
  ip_accu_abar<<<numBlocks_M, blockSize, 0, stream>>>(M, abar, ubar, Kv);
}

void update_P(double *P, int M, int N, double *u, double *K, double *v,
              cudaStream_t &stream) {
  // P = diag(u) * K * diag(v)
  int blockSize = BLOCK_SIZE;
  int numBlocks_MN = (M * N + blockSize - 1) / blockSize;

  nip_diag_scale<<<numBlocks_MN, blockSize, 0, stream>>>(M, N, P, u, K, v);
}

void update_loss(double &loss, double *C, double *P, int M, int N, double reg,
                 cudaStream_t &stream) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_MN = (M * N + blockSize - 1) / blockSize;

  ip_sinkloss<<<numBlocks_MN, blockSize, 0, stream>>>(M * N, &loss, C, P, reg);
}

void sinkhorn_forward(int &iter, double &err, double *u, double *v,
                      double *u_hist, double *v_hist, double *a, double *b,
                      double *K, int M, int N, double *Kv, double *KTu,
                      double *tmp_M, double *tmp_N, int max_iter,
                      double zero_tol, bool withgrad, cudaStream_t &stream,
                      cublasHandle_t &handle) {
  // forward pass of Sinkhorn
  auto D2D = cudaMemcpyDeviceToDevice;
  // auto D2H = cudaMemcpyDeviceToHost;

  // pre-compute Kv
  update_Kv(Kv, K, M, N, v, stream, handle);

  while ((iter < max_iter) && (err >= zero_tol)) {
    update_u(u, a, Kv, M, stream);
    if (withgrad) {
      cudaMemcpyAsync(u_hist + (iter + 1) * M, u, sizeof(double) * M, D2D,
                      stream);
    }

    update_KTu_v(v, b, K, M, N, u, KTu, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(v_hist + (iter + 1) * N, v, sizeof(double) * N, D2D,
                      stream);
    }

    update_Kv(Kv, K, M, N, v, stream, handle);
    update_err(err, u, v, Kv, KTu, a, b, tmp_M, tmp_N, M, N, stream, handle);
    // cudaMemcpyAsync(&err, d_err, sizeof(double), D2H, stream);
    iter++;
  }
  // done with the forward pass
  // iter_out = iter;
  // err_out = err;
}

void sinkhorn_backward(double *ubar, double *vbar, double *abar, int &iter,
                       double *u, double *v, double *u_hist, double *v_hist,
                       double *K, int M, int N, double *Kv, double *KTu,
                       double *P, double *PbarK, double *C, double reg,
                       cudaStream_t &stream, cublasHandle_t &handle) {
  // backward pass of Sinkhorn
  update_PbarK(PbarK, M, N, C, P, K, reg, stream);

  for (int l = iter; l > 0; --l) {
    if (l == iter) {
      // update vbar_L
      update_vbar_L(vbar, PbarK, M, N, u, stream, handle);
      // update ubar_L
      update_ubar_L(ubar, vbar, PbarK, K, M, N, u, v, KTu, stream, handle);
    } else {
      // update vbar_l
      update_vbar_l(vbar, ubar, K, M, N, Kv, u_hist, v_hist, l, stream, handle);
      // update ubar_l
      update_ubar_l(ubar, vbar, K, M, N, KTu, u_hist, v_hist, l, stream,
                    handle);
    }
    // accumulate abar
    update_abar(abar, ubar, K, M, N, Kv, v_hist, l, stream, handle);
  }
}

// sinkhorn CUDA

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

  int iter = 0;
  double err = 1000.;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: allocate memory for the variables */
  cudaMalloc(reinterpret_cast<void **>(&d_a), sizeof(double) * M);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_u), sizeof(double) * M);
  cudaMalloc(reinterpret_cast<void **>(&d_v), sizeof(double) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_P), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_Kv), sizeof(double) * M);
  cudaMalloc(reinterpret_cast<void **>(&d_KTu), sizeof(double) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_loss), sizeof(double));
  cudaMemset(d_loss, 0, sizeof(double)); // set loss to zero

  cudaMalloc(reinterpret_cast<void **>(&d_tmp_M), sizeof(double) * M);
  cudaMalloc(reinterpret_cast<void **>(&d_tmp_N), sizeof(double) * N);

  if (withgrad) {
    cudaMalloc(reinterpret_cast<void **>(&d_u_hist),
               sizeof(double) * M * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_v_hist),
               sizeof(double) * N * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_PbarK), sizeof(double) * M * N);
    cudaMalloc(reinterpret_cast<void **>(&d_ubar), sizeof(double) * M);
    cudaMalloc(reinterpret_cast<void **>(&d_vbar), sizeof(double) * N);
    cudaMalloc(reinterpret_cast<void **>(&d_abar), sizeof(double) * M);

    cudaMemset(d_ubar, 0, sizeof(double) * M);
    cudaMemset(d_vbar, 0, sizeof(double) * N);
    cudaMemset(d_abar, 0, sizeof(double) * M);
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

  /* step 4: forward pass */
  sinkhorn_forward(iter, err, d_u, d_v, d_u_hist, d_v_hist, d_a, d_b, d_K, M, N,
                   d_Kv, d_KTu, d_tmp_M, d_tmp_N, max_iter, zero_tol, withgrad,
                   stream, handle);

  // compute optimal coupling P
  update_P(d_P, M, N, d_u, d_K, d_v, stream);

  /* step 5: backward pass */
  if (withgrad) {
    sinkhorn_backward(d_ubar, d_vbar, d_abar, iter, d_u, d_v, d_u_hist,
                      d_v_hist, d_K, M, N, d_Kv, d_KTu, d_P, d_PbarK, d_C, reg,
                      stream, handle);
  }

  update_loss(*d_loss, d_C, d_P, M, N, reg, stream);

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
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_K);
  cudaFree(d_C);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_P);
  cudaFree(d_Kv);
  cudaFree(d_KTu);
  cudaFree(d_loss);
  cudaFree(d_tmp_M);
  cudaFree(d_tmp_N);
  if (withgrad) {
    cudaFree(d_u_hist);
    cudaFree(d_v_hist);
    cudaFree(d_PbarK);
    cudaFree(d_ubar);
    cudaFree(d_vbar);
    cudaFree(d_abar);
  }
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}

/*
  TODO: kernels to be moved
*/

void update_KV(double *KV, double *K, double *V, int M, int N, int S,
               cublasHandle_t &handle) {
  // KV = K * V
  dgemm(KV, 1.0, K, false, V, false, M, S, N, 0.0, handle);
}

void update_U(double *U, double *V, double *A, double *K, double *KV, int M,
              int N, int S, cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_MS = (M * S + blockSize - 1) / blockSize;
  // U = A ./ KV
  nip_div<<<numBlocks_MS, blockSize, 0, stream>>>(M * S, U, A, KV);
}

void update_b(double *b, double *K, double *U, double *w, double *KTU,
              double *tmp_NS, int M, int N, int S, cudaStream_t &stream,
              cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  // int numBlocks_N = (N + blockSize - 1) / blockSize;
  int numBlocks_NS = (N * S + blockSize - 1) / blockSize;
  int share_size = blockSize * sizeof(double);
  // KTU = K^T * U
  dgemm(KTU, 1.0, K, true, U, false, N, S, M, 0.0, handle);
  // cudaStreamSynchronize(stream);

  // FIXME: rowwise product!
  // logb = log(KTU) * w, b = exp(logb)
  // nip_log<<<numBlocks_NS, blockSize, 0, stream>>>(N * S, tmp_NS, KTU);
  // dgemv(b, 1.0, tmp_NS, N, S, false, w, 0.0, handle);
  // ip_exp<<<numBlocks_NS, blockSize, 0, stream>>>(N, b);

  nip_KTU_w<<<numBlocks_NS, blockSize, 0, stream>>>(N, S, KTU, w, tmp_NS);
  // print tmp_NS
  // print_device_matrix(w, S, 1, "w");
  // print_device_matrix(tmp_NS, N, S, "KTU^w (tmp_NS)");
  // nip_rowprod<<<numBlocks_N, blockSize, 0, stream>>>(N, S, tmp_NS, b);
  // print_device_matrix(b, N, 1, "b");
  nip_rowprod_shared<<<N, blockSize, share_size, stream>>>(N, S, tmp_NS, b);
}

void update_V(double *V, double *U, double *b, double *K, double *KTU, int M,
              int N, int S, cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_NS = (N * S + blockSize - 1) / blockSize;
  // KTU = K^T * U
  // dgemm(KTU, 1.0, K, true, U, false, N, S, M, 0.0, handle);
  nip_b_div_KTU<<<numBlocks_NS, blockSize, 0, stream>>>(N, S, V, b, KTU);
}

void update_bbar_L(double *bbar, double *b, double *b_ext, int N,
                   cudaStream_t &stream, cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_N = (N + blockSize - 1) / blockSize;
  // bbar_L = 2 ( b_L - b_ext )
  nip_minus<<<numBlocks_N, blockSize, 0, stream>>>(N, bbar, b, b_ext);
  scal(bbar, N, 2.0, handle);
}

void update_Ubar_L(double *Ubar, double *V, double *b, double *w, double *K,
                   double *tmp_NS, int M, int N, int S, cudaStream_t &stream,
                   cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_NS = (N * S + blockSize - 1) / blockSize;
  // compute outer product b * w^T
  // set the tem results into zero
  cudaMemset(tmp_NS, 0, N * S * sizeof(double));
  dger(tmp_NS, N, S, 1.0, b, w, handle);
  // tmp_NS = tmp_NS % V_L
  ip_dot<<<numBlocks_NS, blockSize, 0, stream>>>(N * S, V, tmp_NS);
  // Ubar = K * tmp_NS
  dgemm(Ubar, 1.0, K, false, tmp_NS, false, M, S, N, 0.0, handle);
}

void update_Vbar_l(double *Vbar, double *Ubar, double *K, int M, int N, int S,
                   double *KV, double *V_hist, int l, cudaStream_t &stream,
                   cublasHandle_t &handle) {
  // Vbar only has l = 1, ..., L-1
  int blockSize = BLOCK_SIZE;
  int numBlocks_NS = (N * S + blockSize - 1) / blockSize;
  // update KV
  dgemm(KV, 1.0, K, false, V_hist + l * (M * S), false, M, S, N, 0.0, handle);
  ip_dot_div<<<numBlocks_NS, blockSize, 0, stream>>>(N * S, Vbar,
                                                     V_hist + l * (M * S), KV);
  // Vbar = - K^T * KV
  dgemm(Vbar, -1.0, K, true, KV, false, N, S, M, 0.0, handle);
}

void update_bbar_l() {
  // sum_row Vbar / KTU => (Vbar / KTU) * 1_S
}

void update_Ubar_l() {
  //
}

void update_err(double &err, double *U, double *V, double *KV, double *A,
                double *tmp_MS, int M, int S, cudaStream_t &stream,
                cublasHandle_t &handle) {
  int blockSize = BLOCK_SIZE;
  int numBlocks_MS = (M * S + blockSize - 1) / blockSize;

  ip_dot_minus<<<numBlocks_MS, blockSize, 0, stream>>>(M * S, tmp_MS, U, KV, A);

  // compute norm
  dnrm2(err, tmp_MS, M * S, handle);
}

void barycenter_forward(int &iter, double &err, double *U, double *V,
                        double *U_hist, double *V_hist, double *b_hist,
                        double *b, double *K, double *A, double *w, double *KV,
                        double *KTU, double *tmp_MS, double *tmp_NS, int M,
                        int N, int S, int max_iter, double zero_tol,
                        bool withgrad, cudaStream_t &stream,
                        cublasHandle_t &handle) {
  // forward pass
  auto D2D = cudaMemcpyDeviceToDevice;

  // pre-compute KV
  update_KV(KV, K, V, M, N, S, handle);

  while ((iter < max_iter) && (err >= zero_tol)) {
    update_U(U, V, A, K, KV, M, N, S, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(U_hist + (iter + 1) * M * S, U, sizeof(double) * M * S,
                      D2D, stream);
    }

    update_b(b, K, U, w, KTU, tmp_NS, M, N, S, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(b_hist + (iter + 1) * N, b, sizeof(double) * N, D2D,
                      stream);
    }

    update_V(V, U, b, K, KTU, M, N, S, stream, handle);
    if (withgrad) {
      cudaMemcpyAsync(V_hist + (iter + 1) * N * S, V, sizeof(double) * N * S,
                      D2D, stream);
    }

    update_KV(KV, K, V, M, N, S, handle);
    update_err(err, U, V, KV, A, tmp_MS, M, S, stream, handle);
    iter++;
  }
}

void barycenter_backward() {
  // backward pass
}

// implement the cuda barycenter_parallel function
void cuda_barycenter_parallel(double *b, double *grad_A, double *grad_w,
                              double *U, double *V, double *loss, int *iter_out,
                              double *err_out, const double *A, const double *C,
                              const double *w, const double *b_ext, const int M,
                              const int N, const int S, const double reg,
                              const bool withgrad, const int max_iter,
                              const double zero_tol) {
  cublasHandle_t handle;
  cudaStream_t stream;
  auto H2D = cudaMemcpyHostToDevice;
  auto D2H = cudaMemcpyDeviceToHost;
  auto D2D = cudaMemcpyDeviceToDevice;

  // data: A, b_ext, C, w
  double *d_A = nullptr, *d_b_ext = nullptr, *d_C = nullptr, *d_w = nullptr;
  double *d_K = nullptr;
  // scaling matrices
  double *d_U = nullptr, *d_V = nullptr;
  // output b
  double *d_b = nullptr;
  // temp matrices
  double *d_KV = nullptr, *d_KTU = nullptr;
  // loss
  double *d_loss = nullptr;

  // history of U and V
  double *d_U_hist = nullptr, *d_V_hist = nullptr, *d_b_hist = nullptr;
  // adjoints
  double *d_Ubar = nullptr, *d_Vbar = nullptr, *d_bbar = nullptr;
  double *d_Abar = nullptr, *d_wbar = nullptr;
  // tmp mats
  double *d_tmp_MS = nullptr, *d_tmp_NS = nullptr, *d_ones_S = nullptr;

  int iter = 0;
  double err = 1000.;

  /* step 1: create cublas handle, bind a stream */
  cublasCreate(&handle);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cublasSetStream(handle, stream);

  /* step 2: allocate memory for the variables */
  cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_b_ext), sizeof(double) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_C), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_w), sizeof(double) * S);
  cudaMalloc(reinterpret_cast<void **>(&d_K), sizeof(double) * M * N);
  cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&d_b), sizeof(double) * N);
  cudaMalloc(reinterpret_cast<void **>(&d_KV), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_KTU), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&d_loss), sizeof(double));
  cudaMemset(d_loss, 0, sizeof(double)); // set loss to zero

  cudaMalloc(reinterpret_cast<void **>(&d_tmp_MS), sizeof(double) * M * S);
  cudaMalloc(reinterpret_cast<void **>(&d_tmp_NS), sizeof(double) * N * S);
  cudaMalloc(reinterpret_cast<void **>(&d_ones_S), sizeof(double) * S);

  if (withgrad) {
    cudaMalloc(reinterpret_cast<void **>(&d_U_hist),
               sizeof(double) * M * S * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_V_hist),
               sizeof(double) * N * S * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_b_hist),
               sizeof(double) * N * (max_iter + 1));
    cudaMalloc(reinterpret_cast<void **>(&d_Ubar), sizeof(double) * M * S);
    cudaMalloc(reinterpret_cast<void **>(&d_Vbar), sizeof(double) * N * S);
    cudaMalloc(reinterpret_cast<void **>(&d_bbar), sizeof(double) * N);
    cudaMalloc(reinterpret_cast<void **>(&d_Abar), sizeof(double) * M * S);
    cudaMalloc(reinterpret_cast<void **>(&d_wbar), sizeof(double) * S);

    cudaMemset(d_Ubar, 0, sizeof(double) * M * S);
    cudaMemset(d_Vbar, 0, sizeof(double) * N * S);
    cudaMemset(d_Abar, 0, sizeof(double) * M * S);
  }

  /* step 3: copy data to device */
  cudaMemcpyAsync(d_A, A, sizeof(double) * M * S, H2D, stream);
  cudaMemcpyAsync(d_b_ext, b_ext, sizeof(double) * N, H2D, stream);
  cudaMemcpyAsync(d_C, C, sizeof(double) * M * N, H2D, stream);
  cudaMemcpyAsync(d_w, w, sizeof(double) * S, H2D, stream);

  // compute the Gibbs kernel K
  cudaMemcpyAsync(d_K, C, sizeof(double) * M * N, H2D, stream);
  update_K(d_K, M, N, reg, stream);

  // init U, V to ones
  init_ones(d_U, M * S, stream);
  init_ones(d_V, N * S, stream);
  init_ones(d_ones_S, S, stream);

  if (withgrad) {
    cudaMemcpyAsync(d_U_hist + iter * M * S, d_U, sizeof(double) * M * S, D2D,
                    stream);
    cudaMemcpyAsync(d_V_hist + iter * N * S, d_V, sizeof(double) * N * S, D2D,
                    stream);
  }

  /* step 4: forward pass */
  barycenter_forward(iter, err, d_U, d_V, d_U_hist, d_V_hist, d_b_hist, d_b,
                     d_K, d_A, d_w, d_KV, d_KTU, d_tmp_MS, d_tmp_NS, M, N, S,
                     max_iter, zero_tol, withgrad, stream, handle);

  /* step 5: backward pass */
  // TODO: add later

  /* step 6: copy data back to host */
  cudaMemcpyAsync(U, d_U, sizeof(double) * M * S, D2H, stream);
  cudaMemcpyAsync(V, d_V, sizeof(double) * N * S, D2H, stream);
  cudaMemcpyAsync(b, d_b, sizeof(double) * N, D2H, stream);
  cudaMemcpyAsync(loss, d_loss, sizeof(double), D2H, stream);
  if (withgrad) {
    // copy d_Abar to grad_A
    cudaMemcpyAsync(grad_A, d_Abar, sizeof(double) * M * S, D2H, stream);
    // copy d_bbar to grad_w
    cudaMemcpyAsync(grad_w, d_wbar, sizeof(double) * S, D2H, stream);
  }
  cudaStreamSynchronize(stream);
  // other info
  *iter_out = iter;
  *err_out = err;

  /* step 7: free resources */
  cudaFree(d_A);
  cudaFree(d_b_ext);
  cudaFree(d_C);
  cudaFree(d_w);
  cudaFree(d_K);
  cudaFree(d_U);
  cudaFree(d_V);
  cudaFree(d_b);
  cudaFree(d_KV);
  cudaFree(d_KTU);
  cudaFree(d_loss);
  cudaFree(d_tmp_MS);
  cudaFree(d_tmp_NS);
  cudaFree(d_ones_S);
  if (withgrad) {
    cudaFree(d_U_hist);
    cudaFree(d_V_hist);
    cudaFree(d_b_hist);
    cudaFree(d_Ubar);
    cudaFree(d_Vbar);
    cudaFree(d_bbar);
    cudaFree(d_Abar);
    cudaFree(d_wbar);
  }
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
}
