// implement optimizers: SGD, Adam, AdamW
// mirrors optimizer.hpp for CUDA

#include "cuda_kernels.cuh"

/*
  Optimizer kernels
*/

// SGD: theta = theta - eta * g
__global__ void opt_sgd(int n, double *theta, double *g, double eta) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    theta[i] = theta[i] - eta * g[i];
}

// Adam with decoupled weight decay `gamma` (gamma = 0 gives plain Adam):
// update m, v in-place, then update theta
__global__ void opt_adam(int n, double *theta, double *g, double *m, double *v,
                         double eta, double gamma, double beta1, double beta2,
                         double eps, int step) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  double bc1 = 1.0 - pow(beta1, step);
  double bc2 = 1.0 - pow(beta2, step);
  for (int i = index; i < n; i += stride) {
    m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
    v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
    double mhat = m[i] / bc1;
    double vhat = v[i] / bc2;
    theta[i] =
        (1.0 - eta * gamma) * theta[i] - eta * (mhat / (sqrt(vhat) + eps));
  }
}

/*
  Wrapper functions

  opt = 0: SGD
  opt = 1: Adam
  opt = 2: AdamW

  d_m, d_v: device momentum buffers (zeroed by caller, unused for SGD)
  step: current step counter (managed by caller)
*/

void optimizer_step(double *d_theta, double *d_g, double *d_m, double *d_v,
                    int opt, int n, double eta, double gamma, double beta1,
                    double beta2, double eps, int step, cudaStream_t &stream) {
  if (opt == 0) {
    opt_sgd<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(n, d_theta, d_g, eta);
  } else {
    const double decay = (opt == 2) ? gamma : 0.0;
    opt_adam<<<nblocks(n), BLOCK_SIZE, 0, stream>>>(
        n, d_theta, d_g, d_m, d_v, eta, decay, beta1, beta2, eps, step);
  }
}
