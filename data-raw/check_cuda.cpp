// g++ --std=c++17 -I/usr/local/cuda/include -o check_cuda.cpp

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#define HAVE_CUBLAS 1
#else
#define HAVE_CUBLAS 0
#endif

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define HAVE_CUDA 1
#else
#define HAVE_CUDA 0
#endif

namespace cuda_compat {
inline constexpr bool has_cublas = HAVE_CUBLAS;
inline constexpr bool has_cuda = HAVE_CUDA;

// CUDA Runtime version
#if HAVE_CUDA
inline constexpr int cudart_version = CUDART_VERSION;
inline constexpr int cudart_major = CUDART_VERSION / 1000;
inline constexpr int cudart_minor = (CUDART_VERSION % 1000) / 10;
#else
inline constexpr int cudart_version = 0;
inline constexpr int cudart_major = 0;
inline constexpr int cudart_minor = 0;
#endif

// cuBLAS version
#if HAVE_CUBLAS
inline constexpr int cublas_major = CUBLAS_VER_MAJOR;
inline constexpr int cublas_minor = CUBLAS_VER_MINOR;
inline constexpr int cublas_patch = CUBLAS_VER_PATCH;
#else
inline constexpr int cublas_major = 0;
inline constexpr int cublas_minor = 0;
inline constexpr int cublas_patch = 0;
#endif
}

#include <iostream>

int main() {
  std::cout << "CUDA detected: " << (cuda_compat::has_cuda ? "yes" : "no") << "\n";
  std::cout << "cuBLAS detected: " << (cuda_compat::has_cublas ? "yes" : "no") << "\n";

  if constexpr (cuda_compat::has_cuda) {
    std::cout << "CUDA Runtime: " << cuda_compat::cudart_major
              << "." << cuda_compat::cudart_minor << "\n";
  }

  if constexpr (cuda_compat::has_cublas) {
    std::cout << "cuBLAS: " << cuda_compat::cublas_major
              << "." << cuda_compat::cublas_minor
              << "." << cuda_compat::cublas_patch << "\n";
  }

  return 0;
}
