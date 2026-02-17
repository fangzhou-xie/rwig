// header to check for CUDA availability and version

#ifndef RWIG_CHECK_CUDA_H
#define RWIG_CHECK_CUDA_H

#if __has_include(<cublas_v2.h>)
#include <cublas_v2.h>
#define HAVE_CUBLAS 1
#else
#define HAVE_CUBLAS 0
#endif

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define HAVE_CUDA_RUNTIME 1
#else
#define HAVE_CUDA_RUNTIME 0
#endif

namespace check_cuda {
inline constexpr bool has_cublas = HAVE_CUBLAS;
inline constexpr bool has_cuda_runtime = HAVE_CUDA_RUNTIME;
inline constexpr bool has_cuda = HAVE_CUBLAS && HAVE_CUDA_RUNTIME;
} // namespace check_cuda

#endif // RWIG_CHECK_CUDA_H
