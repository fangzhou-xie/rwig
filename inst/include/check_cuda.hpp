// header to check for CUDA availability
// HAVE_CUBLAS and HAVE_CUDA_RUNTIME are defined by autoconf
// via -D flags in PKG_CXXFLAGS when CUDA is detected

#ifndef RWIG_CHECK_CUDA_H
#define RWIG_CHECK_CUDA_H

#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#endif // RWIG_CHECK_CUDA_H
