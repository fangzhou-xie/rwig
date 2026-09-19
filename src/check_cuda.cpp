// check cuda availability

#include "r_glue.hpp"

extern "C" SEXP rwig_cuda_available_cpp(void) {
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  return Rf_ScalarLogical(TRUE);
#else
  return Rf_ScalarLogical(FALSE);
#endif
}
