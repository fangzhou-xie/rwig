// check cuda availability and versions

#include <Rcpp.h>

// [[Rcpp::export]]
bool cuda_available_cpp() {
#if defined(HAVE_CUBLAS) && defined(HAVE_CUDA_RUNTIME)
  return true;
#else
  return false;
#endif
}
