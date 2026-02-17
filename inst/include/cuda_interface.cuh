// cuda header interface for the OT functions in C++

#ifndef RWIG_CUDA_INTERFACE_CUH
#define RWIG_CUDA_INTERFACE_CUH

#ifdef HAVE_CUBLAS
#ifdef HAVE_CUDA_RUNTIME

/*
  interface to C++ side (then to R)
*/

void cuda_sinkhorn_vanilla(double *P, double *grad_a, double *u, double *v,
                           double *loss, int *iter_out, double *err_out,
                           const double *a, const double *b, const double *C,
                           const int m, const int n, const double reg,
                           const bool withgrad, const int max_iter,
                           const double zero_tol);

void cuda_barycenter_parallel(double *U, double *V, double *b, double *grad_A,
                              double *grad_w, double *loss, int *iter_out,
                              double *err_out, const double *A, const double *w,
                              const double *C, const double *b_ext, const int M,
                              const int N, const int S, const double reg,
                              const bool withgrad, const int max_iter,
                              const double zero_tol);

// TODO: maybe the intermediate barycenter interface too (for WDL)
// TODO: implement wdl (high priority)

void cuda_wdl();

// TODO: implement log methods (low priority)

#endif
#endif

#endif // RWIG_CUDA_INTERFACE_CUH
