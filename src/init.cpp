// registration of the .Call entry points

#include "r_glue.hpp"

#include <R_ext/Rdynload.h>

extern "C" {
SEXP rwig_sinkhorn_vanilla_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                               SEXP);
SEXP rwig_sinkhorn_log_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                           SEXP);
SEXP rwig_barycenter_parallel_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                                  SEXP, SEXP, SEXP);
SEXP rwig_barycenter_log_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                             SEXP, SEXP);
SEXP rwig_wdl_cpp(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP,
                  SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
SEXP rwig_tsvd_cpp(SEXP, SEXP, SEXP);
SEXP rwig_euclidean_cpp(SEXP);
SEXP rwig_doc2dist_cpp(SEXP, SEXP);
SEXP rwig_cuda_available_cpp(void);
}

#define CALLDEF(name, n) {#name, (DL_FUNC)&rwig_##name, n}

static const R_CallMethodDef CallEntries[] = {
    CALLDEF(sinkhorn_vanilla_cpp, 9),
    CALLDEF(sinkhorn_log_cpp, 9),
    CALLDEF(barycenter_parallel_cpp, 10),
    CALLDEF(barycenter_log_cpp, 10),
    CALLDEF(wdl_cpp, 19),
    CALLDEF(tsvd_cpp, 3),
    CALLDEF(euclidean_cpp, 1),
    CALLDEF(doc2dist_cpp, 2),
    CALLDEF(cuda_available_cpp, 0),
    {NULL, NULL, 0}};

extern "C" void R_init_rwig(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
  R_forceSymbols(dll, TRUE);
}
