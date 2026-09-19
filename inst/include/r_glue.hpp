// Glue between R's C API and the la::Vec / la::Mat containers.
//
// Everything R-facing lives here: argument coercion, result construction
// with PROTECT bookkeeping, messages, interrupts and the try/catch wrapper
// that turns C++ exceptions into R errors at the .Call boundary.

#ifndef RWIG_R_GLUE_H
#define RWIG_R_GLUE_H

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#ifndef STRICT_R_HEADERS
#define STRICT_R_HEADERS
#endif
#include <R.h>
#include <Rinternals.h>
#include <R_ext/Utils.h> // R_CheckUserInterrupt

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "linalg.hpp"

namespace rr {

/*
  Messages and interrupts
*/

// same as message() in R, so suppressMessages() still applies
inline void message(const std::string &msg) {
  SEXP txt = PROTECT(Rf_mkString(msg.c_str()));
  SEXP call = PROTECT(Rf_lang2(Rf_install("message"), txt));
  Rf_eval(call, R_BaseEnv);
  UNPROTECT(2);
}

struct interrupt_exception : public std::runtime_error {
  interrupt_exception() : std::runtime_error("user interrupt") {}
};

// R_CheckUserInterrupt() longjmps past C++ destructors; run it inside a
// top-level context instead and throw, so worker threads and buffers are
// released before the error reaches R (same pattern as cuda_wdl.cu)
inline void check_interrupt_fn(void *) { R_CheckUserInterrupt(); }
inline void check_interrupt() {
  if (!R_ToplevelExec(check_interrupt_fn, nullptr)) throw interrupt_exception();
}

/*
  Arguments
*/

inline double as_double(SEXP x) { return Rf_asReal(x); }
inline int as_int(SEXP x) { return Rf_asInteger(x); }
inline bool as_bool(SEXP x) { return Rf_asLogical(x) == TRUE; }

// PROTECT bookkeeping for a scope: everything added is unprotected in the
// destructor (LIFO with respect to other protections made after it)
class Protector {
public:
  Protector() = default;
  ~Protector() { UNPROTECT(_n); }
  Protector(const Protector &) = delete;
  Protector &operator=(const Protector &) = delete;
  SEXP add(SEXP x) {
    PROTECT(x);
    ++_n;
    return x;
  }

private:
  int _n = 0;
};

// numeric input as a double vector/matrix (integer and logical are coerced);
// the returned SEXP is protected by `p`
inline SEXP as_real(SEXP x, Protector &p) {
  return TYPEOF(x) == REALSXP ? x : p.add(Rf_coerceVector(x, REALSXP));
}

inline la::Vec vec_from_R(SEXP x) {
  Protector p;
  SEXP y = as_real(x, p);
  return la::Vec(REAL(y), (la::idx)Rf_xlength(y));
}

inline la::Mat mat_from_R(SEXP x) {
  Protector p;
  SEXP y = as_real(x, p);
  return la::Mat(REAL(y), (la::idx)Rf_nrows(x), (la::idx)Rf_ncols(x));
}

/*
  Results (allocations are returned unprotected; protect on the caller side)
*/

inline SEXP to_R(const la::Vec &v) {
  SEXP out = Rf_allocVector(REALSXP, (R_xlen_t)v.size());
  std::copy(v.data(), v.data() + v.size(), REAL(out));
  return out;
}

inline SEXP to_R(const la::Mat &m) {
  SEXP out = Rf_allocMatrix(REALSXP, (int)m.nrow(), (int)m.ncol());
  std::copy(m.data(), m.data() + m.size(), REAL(out));
  return out;
}

inline SEXP alloc_matrix(int nrow, int ncol, Protector &p) {
  return p.add(Rf_allocMatrix(REALSXP, nrow, ncol));
}

inline SEXP alloc_vector(R_xlen_t n, Protector &p) {
  return p.add(Rf_allocVector(REALSXP, n));
}

// named list builder: values are protected as they are added and released
// when the list is built (the list itself is returned unprotected)
class ListBuilder {
public:
  ListBuilder &add(const char *name, SEXP value) {
    PROTECT(value);
    ++_nprot;
    _names.push_back(name);
    _values.push_back(value);
    return *this;
  }
  ListBuilder &add(const char *name, double value) {
    return add(name, Rf_ScalarReal(value));
  }
  ListBuilder &add(const char *name, int value) {
    return add(name, Rf_ScalarInteger(value));
  }
  ListBuilder &add(const char *name, bool value) {
    return add(name, Rf_ScalarLogical(value ? TRUE : FALSE));
  }

  SEXP build() {
    const int n = (int)_values.size();
    SEXP out = PROTECT(Rf_allocVector(VECSXP, n));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, n));
    for (int i = 0; i < n; ++i) {
      SET_VECTOR_ELT(out, i, _values[i]);
      SET_STRING_ELT(names, i, Rf_mkChar(_names[i].c_str()));
    }
    Rf_setAttrib(out, R_NamesSymbol, names);
    UNPROTECT(2 + _nprot);
    _nprot = 0;
    _values.clear();
    _names.clear();
    return out;
  }

private:
  std::vector<std::string> _names;
  std::vector<SEXP> _values;
  int _nprot = 0;
};

/*
  RNG state (needed around norm_rand() / unif_rand())
*/

struct RNGScope {
  RNGScope() { GetRNGstate(); }
  ~RNGScope() { PutRNGstate(); }
};

/*
  .Call boundary: run `f`, turning any C++ exception into an R error once
  every C++ object has been destroyed (Rf_error longjmps)
*/

template <typename F> inline SEXP call_guard(F &&f) {
  static char buf[1024];
  bool failed = false;
  try {
    return f();
  } catch (const std::exception &e) {
    std::snprintf(buf, sizeof(buf), "%s", e.what());
    failed = true;
  } catch (...) {
    std::snprintf(buf, sizeof(buf), "unknown C++ exception");
    failed = true;
  }
  if (failed) Rf_error("%s", buf);
  return R_NilValue; // not reached
}

} // namespace rr

#endif // RWIG_R_GLUE_H
