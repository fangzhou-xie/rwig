#include <cpp11.hpp>

#include "ctrack.hpp"

#include "blas_kernels.hpp"  // BLAS kernels
#include "blas_utils.hpp"
#include "barycenter_blas.hpp"

// #include "wig_blas_utils.hpp"

using namespace wig;

using uint = unsigned int;


void Barycenter::compute_parallel() {
  CTRACK;
  reset_counter();

  // std::cout << "0" << std::endl;

  // init
  _U = ones(_M, _S);
  _V = ones(_N, _S);
  _K = C2K(_C, _reg);
  b = zeros(_N);

  // print_mat(_U, "U:");
  // print_mat(_V, "V:");
  // print_mat(_K, "K:");
  // print_vec(b, "b:");
  print_mat(_A, "A");
  print_mat(_C, "C");
  print_vec(_w, "w");

  impl_parallel();
}

void Barycenter::impl_parallel() {
  CTRACK;

  // std::cout << "1" << std::endl;

  // reset vars
  // reset(b, 0);
  // reset(_U, 1.);
  // reset(_V, 1.);

  // init KV, KTU
  writable::doubles _KV(_M*_S), _KTU(_N*_S);
  set_dim(_KV, _M, _S);
  set_dim(_KTU, _N, _S);

  // compute onesN * w.t();
  writable::doubles onesNwT(_N*_S);
  double * o_ = REAL(onesNwT.data());
  double * w_ = REAL(_w.data());
  for (uint j{0}; j < _S; ++j) {
    for (uint i{0}; i < _N; ++i) {
      o_[i + j * _N] = w_[i];
    }
  }

  print_mat(onesNwT, "onesNwT");

  while ((_iter < _maxiter) && (_err >= _zerotol)) {
    update_parallel_U(_KV);
    update_parallel_b(onesNwT);
    update_parallel_V(_KTU);

    _iter++;
    _err =  norm(_U, _KV, _A);
  }

}

void Barycenter::update_parallel_U(writable::doubles& _KV) {
  CTRACK;

  if (_C_is_symm) {
    dsymm(_KV, _K, _V, true, false, 1., 0.);
  } else {
    dgemm(_KV, _K, _V, false, false, 1., 0.);
  }

  over(_U, _A, _KV);
}

void Barycenter::update_parallel_V(writable::doubles& _KTU) {
  CTRACK;

  if (_C_is_symm) {
    dsymm(_KTU, _K, _U, true, false, 1., 0.);
  } else {
    dgemm(_KTU, _K, _U, true, false, 1., 0.);
  }

  double * V_ = REAL(_V.data());
  double * b_ = REAL(b.data());
  double * KTU_ = REAL(_KTU.data());
  for (uint j{0}; j < _S; ++j) {
    for (uint i{0}; i < _M; ++i) {
      V_[i + j * _N] = b_[j] / KTU_[i + j * _N];
    }
  }
}

void Barycenter::update_parallel_b(const doubles& onesNwT) {
  CTRACK;
}
