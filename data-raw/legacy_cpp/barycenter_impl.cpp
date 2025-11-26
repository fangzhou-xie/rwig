
// actual implementation of the Barycenter class

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>

#include "barycenter.hpp"

#include "ctrack.hpp"

using namespace arma;

/////////////////////////////////////////////////////////////////////////
// Algo 6.1/5.1: Parallel Barycenter with/without Jacobian wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for parallel barycenter
void Barycenter::compute_parallel() {
  CTRACK;

  // reset the counter
  reset_counter();

  // init the vars
  init_parallel();

  // start the computation
  compute_parallel_impl();

  // ctrack::result_print();
}


/////////////////////////////////////////////////////////////////////////
// Algo 6.2/5.2: Log Barycenter with/without Jacobian wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for log barycenter
void Barycenter::compute_log() {

  // reset the counter
  reset_counter();

  // init the vars
  init_log();

  // start the computation
  compute_log_impl();

  // ctrack::result_print();
}



/////////////////////////////////////////////////////////////////////////
// private methods for Barycenter class
/////////////////////////////////////////////////////////////////////////

// init the vars used in parallel method
void Barycenter::init_parallel() {
  CTRACK;
  // _M, _N, _S => init all the hidden vars
  // the main loop only needs to call `.zeros()`

  _U = mat(_M, _S, fill::ones); // U
  _V = mat(_N, _S, fill::ones); // V
  _K = exp( - _C / _reg); // K
  _KT = _K.t();

  if (_withjac) {
    // U,V,b wrt A
    _JUA = mat(_M*_S, _M*_S, fill::zeros);
    _JVA = mat(_N*_S, _M*_S, fill::zeros);
    _JbA = mat(_N, _M*_S, fill::zeros);
    // U,V,b wrt w
    _JUw = mat(_M*_S, _S, fill::zeros);
    _JVw = mat(_N*_S, _S, fill::zeros);
    _Jbw = mat(_N, _S, fill::zeros);
  }
}

void Barycenter::compute_parallel_impl() {
  CTRACK;

  // reset output b
  b.zeros();

  // reset intermediate vars
  _U.ones();
  _V.ones();

  if (_withjac) {
    // U,V,b wrt A
    _JUA.zeros();
    _JVA.zeros();
    _JbA.zeros();
    // U,V,b wrt w
    _JUw.zeros();
    _JVw.zeros();
    _Jbw.zeros();
  }

  // intermediate vars
  mat onesNwT = _onesN * _w.t();
  // for `update_parallel_JU2()`
  // _KTU = _KT * _U;

  while ((_iter < _maxiter) && (_err >= _zerotol)) {
    _timer.tic();

    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    update_parallel_U();
    update_parallel_b(onesNwT);
    update_parallel_V();

    // termination condition
    _iter++;
    _err = norm((_U % (_K * _V)) - _A, 2);
    _timer.toc();
    // logging
    if ((_verbose != 0) && ((_iter-1) % _verbose) == 0) {
      cpp11::message(
        "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
        _iter, _err,
        _timer.speed_last(), _timer.speed_avg()
      );
    }
  }

  // rescale to 1
  b = b / accu(b);
}


void Barycenter::update_parallel_JU() {
  CTRACK;

  mat _diag1kv = mat(_M, _M, fill::zeros);
  // mat akv2_onesM = mat(_M, _M, fill::zeros);
  mat _diagukv = mat(_M, _M, fill::zeros);

  for (uword t = 0; t < _S; ++t) {
    _diag1kv = diagmat(1 / _KV.col(t));
    _diagukv = diagmat(_U.col(t) / _KV.col(t));

    // update JUA
    for (uword s = 0; s < _S; ++s) {

      cpp11::check_user_interrupt();

      if (t == s) {
        _JUA(_M*t, _M*s, size(_M, _M)) =
          _diag1kv - _diagukv * _K * _JVA(_N*t, _M*s, size(_N, _M));
      } else {
        _JUA(_M*t, _M*s, size(_M, _M)) =
          - _diagukv * _K * _JVA(_N*t, _M*s, size(_N, _M));
      }
    }
    // update JUw
    _JUw(_M*t, 0, size(_M, _S)) = - _diagukv * _K * _JVw(_N*t, 0, size(_N, _S));
  }
}

void Barycenter::update_parallel_JV() {
  CTRACK;
  mat _diag1ktu = mat(_N, _N, fill::zeros);
  mat _diagvktu = mat(_N, _N, fill::zeros);

  for (uword t = 0; t < _S; ++t) {
    _diag1ktu = diagmat(1 / _KTU.col(t));
    _diagvktu = diagmat(_V.col(t) / _KTU.col(t));

    // update JVA
    for (uword s = 0; s < _S; ++s) {

      cpp11::check_user_interrupt();

      _JVA(_N*t, _M*s, size(_N, _M)) =
        _diag1ktu * _JbA(0, _M*s, size(_N, _M)) -
        _diagvktu * _KT * _JUA(_M*t, _M*s, size(_M, _M));
    }
    // update JVw
    _JVw(_N*t, 0, size(_N, _S)) =
      _diag1ktu * _Jbw
    - _diagvktu * _KT * _JUw(_M*t, 0, size(_M, _S));
  }
}

void Barycenter::update_parallel_Jb() {
  CTRACK;
  mat Jlogb_as = mat(_N, _M, fill::zeros);
  mat logKTU = log(_KTU);
  mat _diag1ktu = mat(_N, _N, fill::zeros);

  // update JbA: N*MS
  for (uword s = 0; s < _S; ++s) {
    Jlogb_as.zeros();
    for (uword t = 0; t < _S; ++t) {

      cpp11::check_user_interrupt();

      _diag1ktu = diagmat(1 / _KTU.col(t));

      Jlogb_as += _w(t) * (
        _diag1ktu * _KT * _JUA(_M*t, _M*s, size(_M, _M))
      );

      // if s = 0
      if (s == 0) {
        logKTU += _w(t) * (
          _diag1ktu * _KT * _JUw(_M*t, 0, size(_M, _S))
        );
      }
    }
    _JbA(0, _N*s, size(_N, _M)) = diagmat(b) * Jlogb_as;
  }

  _Jbw = diagmat(b) * logKTU;
}





// init the vars used in log method
void Barycenter::init_log() {
  // CTRACK;
  // _M, _N, _S => init all the hidden vars

  _U = mat(_M, _S, fill::zeros);   // F
  _V = mat(_N, _S, fill::zeros);   // G
  _K = mat(_M, _N, fill::zeros);   // R
  _KV = mat(_M, _S, fill::zeros);  // Rminrow
  _KTU = mat(_N, _S, fill::zeros); // Rmincol

  // weight matrices
  if (_withjac) {
    _Ws = cube(_M, _N, _S, fill::zeros);
    _Xs = cube(_N, _M, _S, fill::zeros);
  }

  if (_withjac) {
    // F,G,b wrt A
    _JUA = mat(_M*_S, _M*_S, fill::zeros);
    _JVA = mat(_N*_S, _M*_S, fill::zeros);
    _JbA = mat(_N, _M*_S, fill::zeros);
    // F,G,b wrt w
    _JUw = mat(_M*_S, _S, fill::zeros);
    _JVw = mat(_N*_S, _S, fill::zeros);
    _Jbw = mat(_N, _S, fill::zeros);
    // logb wrt A,w
    _JlogbA = mat(_N, _M*_S, fill::zeros);
    _Jlogbw = mat(_N, _S, fill::zeros);
  }
}

void Barycenter::compute_log_impl() {
  // CTRACK;

  // TODO: raise an error if there are elements of A being 0
  if (accu(_A.elem(find(_A == 0)))) {
    cpp11::stop("Some elements in A are zero. Currently not supported in Log Barycenter algorithm");
  }

  // reset b
  b.zeros();

  // reset intermediate vars
  _U.zeros();
  _V.zeros();
  _K.zeros();
  _KV.zeros();
  _KTU.zeros();

  if (_withjac) {
    // F,G,b wrt A
    _JUA.zeros();
    _JVA.zeros();
    _JbA.zeros();
    // F,G,b wrt w
    _JUw.zeros();
    _JVw.zeros();
    _Jbw.zeros();
    // logb wrt A,w
    _JlogbA.zeros();
    _Jlogbw.zeros();
    // weight matrices
    _Ws.zeros();
    _Xs.zeros();
  }

  vec logb = vec(_N, fill::zeros);
  mat logA = log(_A);

  update_log_Rminrow(); // mutate _KV (Rminrow)

  while ((_iter < _maxiter) && (_err >= _zerotol)) {
    _timer.tic();
    // check user interrupt if computation is stuck
    cpp11::check_user_interrupt();

    // update F
    // update_log_Rminrow(); // mutate _KV (Rminrow)
    _U = _U + _reg * logA + _KV;

    if (_withjac) {
      update_log_JF();
    }
    // std::cout << "finish F" << std::endl;

    // update logb
    update_log_Rmincol(); // mutate _KTU (Rmincol)
    logb = - (_V + _KTU) * _w / _reg;
    // logb = - _V * _w / _reg - _KTU * _w / _reg;

    if (_withjac) {
      update_log_Jlogb();
    }
    // std::cout << "finish logb" << std::endl;

    // TODO: how to implement the algo correctly, then optimize?

    // update G
    _V = _V + _reg * logb * _onesS.t() + _KTU;

    if (_withjac) {
      update_log_JG();
    }
    // std::cout << "finish G" << std::endl;

    // termination
    _iter++;
    update_log_Rminrow(); // mutate _KV (Rminrow)
    _err = norm(- _KV / _reg - logA, 2);
    _timer.toc();
    // logging
    if ((_verbose != 0) && ((_iter-1) % _verbose) == 0) {
      cpp11::message(
        "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
        _iter, _err,
        _timer.speed_last(), _timer.speed_avg()
      );
    }

  }

  // output var
  b = exp(logb);
  b = b / accu(b); // rescale to 1
  if (_withjac) {
    _JbA = diagmat(b) * _JlogbA;
    _Jbw = diagmat(b) * _Jlogbw;
  }
}

void Barycenter::update_log_Rminrow() {
  // CTRACK;

  vec rminrow = vec(_M, fill::zeros);
  vec sr = vec(_N, fill::zeros);
  double Rmin;
  // update Rminrow (var: _KV)
  for (uword t = 0; t < _S; ++t) {
    _K = _C - _U.col(t) * _onesN.t() - _onesM * _V.col(t).t();
    _KT = _K.t();
    for (uword i = 0; i < _M; ++i) {

      cpp11::check_user_interrupt();

      Rmin = _KT.col(i).min();
      sr = exp(-(_KT.col(i)-Rmin)/_reg);
      rminrow(i) = Rmin - _reg * log(accu(sr));
      if (_withjac) {
        _Ws.slice(t).row(i) = sr.t() / accu(sr);
      }
    }

    // update Rminrow's t-th column
    _KV.col(t) = rminrow;
  }
}

void Barycenter::update_log_Rmincol() {
  // CTRACK;
  vec rmincol = vec(_N, fill::zeros);
  rowvec sr = rowvec(_M, fill::zeros);
  double Rmin;
  // update Rmincol (var: _KTU)
  for (uword t = 0; t < _S; ++t) {
    _K = _C - _U.col(t) * _onesN.t() - _onesM * _V.col(t).t();
    for (uword j = 0; j < _N; ++j) {

      cpp11::check_user_interrupt();

      Rmin = _K.col(j).min();
      sr = exp(-(_K.col(j)-Rmin)/_reg).t();
      rmincol(j) = Rmin - _reg * log(accu(sr));
      if (_withjac) {
        _Xs.slice(t).row(j) = sr / accu(sr);
      }
    }
    // update Rmincol's t-th column
    _KTU.col(t) = rmincol;
  }
}

void Barycenter::update_log_JF() {
  // CTRACK;

  // update JU
  for (uword t = 0; t < _S; ++t) {

    // update JFA
    for (uword s = 0; s < _S; ++s) {

      cpp11::check_user_interrupt();

      if (t == s) {
        _JUA(_M*t, _M*s, size(_M, _M)) =
          _reg * diagmat(1 / _A.col(t)) - _Ws.slice(t) * _JVA(_N*t,_M*s,size(_N,_M));
      } else {
        _JUA(_M*t, _M*s, size(_M, _M)) =
          - _Ws.slice(t) * _JVA(_N*t,_M*s,size(_N,_M));
      }
    }

    // update JFw
    _JUw(_M*t, 0, size(_M,_S)) = - _Ws.slice(t) * _JVw(_N*t, 0, size(_N,_S));
  }
}

void Barycenter::update_log_JG() {
  // CTRACK;

  // update JG
  for (uword t = 0; t < _S; ++t) {

    // update JGA
    for (uword s = 0; s < _S; ++s) {

      cpp11::check_user_interrupt();

      _JVA(_N*t,_M*s,size(_N,_M)) =
        _reg * _JlogbA(0,_M*s,size(_N,_M))
        - _Xs.slice(t) * _JUA(_M*t,_M*s,size(_M,_M));
    }

    // update JGw
    _JVw(_N*t,0,size(_N,_S)) =
      _reg * _Jlogbw - _Xs.slice(t) * _JUw(_M*t,0,size(_M,_S));
  }
}

void Barycenter::update_log_Jlogb() {
  // CTRACK;

  // update JlogbA
  _JlogbA.zeros();
  for (uword s = 0; s < _S; ++s) {
    for (uword t = 0; t < _S; ++t) {

      cpp11::check_user_interrupt();

      _JlogbA(0,_M*s,size(_N,_M)) +=
        _w(t) * _Xs.slice(t) * _JUA(_M*t,_M*s,size(_M,_M)) / _reg;
    }
  }

  // update Jlogbw
  _Jlogbw = - (_V + _KTU) / _reg;
  for (uword t = 0; t < _S; ++t) {

    cpp11::check_user_interrupt();

    _Jlogbw += (_w(t) * _Xs.slice(t) * _JUw(_M*t,0,size(_M,_S))) / _reg;
  }

}



// void Barycenter::update_parallel_JU2() {
//   // alternative impl of JU
//   // CTRACK;
//
//   mat _diag1kv = mat(_M, _M, fill::zeros);
//   mat _diagukv = mat(_M, _M, fill::zeros);
//   mat _diag1ktu = mat(_N, _N, fill::zeros);
//   mat _diagvktu = mat(_N, _N, fill::zeros);
//
//   for (uword t = 0; t < _S; ++t) {
//     _diag1kv = diagmat(1 / _KV.col(t));
//     _diagukv = diagmat(_U.col(t) / _KV.col(t));
//     _diag1ktu = diagmat(1 / _KTU.col(t));
//     _diagvktu = diagmat(_V.col(t) / _KTU.col(t));
//     // update JUA
//     for (uword s = 0; s < _S; ++s) {
//       if (t == s) {
//         _JUA(_M*t, _M*s, size(_M, _M)) =
//           _diag1kv
//           - _diagukv * _K * _diag1ktu * _JbA(0, _M*s, size(_N, _M))
//         + _diagukv * _K * _diagvktu * _KT * _JUA(_M*t, _M*s, size(_M, _M));
//       } else {
//         _JUA(_M*t, _M*s, size(_M, _M)) =
//           _diagukv * _K * _diag1ktu * _JbA(0, _M*s, size(_N, _M))
//         + _diagukv * _K * _diagvktu * _KT * _JUA(_M*t, _M*s, size(_M, _M));
//       }
//     }
//     // update JUw
//     _JUw(_M*t, 0, size(_M, _S)) =
//       _diagukv * _K * _diagvktu * _KT * _JUw(_M*t, 0, size(_M, _S))
//       - _diagukv * _K * _diag1ktu * _Jbw;
//   }
// }

// void Barycenter::update_parallel_JU3() {
//   // CTRACK;
//   mat IS{mat(_S, _S, fill::eye)};
//   _JUA = diagmat(vectorise(1 / _KV)) -
//     diagmat(vectorise(_U / _KV)) * kron(IS, _K) * _JVA;
// }
