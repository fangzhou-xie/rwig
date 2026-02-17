
// actual implementation of the Sinkhorn log (serial version)

#include <thread> // std::thread
#include <vector> // std::vector
// #include <iostream> // std::cout

#include "common.hpp"

#include "sinkhorn_impl.hpp"
#include "vformat.hpp"

// #include "ctrack.hpp"
// #include "timer.hpp"

using namespace arma;

////////////////////////////////////////////////////////////////
// utilities: minrow, mincol (serial)
////////////////////////////////////////////////////////////////

// faster version of R function (vectorized)
void Sinkhorn::_compute_R() {
  // CTRACK;
  _K = _C;
  _K.each_col() -= _u;
  _K.each_row() -= _v.t();
}

// minrow of R
void Sinkhorn::_minrow_serial() {
  // CTRACK;

  _compute_R(); // update R (in _K)
  vec Rmin = arma::min(_K, 1);
  mat expRi = exp(-(_K.each_col() - Rmin) / _reg);
  _Rminrow =  Rmin - _reg * log(arma::sum(expRi, 1));
  // if (_withgrad) {
  //   _Xhist.push_back((expRi.each_col() / arma::sum(expRi, 1)).t());
  // }
}

// mincol of R: also update W(f{^\ell}, g^{\ell-1})
void Sinkhorn::_mincol_serial() {
  // CTRACK;

  _compute_R(); // update R (in _K)
  rowvec Rmin = arma::min(_K, 0);
  mat expRj = arma::exp(-(_K.each_row() - Rmin) / _reg);
  _Rmincol = Rmin.t() - _reg * log(arma::sum(expRj, 0).t());
  // if (_withgrad) {
  //   _Whist.push_back(expRj.each_row() / arma::sum(expRj, 0));
  // }
}


// mincol and minrow for the err calculation
void Sinkhorn::_minrowcol_serial() {
  // CTRACK;

  vec Rmin;
  mat expR;

  _compute_R();

  Rmin = arma::min(_K, 1);
  expR = exp(-(_K.each_col() - Rmin) / _reg);

  _Rminrow =  Rmin - _reg * log(arma::sum(expR, 1));

  Rmin = arma::min(_K, 0).t();
  expR = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);

  _Rmincol = Rmin - _reg * log(arma::sum(expR, 0).t());
}

////////////////////////////////////////////////////////////////
// utilities: minrow, mincol (thread)
////////////////////////////////////////////////////////////////

// minrow of R (threading version)
void Sinkhorn::_minrow_thread(const int& n_threads) {
  // CTRACK:

  _compute_R(); // update R in (_K)

  std::vector<std::thread> threads;
  std::vector<double> res(_M); // number of rows
  int chunk_size = _M / n_threads;

  // set up the workers
  for (int t = 0; t < n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_threads - 1)) ? _M : (t + 1) * chunk_size;

    threads.emplace_back(
      [
       &res, &R = this->_K, &reg = this->_reg, start, end
    ]() {
      for (int i = start; i < end; ++i) {
        double Rimin = R.row(i).min();
        // vec expRi = exp(-(R.row(i) - Rimin) / reg).t();
        res[i] = Rimin - reg * log(accu(
          exp(-(R.row(i).t() -  Rimin) / reg)
        ));
      }
    });
  }

  // join the threads for results
  for (auto& t : threads) { t.join(); }

  // convert std vec back to arma vec
  _Rminrow = conv_to<arma::colvec>::from(res);
}


// mincol of R (threading version)
void Sinkhorn::_mincol_thread(const int& n_threads) {
  // CTRACK;

  _compute_R(); // update R in (_K)

  std::vector<std::thread> threads;
  std::vector<double> res(_N); // number of cols
  int chunk_size = _N / n_threads;

  // set up the workers
  for (int t = 0; t < n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_threads - 1)) ? _N : (t + 1) * chunk_size;

    threads.emplace_back(
      [
       &res, &R = this->_K, &reg = this->_reg, start, end
    ]() {
      for (int j = start; j < end; ++j) {
        double Rjmin = R.col(j).min();
        // vec expRj = exp(-(R.col(j) - Rjmin) / reg);
        res[j] = Rjmin - reg * log(accu(
          exp(-(R.col(j) - Rjmin) / reg)
        ));
      }
    });
  }

  // join the threads for results
  for (auto& t : threads) { t.join(); }

  // convert std vec to arma vec
  _Rmincol = conv_to<arma::colvec>::from(res);
}


// mincol and minrow for the err calculation (thread)
void Sinkhorn::_minrowcol_thread(const int& n_threads) {
  // CTRACK;

  _compute_R();

  std::vector<std::thread> threads;
  int chunk_size;

  std::vector<double> res_row(_M); // number of rows
  std::vector<double> res_col(_N); // number of cols

  chunk_size = _M / n_threads;

  // set up the workers
  for (int t = 0; t < n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_threads - 1)) ? _M : (t + 1) * chunk_size;

    threads.emplace_back(
      [
       &res_row, &R = this->_K, &reg = this->_reg, start, end
    ]() {
      for (int i = start; i < end; ++i) {
        double Rimin = R.row(i).min();
        res_row[i] = Rimin - reg * log(accu(
          exp(-(R.row(i) - Rimin) / reg)
        ));
      }
    });
  }

  // join the threads for results
  for (auto& t : threads) { t.join(); }

  // destroy the threads and start again
  threads.clear();
  chunk_size = _N / n_threads;

  // set up workers
  for (int t = 0; t < n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_threads - 1)) ? _N : (t + 1) * chunk_size;

    threads.emplace_back([
                           &res_col, &R = this->_K, &reg = this->_reg, start, end
    ]() {
      for (int j = start; j < end; ++j) {
        double Rjmin = R.col(j).min();
        res_col[j] = Rjmin - reg * log(accu(
          exp(-(R.col(j) - Rjmin) / reg)
        ));
      }
    });
  }

  // join the threads for results
  for (auto& t : threads) { t.join(); }

  // convert the vectors again
  _Rminrow = conv_to<arma::colvec>::from(res_row);
  _Rmincol = conv_to<arma::colvec>::from(res_col);
}



////////////////////////////////////////////////////////////////
// Algo 4.2/3.2: Log Sinkhorn with/without Gradient wrt a
////////////////////////////////////////////////////////////////

void Sinkhorn::compute_log(const vec& a, const vec& b,
                           const mat& C, double reg,
                           const int& n_threads) {
  // CTRACK;

  // reset the counter
  _reset_counter();

  int M = C.n_rows;
  int N = C.n_cols;
  // indices where a and b are not 0
  // uvec a_ind{find(a != 0)};
  // uvec b_ind{find(b != 0)};
  uvec a_ind = find(a != 0);
  uvec b_ind = find(b != 0);

  // convert problem to reduced form
  this->_a = a.elem(a_ind);
  this->_b = b.elem(b_ind);
  this->_C = C.submat(a_ind, b_ind);
  this->_C_is_symm = _C.is_symmetric();

  this->_reg = reg;
  this->_M = _C.n_rows;
  this->_N = _C.n_cols;
  this->_onesM = vec(_M, fill::ones);
  this->_onesN = vec(_N, fill::ones);

  // if (_withgrad) {
  //   // this->_uhist = std::vector<vec>(_maxiter); // FIXME: or plus 1?
  //   // this->_vhist = std::vector<vec>(_maxiter);
  //   this->_uhist = mat(_M, _maxiter+1, fill::zeros);  // f
  //   this->_vhist = mat(_N, _maxiter+1, fill::zeros);  // g
  //   // this->_Whist = cube(_M, _N, _maxiter+1, fill::zeros); // W
  //   // this->_Xhist = cube(_N, _M, _maxiter+1, fill::zeros); // X
  // }

  // start the computation
  this->_fwd_log(n_threads);

  // recover original solution
  this->_P = exp(- _K / _reg);
  this->u = vec(M, fill::value(-datum::inf));
  this->v = vec(N, fill::value(-datum::inf));
  this->u.elem(a_ind) = _u;
  this->v.elem(b_ind) = _v;

  // output P
  this->P = mat(M, N, fill::zeros);
  this->P.submat(a_ind, b_ind) = this->_P;

  // record the loss for the reduced problem
  this->loss = accu(_C % _P) + _reg * accu(_P % (log(_P) - 1));

  // ifgrad -> backward
  if (this->_withgrad) {
    this->_bwd_log(n_threads); // update `_grad_a`
    this->grad_a = vec(M, fill::zeros);
    this->grad_a.elem( a_ind ) = this->_grad_a;
  }

  // determine return code
  if (this->err <= _zerotol) {
    this->return_code = 0;
  } else if (this->iter == _maxiter) {
    this->return_code = 1;
  } else {
    this->return_code = 2;
  }
}


// forward of log sinkhorn
void Sinkhorn::_fwd_log(const int& n_threads) {
  // CTRACK;

  _u = vec(_M, fill::zeros); // f
  _v = vec(_N, fill::zeros); // g
  if (_withgrad) {
    // reserver space
    _uhist.clear();
    _vhist.clear();
    _uhist.reserve(_maxiter+1);
    _vhist.reserve(_maxiter+1);

    _uhist.push_back(_u);
    _vhist.push_back(_v);
  }
  const vec loga = log(_a);
  const vec logb = log(_b);
  _Rminrow = vec(_M, fill::zeros);
  _Rmincol = vec(_N, fill::zeros);

  // logging for forward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }

  // update Rminrow
  // update Rminrow
  n_threads ? _minrow_thread(n_threads) : _minrow_serial();

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    // cpp11::check_user_interrupt();
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) { _timer.tic(); }

    // update f
    // _K = _C - _u * _onesN.t() - _onesM * _v.t();
    // n_threads ? _minrow_thread(n_threads) : _minrow_serial();
    _u += _reg * loga + _Rminrow;
    // keep track of history of f
    // if (_withgrad) { _uhist.col(this->iter) = _u; }
    if (_withgrad) { _uhist.push_back(_u); }

    // Rcpp::Rcout << "R: \n" << _K << std::endl;
    // Rcpp::Rcout << "Rminrow: \n" << _Rminrow << std::endl;

    // update g
    // _K = _C - _u * _onesN.t() - _onesM * _v.t();
    // _mincol_serial();
    // update Rmincol
    n_threads ? _mincol_thread(n_threads) : _mincol_serial();
    _v += _reg * logb + _Rmincol;
    // keep track of history of g
    // if (_withgrad) { _vhist.col(this->iter) = _v; }
    if (_withgrad) { _vhist.push_back(_v); }

    // Rcpp::Rcout << "R: \n" << _K << std::endl;
    // Rcpp::Rcout << "Rmincol: \n" << _Rmincol << std::endl;

    // _K = _C - _u * _onesN.t() - _onesM * _v.t();
    // _minrowcol_serial();
    n_threads ? _minrowcol_thread(n_threads) : _minrowcol_serial();
    this->err = norm(-_Rminrow/_reg - loga, 2) + norm(-_Rmincol/_reg - logb, 2);
    if (_verbose != 0) { _timer.toc(); }

    // logging
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, err: %.4f, last speed: %.3f, avg speed: %.3f",
          this->iter, this->err,
          _timer.speed_last(), _timer.speed_avg()
      ).c_str()));

    }
  }
}

// backward (reverse) of vanilla sinkhorn
void Sinkhorn::_bwd_log(const int& n_threads) {
  // CTRACK;

  // gradients (adjoints) for all the intermediate vars
  mat PbarP = (_C + _reg * log(_P)) % _P; // adjoint of P dot P
  vec fbar(_M, fill::zeros);
  vec gbar(_N, fill::zeros);

  mat W(_M, _N, fill::zeros);
  mat X(_N, _M, fill::zeros);
  // mat expR; //
  vec Rmin; //
  // vec expRi{vec(_N, fill::zeros)};
  // vec expRj{vec(_M, fill::zeros)};

  // adjoint for the output gradient in reduced form
  _grad_a = vec(_M, fill::zeros);

  int chunk_size;
  std::vector<std::thread> threads;

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  for (int l = this->iter; l > 0; --l) {
    if (_verbose != 0) { _timer.tic(); }

    ////////////////////////////////
    // without packing the updates into functions
    ////////////////////////////////

    // update adjoint of g
    if (l == this->iter) {                  // \ell = L
      gbar = PbarP.t() * _onesM / _reg;
    } else {                                // \ell = L-1, \ldots, 1

      // update R
      _u = _uhist[l];
      _v = _vhist[l];
      _compute_R();

      if (n_threads) {
        // threading version

        chunk_size = _M / n_threads;

        for (int t = 0; t < n_threads; ++t) {
          int start = t * chunk_size;
          int end = (t == (n_threads - 1)) ? _M : (t + 1) * chunk_size;

          threads.emplace_back(
            [&X, &R = this->_K, &reg = this->_reg, start, end]() {
              for (int i = start; i < end; ++i) {
                vec expRi = exp(-(R.row(i).t() - R.row(i).min()) / reg);
                X.col(i) = expRi / accu(expRi);
              }
            }
          );
        }
        for (auto& t : threads) { t.join(); }
        threads.clear();

      } else {
        // serial version
        Rmin = arma::min(_K, 1);
        // expR = arma::exp(-(_K.each_col() - Rmin) / _reg);
        // X = expR.each_col() / arma::sum(expR, 1);
        // gbar = - X.t() * fbar;
        X = arma::exp(-(_K.each_col() - Rmin) / _reg).t();
        X = X.each_row() / arma::sum(X, 0);
      }

      // update gbar
      gbar = - X * fbar;
    }

    // update adjoint of f
    if (n_threads) {

      chunk_size = _N / n_threads;

      for (int t = 0; t < n_threads; ++t) {
        int start = t * chunk_size;
        int end = (t == (n_threads - 1)) ? _N : (t + 1) * chunk_size;

        threads.emplace_back(
          [&W, &R = this->_K, &reg = this->_reg, start, end]() {
            for (int j = start; j < end; ++j) {
              vec expRj = exp(-(R.col(j) - R.col(j).min()) / reg);
              W.col(j) = expRj / accu(expRj);
            }
          }
        );
      }
      for (auto& t : threads) { t.join(); }
      threads.clear();

    } else {

      Rmin = arma::min(_K, 0).t();
      // expR = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);
      // W = expR.each_row() / arma::sum(expR, 0);
      W = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);
      W = W.each_row() / arma::sum(W, 0);
    }

    fbar = - W * gbar;
    if (l == this->iter) { fbar += PbarP * _onesN / _reg; }

    ////////////////////////////////
    // using the packaged version
    ////////////////////////////////
    // if (n_threads) {
    //   _update_gbar_thread(fbar, gbar, PbarP, l, n_threads);
    //   _update_fbar_thread(fbar, gbar, PbarP, l, n_threads);
    // } else {
    //   _update_gbar_serial(fbar, gbar, PbarP, l);
    //   _update_fbar_serial(fbar, gbar, PbarP, l);
    // }

    if (_verbose != 0) { _timer.toc(); }
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, last speed: %.3f, avg speed: %.3f",
          l, _timer.speed_last(), _timer.speed_avg()
      ).c_str()));
    }

    // accumulate abar (_grad_a)
    _grad_a += fbar / _a;
  }
  _grad_a *= _reg;

  // also revert the f and g before returning
  _u = _uhist[this->iter];
  _v = _vhist[this->iter];
}

void Sinkhorn::_update_fbar_serial(vec& fbar, vec& gbar, mat& PbarP, int& l) {
  // update adjoint of f
  vec Rmin = arma::min(_K, 0).t();
  mat expR = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);
  mat W = expR.each_row() / arma::sum(expR, 0);
  fbar = - W * gbar;
  if (l == this->iter) { fbar += PbarP * _onesN / _reg; }
}

void Sinkhorn::_update_gbar_serial(vec& fbar, vec& gbar, mat& PbarP, int& l) {
  // update adjoint of g
  if (l == this->iter) {                  // \ell = L
    gbar = PbarP.t() * _onesM / _reg;
  } else {                                // \ell = L-1, \ldots, 1
    _u = _uhist[l];
    _v = _vhist[l];
    _compute_R();
    vec Rmin = arma::min(_K, 1);
    mat expR = arma::exp(-(_K.each_col() - Rmin) / _reg);
    mat X = expR.each_col() / arma::sum(expR, 1);
    gbar = - X.t() * fbar;
  }
}

void Sinkhorn::_update_fbar_thread(vec& fbar, vec& gbar, mat& PbarP, int& l,
                                   const int& n_threads) {
  // update adjoint of f (threading)

  std::vector<std::thread> threads;
  mat W(_M, _N, fill::zeros);
  int chunk_size = _N / n_threads;

  // set up workers
  for (int t = 0; t < n_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == (n_threads - 1)) ? _N : (t + 1) * chunk_size;

    threads.emplace_back(
      [
        &W, &R = this->_K, &reg = this->_reg, start, end
      ](){
        for (int j = start; j < end; ++j) {
          vec expRj = exp(-(R.col(j) - R.col(j).min()) / reg);
          W.col(j) = expRj / accu(expRj);
        }
      }
    );
  }

  // join the threads
  for (auto& t: threads) { t.join(); }

  fbar = - W * gbar;
  if (l == this->iter) { fbar += PbarP * _onesN / _reg; }
}

void Sinkhorn::_update_gbar_thread(vec& fbar, vec& gbar, mat& PbarP, int& l,
                                   const int& n_threads) {
  // update adjoint of g (threading)
  if (l == this->iter) {                  // \ell = L
    gbar = PbarP.t() * _onesM / _reg;
  } else {                                // \ell = L-1, \ldots, 1

    std::vector<std::thread> threads;
    mat X(_N, _M, fill::zeros);
    int chunk_size = _M / n_threads;

    // set up workers
    for (int t = 0; t < n_threads; ++t) {
      int start = t * chunk_size;
      int end = (t == (n_threads - 1)) ? _M : (t + 1) * chunk_size;

      threads.emplace_back(
        [
          &X, &R = this->_K, &reg = this->_reg, start, end
        ]() {
          for (int i = start; i < end; ++i) {
            vec expRi = exp(-(R.row(i).t() - R.row(i).min()) / reg);
            X.col(i) = expRi / accu(expRi);
          }
        }
      );
    }

    // join the threads
    for (auto& t: threads) { t.join(); }

    gbar = - X * fbar;
  }
}

// void Sinkhorn::_minrow() {
//   CTRACK;
//   double Rmin;
//   vec expRi{vec(_N, fill::zeros)};
//   for (uword i = 0; i < _M; ++i) { // i = 1, ..., M
//     Rmin = _K.row(i).min();
//     expRi = exp(-(_K.row(i) - Rmin) / _reg).t();
//     _Rminrow(i) = Rmin - _reg * log(accu(expRi));
//     // if (_withgrad) { _Xhist.slice(this->iter).col(i) = expRi / accu(expRi);}
//   }
// }

// void Sinkhorn::_mincol() {
//   CTRACK;
//   double Rmin;
//   vec expRj{vec(_M, fill::zeros)};
//   for (uword j = 0; j < _N; ++j) { // j = 1, ..., N
//     Rmin = _K.col(j).min();
//     expRj = exp(-(_K.col(j) - Rmin) / _reg);
//     _Rmincol(j) = Rmin -  _reg * log(accu(expRj));
//     // if (_withgrad) { _Whist.slice(this->iter).col(j) = expRj / accu(expRj);}
//   }
// }

// slow version!!
// void Sinkhorn::_compute_R1() {
//   CTRACK;
//   _K = _C - _u * _onesN.t() - _onesM * _v.t();
// }
