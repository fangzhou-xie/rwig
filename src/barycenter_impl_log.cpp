
// actual implementation of the Barycenter class


#include <vector> // std::vector
#include <thread> // std::thread
// #include <iostream> // std::cout

#include "common.hpp"

#include "barycenter_impl.hpp"
#include "vformat.hpp"

// #include "ctrack.hpp"

using namespace arma;

/////////////////////////////////////////////////////////////////////////
// Algo 6.2/5.2: Log Barycenter with/without Gradients wrt A and w
/////////////////////////////////////////////////////////////////////////

// public method to call for log barycenter
void Barycenter::compute_log(const int& n_threads) {
  // CTRACK;

  // reset the counter
  _reset_counter();

  // set the ones variables
  _onesM = vec(_M, fill::ones);
  _onesN = vec(_N, fill::ones);
  _onesS = vec(_S, fill::ones);

  // forward loop for the barycenter computation
  _fwd_log(n_threads);

  // backward loop for the gradients
  if (_withgrad) { _bwd_log(n_threads); }

  // determine return code
  if (this->err <= _zerotol) {
    this->return_code = 0;
  } else if (this->iter == _maxiter) {
    this->return_code = 1;
  } else {
    this->return_code = 2;
  }
}

/////////////////////////////////////////////////////////////////////////
// private methods for Barycenter log (serial)
/////////////////////////////////////////////////////////////////////////

// other utils for log algos
void Barycenter::_minrow_serial() {
  // CTRACK;

  vec Rmin{vec(_M, fill::zeros)};
  mat expR{mat(_M, _N, fill::zeros)};

  for (uword s = 0; s < _S; ++s) {
    // first update R matrix (by _K), with vectorization
    _K = _C;
    _K.each_col() -= this->U.col(s);
    _K.each_row() -= this->V.col(s).t();
    // calculate minimum of R by rows
    Rmin = arma::min(_K, 1);
    expR = arma::exp(-(_K.each_col() - Rmin) / _reg);
    _KV.col(s) = Rmin - _reg * log(arma::sum(expR, 1));
  }
  // output: _KV (R_min^row)
}

void Barycenter::_mincol_serial() {
  // CTRACK;

  rowvec Rmin{rowvec(_N, fill::zeros)};
  mat expR{mat(_M, _N, fill::zeros)};

  for (uword s = 0; s < _S; ++s) {
    // first update R matrix (by _K), with vectorization
    _K = _C;
    _K.each_col() -= this->U.col(s);
    _K.each_row() -= this->V.col(s).t();
    // calculate minimum of R by cols
    Rmin = arma::min(_K, 0);
    expR = arma::exp(-(_K.each_row() - Rmin) / _reg);
    _KTU.col(s) = (Rmin - _reg * log(arma::sum(expR, 0))).t();
  }
  // output: _KTU (R_min^col)
}

/////////////////////////////////////////////////////////////////////////
// private methods for Barycenter log (thread)
/////////////////////////////////////////////////////////////////////////

void Barycenter::_minrow_thread(const int& n_threads) {
  // CTRACK;

  std::vector<std::thread> threads;
  std::vector<double> res_row(_M); // _M
  int chunk_size = _M / n_threads;

  // loop through all the topics
  for (uword s = 0; s < _S; ++s) {
    _K = _C;
    _K.each_col() -= this->U.col(s);
    _K.each_row() -= this->V.col(s).t();

    // set up the workers
    for (int t = 0; t < n_threads; ++t) {
      int start = t * chunk_size;
      int end = (t == (n_threads - 1)) ? _M : (t + 1) * chunk_size;

      threads.emplace_back(
        [
          &res_row, &R = _K, &reg = this->_reg, start, end
        ]() {
          // for each i in the allocated index inside thread t
          for (int i = start; i < end; ++i) {
            double Rimin = R.row(i).min();
            res_row[i] = Rimin - reg * log(accu(
              exp(-(R.row(i).t() - Rimin) / reg)
            ));
          }
        }
      );
    }

    // join the threads
    for (auto& t : threads) { t.join(); }

    // write column s into _KV
    _KV.col(s) = conv_to<arma::colvec>::from(res_row);

    // clear the thread pool
    threads.clear();
  }
  // output: _KV (R_min^row): _M * _S
}

void Barycenter::_mincol_thread(const int& n_threads) {
  // CTRACK;

  std::vector<std::thread> threads;
  std::vector<double> res_col(_N); // _N
  int chunk_size = _N / n_threads;

  // loop through all the topics
  for (uword s = 0; s < _S; ++s) {
    _K = _C;
    _K.each_col() -= this->U.col(s);
    _K.each_row() -= this->V.col(s).t();

    // set up the workers
    for (int t = 0; t < n_threads; ++t) {
      int start = t * chunk_size;
      int end = (t == (n_threads - 1)) ? _N : (t + 1) * chunk_size;

      threads.emplace_back(
        [
          &res_col, &R = _K, &reg = this->_reg, start, end
        ]() {
          // for each i in the allocated index inside thread t
          for (int j = start; j < end; ++j) {
            double Rjmin = R.row(j).min();
            res_col[j] = Rjmin - reg * log(accu(
              exp(-(R.col(j).t() - Rjmin) / reg)
            ));
          }
        }
      );
    }

    // join the threads
    for (auto& t : threads) { t.join(); }

    // write column s into _KTU
    _KTU.col(s) = conv_to<arma::colvec>::from(res_col);

    // clear the thread pool
    threads.clear();
  }
  // output: _KV (R_min^row): _M * _S
}


// forward pass for the log barycenter
void Barycenter::_fwd_log(const int& n_threads) {
  // CTRACK;

  // set logb
  _logb = vec(_N, fill::zeros);
  // reset intermediate vars
  this->U = mat(_M, _S, fill::zeros); // F
  this->V = mat(_N, _S, fill::zeros); // G
  if (_withgrad) {
    // reserve space
    _Uhist.clear();
    _Vhist.clear();
    _logbhist.clear();
    _Uhist.reserve(_maxiter+1);
    _Vhist.reserve(_maxiter+1);
    _logbhist.reserve(_maxiter+1);

    _Uhist.push_back(this->U);  // F hist
    _Vhist.push_back(this->V);  // G hist
    _logbhist.push_back(_logb); // logb hist
  }
  const mat logA = log(_A);

  // setup _KV and _KTU
  _KV = mat(_M, _S, fill::zeros);
  _KTU = mat(_N, _S, fill::zeros);

  // logging for forward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Forward pass:"));
  }
  // _minrow(); // update _KV
  n_threads ? _minrow_thread(n_threads) : _minrow_serial();

  while ((this->iter < _maxiter) && (this->err >= _zerotol)) {
    // cpp11::check_user_interrupt();
    Rcpp::checkUserInterrupt();
    this->iter++;
    if (_verbose != 0) { _timer.tic(); }

    // update F
    // _minrow(); // update _KV
    this->U += _reg * logA + _KV;
    if (_withgrad) { _Uhist.push_back(this->U); }

    // std::cout << "F:" << std::endl << this->U << std::endl;

    // update logb
    // _mincol(); // update _KTU
    n_threads ? _mincol_thread(n_threads) : _mincol_serial();
    _logb = - (this->V + _KTU) * _w / _reg;
    if (_withgrad) { _logbhist.push_back(_logb); }

    // std::cout << "logb:" << std::endl << this->_logb << std::endl;

    // update G
    this->V += _reg * this->_logb * _onesS.t() + _KTU;
    if (_withgrad) { _Vhist.push_back(this->V); }

    // std::cout << "G:" << std::endl << this->V << std::endl;

    // _minrow(); // update _KV
    n_threads ? _minrow_thread(n_threads) : _minrow_serial();
    this->err = norm(- _KV / _reg - logA, 2);
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
  // recover the b from logb and rescale
  this->b = exp(_logb);
  this->b = this->b / accu(this->b);
  // compute the loss
  if (_withgrad) { this->loss = accu(pow(this->b - this->_b_ext, 2)); }
}

// backward pass for the log barycenter
void Barycenter::_bwd_log(const int& n_threads) {
  // CTRACK;

  // adjoints/gradients of the vars for output
  this->grad_A = mat(_M, _S, fill::zeros);
  this->grad_w = vec(_S, fill::zeros);

  // adjoints of the intermediate vars
  mat Fbar{mat(_M, _S, fill::zeros)};
  mat Gbar{mat(_N, _S, fill::zeros)};
  vec logbbar{vec(_N, fill::zeros)};

  // aux vars (weighting matrices)
  mat W{mat(_M, _N, fill::zeros)};
  mat X{mat(_N, _M, fill::zeros)};
  // vec expRi{vec(_N, fill::zeros)};
  // vec expRj{vec(_M, fill::zeros)};

  vec Rmin; // Rminrow or Rmincol
  // mat expR; // exp(-(R - Rmin)/varepsilon), vectorized

  // logging for backward pass
  if (_verbose != 0) {
    Rcpp::message(Rf_mkString("Backward pass:"));
  }

  for (int l = this->iter; l > 0; --l) {

    if (_verbose) { _timer.tic(); }

    // first update adjoint of G only when l < this->iter
    if (l != this->iter) {
      // for each column in S
      for (uword s = 0; s < _S; ++s) {
        // update R first
        _K = _C;
        _K.each_col() -= _Uhist[l].col(s);
        _K.each_row() -= _Vhist[l].col(s).t();
        Rmin = arma::min(_K, 1);
        // expR = arma::exp(-(_K.each_col() - Rmin) / _reg);
        // X = (expR.each_col() / arma::sum(expR, 1)).t();
        X = arma::exp(-(_K.each_col() - Rmin) / _reg);
        X.each_col() /= arma::sum(X, 1);
        Gbar.col(s) = - X.t() * Fbar.col(s);
      }
    } // done updating Gbar

    // update adjoint of logb
    if (l == this->iter) {
      logbbar = 2 * (this->b - this->_b_ext) % this->b;
    } else {
      logbbar = _reg * sum(Gbar, 1);
    } // done with logbbar

    // update adjoint of F
    for (uword s = 0; s < _S; ++s) {
      // update R first
      _K = _C;
      _K.each_col() -= _Uhist[l].col(s);
      _K.each_row() -= _Vhist[l-1].col(s).t();
      Rmin = arma::min(_K, 0).t();
      // expR = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);
      // W = expR.each_row() / arma::sum(expR, 0);
      W = arma::exp(-(_K.each_row() - Rmin.t()) / _reg);
      W.each_row() /= arma::sum(W, 0);
      if (l == this->iter) {
        Fbar.col(s) = (_w(s) / _reg) * (W * logbbar);
      } else {
        // Fbar.col(s) = (_w(s) / _reg) * (W * logbbar) - W * Gbar.col(s);
        Fbar.col(s) = W * ((_w(s) / _reg) * logbbar - Gbar.col(s));
      }
    } // done updating Fbar

    if (_verbose) { _timer.toc(); }
    // logging
    if ((_verbose != 0) && ((this->iter-1) % _verbose) == 0) {

      // first format the msg as c-string
      // convert c-string into SEXP and then print via Rcpp::message
      Rcpp::message(Rf_mkString(vformat(
          "iter: %d, last speed: %.3f, avg speed: %.3f",
          l, _timer.speed_last(), _timer.speed_avg()
      ).c_str()));
    }

    // Rcpp::Rcout << "Fbar: \n" << Fbar << std::endl;
    // Rcpp::Rcout << "Gbar: \n" << Gbar << std::endl;

    // accumulate adjoints of A (without reg scaling)
    this->grad_A += Fbar / _A;

    // accumulate adjoints of w (without reg scaling)
    // first recover F^{\ell} and G^{\ell-1}
    this->U = _Uhist[l];   // F
    this->V = _Vhist[l-1]; // G
    // _mincol(); // Min_\varepsilon^{col} (F^\ell, G^{\ell-1}): _KTU
    n_threads ? _mincol_thread(n_threads) : _mincol_serial(); // _KTU
    this->grad_w -= (_Vhist[l-1] + _KTU).t() *  logbbar;
  }

  // remember to scale it per the formula
  this->grad_A *= _reg;
  this->grad_w /= _reg;

  // revert the F^{L} and G^{L}
  this->U = _Uhist[this->iter];
  this->V = _Vhist[this->iter];
}


// void Barycenter::_minrow() {
//   vec rminrow{vec(_M, fill::zeros)};
//   vec sr{vec(_N, fill::zeros)};
//   double Rmin;
//   // update Rminrow (var: _KV)
//   for (uword s =  0; s < _S; ++s) {
//     _K = _C - this->U.col(s) * _onesN.t() - _onesM * this->V.col(s);
//     for (uword i = 0; i < _N; ++i) {
//       Rmin = _K.row(i).min();
//       sr = exp(-(_K.row(i) - Rmin) / _reg).t();
//       rminrow(i) = Rmin - _reg * log(accu(sr));
//     }
//     _KV.col(s) = rminrow;
//   }
// }

// void Barycenter::_mincol() {
//   CTRACK;
//   vec rmincol{vec(_N, fill::zeros)};
//   vec sr{vec(_M, fill::zeros)};
//   double Rmin;
//   // update Rmincol (var: _KTU)
//   for (uword s = 0; s < _S; ++s) {
//     _K = _C - this->U.col(s) * _onesN.t() - _onesM * this->V.col(s);
//     for (uword j = 0; j < _N; ++j) {
//       Rmin = _K.col(j).min();
//       sr = exp(-(_K.col(j) - Rmin) / _reg);
//       rmincol(j) = Rmin - _reg * log(accu(sr));
//     }
//     _KTU.col(s) = rmincol;
//   }
// }
