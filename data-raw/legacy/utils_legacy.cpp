
// definition file for the utility functions

// #include <vector> // std::vector
// #include <string> // std::string

#include <cpp11.hpp>
#include <cpp11armadillo.hpp>
using namespace arma;


/////////////////////////////////////////////////////////////////////
// to be called in utils.R
/////////////////////////////////////////////////////////////////////

// // Euclidean matrix based on embeddings
// [[cpp11::register]]
// cpp11::doubles_matrix<> euclidean_cpp(const doubles_matrix<>& AR) {
//   // convert AR to A
//   mat A = as_Mat(AR);
//
//   // create output matrix
//   mat euc = mat(A.n_rows, A.n_rows, fill::zeros);
//
//   double c = 0.;
//   for (size_t i = 0; i < A.n_rows; ++i) {
//     for (size_t j = 0; j < A.n_rows; ++j) {
//       if (i < j) {
//         c = sqrt(accu(square(A.row(i) - A.row(j))));
//         euc(i,j) = c;
//         euc(j,i) = c;
//       }
//     }
//   }
//   return as_doubles_matrix(euc);
// }
//
// // TODO: re-implement the `doc2dist` function directly using `cpp11::strings`
// [[cpp11::register]]
// cpp11::doubles_matrix<> doc2dist_cpp(cpp11::list docs, cpp11::strings dict) {
//   // docs: list of character vectors
//   // dict: character vector of the dictionary
//
//   // create output matrix
//   arma::mat docmat(dict.size(), docs.size(), fill::zeros);
//
//   //
//   for (int j = 0; j < docs.size(); ++j) { // loop the documents
//     cpp11::strings docs_j = docs[j];
//     for (int k = 0; k < docs_j.size(); ++k) { // loop the tokens inside doc
//       cpp11::r_string s = docs_j[k];
//       auto it = std::find(dict.begin(), dict.end(), s);
//       int idx = std::distance(dict.begin(), it);
//       idx = (idx == dict.size()) ? dict.size() - 1 : idx;
//       docmat(idx, j) += 1;
//     } // END of loop tokens
//
//     // scale the matrix so that each col sum to 1
//     docmat.col(j) /= accu(docmat.col(j));
//   } // END of loop documents
//
//   return as_doubles_matrix(docmat);
// }


/////////////////////////////////////////////////////////////////////
// helper functions to setup vec/mat faster
/////////////////////////////////////////////////////////////////////

// identity matrix
mat I(const int n) {
  mat In = mat(n, n, fill::eye);
  return In;
}

// ones vector
mat ones(const int n) {
  vec onesn = vec(n, fill::ones);
  return onesn;
}

// zeros vector
mat zeros(const int n) {
  vec zerosn = vec(n, fill::zeros);
  return zerosn;
}

// R function: R = C - f * onesN' - onesN * g', modify R in-place
void Rf(mat &R, mat &C, vec f, vec g) {
  vec onesN = vec(C.n_cols, fill::ones);
  vec onesM = vec(C.n_rows, fill::ones);
  R = C - f * onesN.t() - onesM * g.t();
}


/////////////////////////////////////////////////////////////////////
// softmin (by row and by col) and its Jacobians
/////////////////////////////////////////////////////////////////////


// Minrow function for sinkhorn without Jacobian: softmin for each row
void minrow(vec &Rminrow, mat &R, const double reg) {
  // modify Rminrow in-place
  double Rmin;
  for (size_t i = 0; i < R.n_rows; ++i) { // i = 1, ..., M
    Rmin = R.row(i).min();
    Rminrow(i) = Rmin - reg * log(accu(exp(-(R.row(i)-Rmin)/reg)));
  }
}

// Mincol function for sinkhorn without Jacobian: softmin for each col
void mincol(vec &Rmincol, mat &R, const double reg) {
  // modify Rmincol in-place
  double Rmin;
  for (size_t j = 0; j < R.n_cols; ++j) { // j = 1, ..., N
    Rmin = R.col(j).min();
    Rmincol(j) = Rmin - reg * log(accu(exp(-(R.col(j)-Rmin)/reg)));
  }
}

// Minrow function for sinkhorn WITH Jacobian: softmin for each row
void minrowjac(mat &W, vec &Rminrow, mat &R, const double reg) {
  // modify W and Rminrow in-place
  double Rmin;
  rowvec sr = rowvec(R.n_cols, fill::zeros);
  for (size_t i = 0; i < R.n_rows; ++i) { // i = 1, ..., M
    Rmin = R.row(i).min();
    sr = exp(-(R.row(i)-Rmin)/reg);
    Rminrow(i) = Rmin - reg * log(accu(sr));
    W.row(i) = sr / accu(sr);
  }
}

// Mincol function for sinkhorn WITH Jacobian: softmin for each col
void mincoljac(mat &V, vec &Rmincol, mat &R, const double reg) {
  // modify Rmincol in-place
  double Rmin;
  vec sr = vec(R.n_rows, fill::zeros);
  for (size_t j = 0; j < R.n_cols; ++j) { // j = 1, ..., N
    Rmin = R.col(j).min();
    sr = exp(-(R.col(j)-Rmin)/reg);
    Rmincol(j) = Rmin - reg * log(accu(sr));
    V.row(j) = sr.t() / accu(sr);
  }
}


// Minrow function for barycenter
void minrow(mat &Rminrow, mat &C, mat &F, mat &G, const double reg) {
  int M = C.n_rows;
  int N = C.n_cols;
  vec rminrow = vec(M, fill::zeros);
  mat R = mat(M, N, fill::zeros);

  for (size_t s = 0; s < Rminrow.n_cols; ++s) {
    R = C - F.col(s) * ones(N).t() - ones(M) *  G.col(s).t();
    minrow(rminrow, R, reg);
    Rminrow.col(s) = rminrow;
  }
}

// Mincol function for barycenter
void mincol(mat &Rmincol, mat &C, mat &F, mat &G, const double reg) {
  int M = C.n_rows;
  int N = C.n_cols;
  vec rmincol = vec(N, fill::zeros);
  mat R = mat(M, N, fill::zeros);

  for (size_t s = 0; s < Rmincol.n_cols; ++s) {
    R = C - F.col(s) * ones(N).t() - ones(M) *  G.col(s).t();
    mincol(rmincol, R, reg);
    Rmincol.col(s) = rmincol;
  }
}


// Minrow function for barycenter WITH Jacobian
void minrowjac(sp_mat &W, mat &Rminrow, mat &C, mat &F, mat &G, const double reg) {
  int M = C.n_rows;
  int N = C.n_cols;
  vec rminrow = vec(M, fill::zeros);
  mat R = mat(M, N, fill::zeros);
  mat W_s = mat(M, N, fill::zeros);

  for (size_t s = 0; s < Rminrow.n_cols; ++s) {
    R = C - F.col(s) * ones(N).t() - ones(M) *  G.col(s).t();
    minrowjac(W_s, rminrow, R, reg);
    Rminrow.col(s) = rminrow;
    // update the block-diagonal W
    // submatrix format: X( first_row, first_col, size(n_rows, n_cols) )
    W(M*s, N*s, size(M, N)) = W_s;
  }
}

// Mincol function for barycenter WITH Jacobian
void mincoljac(sp_mat &V, mat &Rmincol, mat &C, mat &F, mat &G, const double reg) {
  int M = C.n_rows;
  int N = C.n_cols;
  vec rmincol = vec(N, fill::zeros);
  mat R = mat(M, N, fill::zeros);
  mat V_s = mat(N, M, fill::zeros);

  for (size_t s = 0; s < Rmincol.n_cols; ++s) {
    R = C - F.col(s) * ones(N).t() - ones(M) *  G.col(s).t();
    mincoljac(V_s, rmincol, R, reg);
    Rmincol.col(s) = rmincol;
    // update the block-diagonal V
    V(N*s, M*s, size(N,M)) = V_s;
  }
}


/////////////////////////////////////////////////////////////////////
// Optimizers
/////////////////////////////////////////////////////////////////////

// // optimizer: SGD
// void optimizer_sgd(mat &theta, mat &gtheta,
//                    mat &mtheta, mat &vtheta,
//                    int &t,
//                    const double eta = .001, const double gamma = .01,
//                    double beta1 = .9, double beta2 = .999,
//                    double eps = 1e-8) {
//   theta = theta - eta * gtheta;
//   t++;
// }
//
// // optimizer: Adam
// void optimizer_adam(mat &theta, mat &gtheta,
//                     mat &mtheta, mat &vtheta,
//                     int &t,
//                     const double eta = .001, const double gamma = .01,
//                     double beta1 = .9, double beta2 = .999,
//                     double eps = 1e-8) {
//   mtheta = beta1 * mtheta + (1 - beta1) * gtheta;
//   vtheta = beta2 * vtheta + (1 - beta2) * pow(gtheta, 2);
//   mat mhat = mtheta / (1 - pow(beta1, t));
//   mat vhat = vtheta / (1 - pow(beta2, t));
//   theta = theta - eta * (mhat / (sqrt(vhat) + eps));
//   t++;
// }
//
//
// // optimizer: AdamW
// void optimizer_adamw(mat &theta, mat &gtheta,
//                      mat &mtheta, mat &vtheta,
//                      int &t,
//                      const double eta = .001, const double gamma = .01,
//                      double beta1 = .9, double beta2 = .999,
//                      double eps = 1e-8) {
//   // gtheta = gtheta + lambda * theta;
//   mtheta = beta1 * mtheta + (1 - beta1) * gtheta;
//   vtheta = beta2 * vtheta + (1 - beta2) * pow(gtheta, 2);
//   mat mhat = mtheta / (1 - pow(beta1, t));
//   mat vhat = vtheta / (1 - pow(beta2, t));
//   // theta = theta - eta * (mhat / (sqrt(vhat) + eps)) - eta * gamma * theta;
//   theta = (1 - eta*gamma) * theta - eta * (mhat / (sqrt(vhat) + eps));
//   t++;
// }

/////////////////////////////////////////////////////////////////////
// Softmax and its Jacobian
/////////////////////////////////////////////////////////////////////

// softmax on mat: softmax on each column of mat
void softmax(mat& softmaxA, mat& A) {
  vec expAj = vec(A.n_rows, fill::zeros);
  for (size_t j = 0; j < A.n_cols; ++j) {
    expAj = exp(A.col(j) - A.col(j).max());
    softmaxA.col(j) = expAj / accu(expAj);
  }
}

// softmax on mat with its Jacobian
void softmaxjac(mat& JsoftmaxA, mat& softmaxA, mat& A) {
  vec expAj = vec(A.n_rows, fill::zeros);
  int m = A.n_rows;
  int n = A.n_cols;
  vec sxj = vec(m, fill::zeros);
  for (size_t j = 0; j < A.n_cols; ++j) {
    expAj = exp(A.col(j) - A.col(j).max());
    sxj = expAj / accu(expAj);
    // update the softmax column
    softmaxA.col(j) = expAj / accu(expAj);
    // update the jacobian
    JsoftmaxA.submat(j*m, j*m, (j+1)*m-1, (j+1)*m-1) = diagmat(sxj) -
      sxj * sxj.t();
  }
}

// softmax but only calculate the jacobian
void softmax_jac_only(mat &JsoftmaxA, mat &A) {
  int m = A.n_rows;
  int n = A.n_cols;
  vec sxj = vec(m, fill::zeros);
  for (size_t j = 0; j < A.n_cols; ++j) {
    sxj = exp(A.col(j) - A.col(j).max()) / accu(exp(A.col(j) - A.col(j).max()));
    JsoftmaxA.submat(j*m, j*m, (j+1)*m-1, (j+1)*m-1) = diagmat(sxj) -
      sxj * sxj.t();
  }
}
