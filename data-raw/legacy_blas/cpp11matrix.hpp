
#ifndef CPP11_MATRIX_H
#define CPP11_MATRIX_H

#include <memory>

// use cpp17
#include <variant>                 // std::variant

#include "R_ext/Error.h"           // for Rf_error, Rf_warning

// #include <cpp11/r_vector.hpp>      // r_vector
#include <cpp11/doubles.hpp>       // doubles


using namespace cpp11;

// extend the `doubles` class?
namespace cpp11 {

// Forward declarations
class Matrix;
class SpecialMatrix;
class TransposedMatrix;
class SymmetricMatrix;

// all the special matrices
class SpecialMatrix {
protected:
  const Matrix* matrix_ptr_;
  explicit SpecialMatrix(const Matrix* mat) : matrix_ptr_(mat) {}
public:
  const Matrix& matrix() const { return *matrix_ptr_; }
  // dims
  virtual int nrow() const;
  virtual int ncol() const;
  // Element access for transposed matrix A^T(i,j) = A(j,i)
  virtual double operator()(int i, int j) const;
  virtual ~SpecialMatrix() = default;
  virtual doubles as_doubles() const;
};

// TODO: need to actually transpose the matrix before returning
// Transposed Matrix
class TransposedMatrix : public SpecialMatrix {
public:
  explicit TransposedMatrix(const Matrix* mat) : SpecialMatrix(mat) {}
  int nrow() const override;
  int ncol() const override;
  double operator()(int i, int j) const override;
  doubles as_doubles() const override;
  ~TransposedMatrix() {}
};

// Symmetric Matrix
class SymmetricMatrix : public SpecialMatrix {
public:
  // Constructor from Matrix?
  explicit SymmetricMatrix(const Matrix* mat) : SpecialMatrix(mat) {}
  // double operator()(int i, int j) const override;
  ~SymmetricMatrix() {}
};


// generic Marix class from the double vector
class Matrix : public r_vector<double> {

private:
  // check if matrix is symmetric: if so, convert
  bool if_symmetric() const;
  void setup_dim(int rows, int cols) {
    // (*this).attr(R_DimSymbol) = r_vector<int>({rows, cols});
    using namespace cpp11::literals;
    SEXP dims = cpp11::safe[Rf_allocVector](INTSXP, 2);
    INTEGER(dims)[0] = rows;
    INTEGER(dims)[1] = cols;
    Rf_setAttrib(static_cast<SEXP>(*this), R_DimSymbol, dims);
  }

public:

  bool is_symmetric = false;
  using TransposeResult = std::variant<SymmetricMatrix,TransposedMatrix>;

  // Constructor with dimensions
  Matrix(int rows, int cols)
    : r_vector<double>(writable::doubles(rows * cols)) {
      // this is creating new vector: need to set up the dims
      this->setup_dim(rows, cols);
      this->check_symmetric();
    }
  // Constructor from existing doubles vector
  Matrix(const doubles& x) : r_vector<double>(x) {
    this->check_symmetric();
  }
  // Constructor from existing doubles vector
  Matrix(const doubles& x, int rows, int cols) : r_vector<double>(x) {
    this->setup_dim(rows, cols);
  }

  ~Matrix() {}

  // get dims
  int nrow() const { return Rf_nrows((*this)); }
  int ncol() const { return Rf_ncols((*this)); }

  // Element access (row-major or column-major, your choice)
  double operator()(int i, int j) const;

  // check if the current matrix is symmetric
  void check_symmetric() { this->is_symmetric = if_symmetric(); }

  // when transpose, return either Transposed or Symmetric
  // Template version for explicit type selection
  TransposedMatrix t() const { return TransposedMatrix(this); }

  // return the doubles: with the dimensions set
  doubles as_doubles() const { return this->data(); }

};

class Vector : public Matrix {
public:
  // constructor
  Vector(int rows) : Matrix(rows, 1) {}
  Vector(const doubles& x): Matrix(x) {
    if (Rf_ncols(x) != 1) {
      Rf_error("Vector constructor: input must have exactly 1 column.");
    }
  } // make sure only one column

};




// implement the class methods for Matrix
inline bool Matrix::if_symmetric() const {
  if (this->nrow() != this->ncol()) { return false; }
  // loop the entire matrix to see if A_ij == A_ji
  for (int i = 0; i < this->nrow(); ++i) {
    for (int j = 0; j < this->ncol(); ++j) {
      if ((i != j) && ((*this)(i,j) != (*this)(j,i))) { return false;}
    }
  }
  return true;
}
inline double Matrix::operator()(int i, int j) const {
  return operator[](i + j * this->nrow()); // Column-major like R
}




// implement the class methods for TransposedMatrix
inline int SpecialMatrix::nrow() const { return matrix_ptr_->nrow(); }
inline int SpecialMatrix::ncol() const { return matrix_ptr_->ncol(); }
inline double SpecialMatrix::operator()(int i, int j) const {
  return (*matrix_ptr_)(i,j);
}
inline doubles SpecialMatrix::as_doubles() const {
  return matrix_ptr_->data();
}

inline int TransposedMatrix::nrow() const { return matrix_ptr_->ncol(); }
inline int TransposedMatrix::ncol() const { return matrix_ptr_->nrow(); }
inline double TransposedMatrix::operator()(int i, int j) const {
  return (*matrix_ptr_)(j,i);
}
inline doubles TransposedMatrix::as_doubles() const {
  // create a new vector
  writable::doubles r(this->ncol() * this->nrow());
  // loop to obtain the transposed matrix
  for (int j = 0; j < this->ncol(); ++j) {
    for (int i = 0; i < this->nrow(); ++i) {
      r[j + i * this->nrow()] = (*this)(j,i);
    }
  }
  return Matrix(r, this->nrow(), this->ncol()).data();
}


// wrapper to convert the `doubles` into Matrix
// inline Matrix as_matrix(const doubles& x) {
//   Matrix xmat{Matrix(x)};
//   return xmat;
// }




// check dimensions for two different matrix classes
template<class T, class S>
inline void dim_error(const T& a, const S& b) {
  Rf_error(
    "Dimension not match: a is %dx%d, b is %dx%d",
    a.nrow(), a.ncol(), b.nrow(), b.ncol()
  );
}
template<class T, class S>
inline void check_elem_dim(const T& a, const S& b) {
  if ((a.nrow() != b.nrow()) || (a.ncol() != b.ncol())) { dim_error(a,b); }
}
template<class T, class S>
inline void check_matmul_dim(const T& a, const S& b) {
  if (a.ncol() != b.nrow()) { dim_error(a,b); }
}


// chech dimensions for the same matrix class
template<class T>
inline void dim_error(const T& a, const T& b) {
  Rf_error(
    "Dimension not match: a is %dx%d, b is %dx%d",
    a.nrow(), a.ncol(), b.nrow(), b.ncol()
  );
}
template<class T>
inline void check_elem_dim(const T& a, const T& b) {
  if ((a.nrow() != b.nrow()) || (a.ncol() != b.ncol())) { dim_error(a,b); }
}
template<class T>
inline void check_matmul_dim(const T& a, const T& b) {
  if (a.ncol() != b.nrow()) { dim_error(a,b); }
}


}

#endif // CPP11_MATRIX_H
