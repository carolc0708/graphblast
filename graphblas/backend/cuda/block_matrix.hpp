#ifndef GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_HPP_
#define GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_HPP_


#include <vector>
#include <iostream>

namespace graphblas {
namespace backend {

template <typename T>
class BlockMatrix32;

template <typename T>
class BlockMatrix {
 public:
    // default constructor
    BlockMatrix() : nrows_(0), ncols_(0), nvals_(0), block_matrix_32_(0, 0),
                    mat_type_(GrB_SPARSE) {}
    explicit BlockMatrix(Index nrows, Index ncols)
            : nrows_(nrows), ncols_(ncols), nvals_(0), block_matrix_32_(nrows, ncols),
              mat_type_(GrB_SPARSE) {}


    // default destructor
    ~BlockMatrix() {}

    // C API methods

    // mutators
    Info nnew(Index nrows, Index ncols);
    Info dup(const BlockMatrix* rhs);
    Info clear();
    Info nrows(Index* nrows_t);
    Info ncols(Index* ncols_t);
    Info nvals(Index* nvals_t);
    template <typename BinaryOpT>
    Info build(const std::vector<Index>* row_indices,
         const std::vector<Index>* col_indices,
         const std::vector<T>*     values,
         Index                     nvals,
         BinaryOpT                 dup,
         char*                     dat_name);
    Info setElement(Index row_index, Index col_index);
    Info extractElement(T* val, Index row_index, Index col_index);
    Info extractTuples(std::vector<Index>* row_indices,
                 std::vector<Index>* col_indices,
                 std::vector<T>*     values,
                 Index*              n);

    // handy method
    const T operator[](Index ind);
    Info print(bool force_update = false);
    Info check();
    Info setNrows(Index nrows);
    Info setNcols(Index ncols);
    Info resize(Index nrows, Index ncols);
    Info setStorage(Storage mat_type);
    Info getStorage(Storage* mat_type) const;
    Info getFormat(SparseMatrixFormat* format) const;
    Info getSymmetry(bool* symmetry) const;
    template <typename MatrixT>
    MatrixT* getMatrix() const;

    template <typename U>
    Info fill(Index axis, Index nvals, U start);
    template <typename U>
    Info fillAscending(Index axis, Index nvals, U start);

 private:
    Index nrows_;
    Index ncols_;
    Index nvals_;

    BlockMatrix32<T> block_matrix_32_;

    // Keeps track of whether matrix is Sparse or Dense
    Storage mat_type_;

}; // class BlockMatrix

// Transfer nrows ncols to Sparse/DenseMatrix data member
template <typename T>
Info BlockMatrix<T>::nnew(Index nrows, Index ncols) {
  CHECK(block_matrix_32_.nnew(nrows, ncols));
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix<T>::dup(const BlockMatrix* rhs) {
  mat_type_ = rhs->mat_type_;
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.dup(&rhs->block_matrix_32_);

  std::cout << "Error: Failed to call dup!\n";
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info BlockMatrix<T>::clear() {
  mat_type_ = GrB_UNKNOWN;
  nvals_    = 0;
  CHECK(block_matrix_32_.clear());
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::nrows(Index* nrows_t) {
  Index nrows;
  if (mat_type_ == GrB_SPARSE)
    CHECK(block_matrix_32_.nrows(&nrows));
  else
    nrows = nrows_;

  // Update nrows_ with latest value
  nrows_   = nrows;
  *nrows_t = nrows;
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::ncols(Index* ncols_t) {
  Index ncols;
  if (mat_type_ == GrB_SPARSE)
    CHECK(block_matrix_32_.ncols(&ncols));
  else
    ncols = ncols_;

  // Update ncols_ with latest value
  ncols_   = ncols;
  *ncols_t = ncols;
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::nvals(Index* nvals_t) {
  Index nvals;
  if (mat_type_ == GrB_SPARSE)
    CHECK(block_matrix_32_.nvals(&nvals));
  else
    nvals = nvals_;

  // Update nvals_ with latest value
  nvals_   = nvals;
  *nvals_t = nvals;
  return GrB_SUCCESS;
}

// Option: Not const to allow sorting
template <typename T>
template <typename BinaryOpT>
Info BlockMatrix<T>::build(const std::vector<Index>* row_indices,
                      const std::vector<Index>* col_indices,
                      const std::vector<T>*     values,
                      Index                     nvals,
                      BinaryOpT                 dup,
                      char*                     dat_name) {
  mat_type_ = GrB_SPARSE;
//  if (block_matrix_32_.nvals_ > 0)
//    block_matrix_32_.clear();
  return block_matrix_32_.build(row_indices, col_indices, values, nvals, dup, dat_name);
}

template <typename T>
Info BlockMatrix<T>::setElement(Index row_index, Index col_index) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.setElement(row_index, col_index);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info BlockMatrix<T>::extractElement(T* val, Index row_index, Index col_index) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.extractElement(val, row_index, col_index);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info BlockMatrix<T>::extractTuples(std::vector<Index>* row_indices,
                              std::vector<Index>* col_indices,
                              std::vector<T>*     values,
                              Index*              n) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.extractTuples(row_indices, col_indices, values, n);
  else
    return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
const T BlockMatrix<T>::operator[](Index ind) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_[ind];
  else
    std::cout << "Error: operator[] not defined for dense matrices!\n";
  return 0.;
}

template <typename T>
Info BlockMatrix<T>::print(bool force_update) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.print(force_update);
  return GrB_UNINITIALIZED_OBJECT;
}

// Error checking function
template <typename T>
Info BlockMatrix<T>::check() {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.check();
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
Info BlockMatrix<T>::setNrows(Index nrows) {
  CHECK(block_matrix_32_.setNrows(nrows));
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix<T>::setNcols(Index ncols) {
  CHECK(block_matrix_32_.setNcols(ncols));
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix<T>::resize(Index nrows, Index ncols) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.resize(nrows, ncols);
  return GrB_UNINITIALIZED_OBJECT;
}

// Private method that sets mat_type, clears and allocates
template <typename T>
Info BlockMatrix<T>::setStorage(Storage mat_type) {
  mat_type_ = mat_type;
  // Note: do not clear before calling SparseMatrix::allocate!
  if (mat_type_ == GrB_SPARSE) {
    CHECK(block_matrix_32_.allocate());
  }
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::getStorage(Storage* mat_type) const {
  *mat_type = mat_type_;
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::getFormat(SparseMatrixFormat* format) const {
  if (mat_type_ == GrB_SPARSE) return block_matrix_32_.getFormat(format);
  else
    std::cout << "Error: Sparse matrix format is not defined for dense matrix!\n";
  return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix<T>::getSymmetry(bool* symmetry) const {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.getSymmetry(symmetry);
  else
    std::cout << "Error: Matrix symmetry is not defined for dense matrix!\n";
  return GrB_SUCCESS;
}

template <typename T>
template <typename MatrixT>
MatrixT* BlockMatrix<T>::getMatrix() const {
  if (mat_type_ == GrB_SPARSE)
    return &block_matrix_32_;

  return NULL;
}

template <typename T>
template <typename U>
Info BlockMatrix<T>::fill(Index axis, Index nvals, U start) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.fill(axis, nvals, start);
  return GrB_UNINITIALIZED_OBJECT;
}

template <typename T>
template <typename U>
Info BlockMatrix<T>::fillAscending(Index axis, Index nvals, U start) {
  if (mat_type_ == GrB_SPARSE)
    return block_matrix_32_.fillAscending(axis, nvals, start);
  return GrB_UNINITIALIZED_OBJECT;
}


} // namespace backend
} // namespace graphblas

#endif // GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_HPP_