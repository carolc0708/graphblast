#ifndef GRAPHBLAS_BLOCK_MATRIX_HPP_
#define GRAPHBLAS_BLOCK_MATRIX_HPP_

#include <vector>

// Opaque data members from the right backend
#define __GRB_BACKEND_MATRIX_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/block_matrix.hpp>
#include __GRB_BACKEND_MATRIX_HEADER
#undef __GRB_BACKEND_MATRIX_HEADER

namespace graphblas {

template <typename T>
class BlockMatrix {
 public:
    // default constructor
    BlockMatrix() : block_matrix_() {}
    BlockMatrix(Index nrows, Index ncols) : block_matrix_(nrows, ncols) {}

    // default destructor
    ~BlockMatrix() {}

    // C API methods
    Info nnew(Index nrows, Index ncols);
    Info dup(const BlockMatrix* rhs);
    Info clear();
    Info nrows(Index* nrows_) const;
    Info ncols(Index* ncols_) const;
    Info nvals(Index* nvals_) const;
    template <typename BinaryOpT>
    Info build(const std::vector<Index>* row_indices,
              const std::vector<Index>* col_indices,
              const std::vector<T>*     values,
              Index                     nvals,
              BinaryOpT                 dup,
              char*                     dat_name); // support one build function only

    Info setElement(Index row_index, Index col_index);
    Info extractElement(T*    val,
                        Index row_index,
                        Index col_index);
    Info extractTuples(std::vector<Index>* row_indices,
                       std::vector<Index>* col_indices,
                       std::vector<T>*     values,
                       Index*              n);
    Info extractTuples(std::vector<T>* values, Index* n);

    // handy methods
    void operator=(const BlockMatrix& rhs);
    const T operator[](Index ind);
    Info print(bool force_update = false);
    Info check();
    Info setNrows(Index nrows);
    Info setNcols(Index ncols);
    Info resize(Index nrows, Index ncols);
    Info setStorage(Storage  mat_type);
    Info getStorage(Storage* mat_type) const;

    template <typename U>
    Info fill(Index axis, Index nvals, U start);
    template <typename U>
    Info fillAscending(Index axis, Index nvals, U start);

    private:
    // Data members that are same for all backends
    backend::BlockMatrix<T> block_matrix_;

}; // class BlockMatrix

template <typename T>
Info BlockMatrix<T>::nnew (Index nrows, Index ncols) {
    if (nrows == 0 || ncols == 0) return GrB_INVALID_VALUE;
    return block_matrix_.nnew(nrows, ncols);
}

template <typename T>
Info BlockMatrix<T>::dup(const BlockMatrix* rhs) {
    if (rhs == NULL) return GrB_NULL_POINTER;
    return block_matrix_.dup(&rhs->block_matrix_);
}

template <typename T>
Info BlockMatrix<T>::clear() {
    return block_matrix_.clear();
}

template <typename T>
Info BlockMatrix<T>::nrows(Index* nrows) const {
    if (nrows == NULL) return GrB_NULL_POINTER;
    backend::BlockMatrix<T>* block_matrix_t = const_cast<backend::BlockMatrix<T>*>(&block_matrix_);
    return block_matrix_t->nrows(nrows);
}

template <typename T>
Info BlockMatrix<T>::ncols(Index* ncols) const {
    if (ncols == NULL) return GrB_NULL_POINTER;
    backend::BlockMatrix<T>* block_matrix_t = const_cast<backend::BlockMatrix<T>*>(&block_matrix_);
    return block_matrix_t->ncols(ncols);
}

template <typename T>
Info BlockMatrix<T>::nvals(Index* nvals) const {
    if (nvals == NULL) return GrB_NULL_POINTER;
    backend::BlockMatrix<T>* block_matrix_t = const_cast<backend::BlockMatrix<T>*>(&block_matrix_);
    return block_matrix_t->nvals(nvals);
}

template <typename T>
template <typename BinaryOpT>
Info BlockMatrix<T>::build(const std::vector<Index>* row_indices,
                            const std::vector<Index>* col_indices,
                            const std::vector<T>*     values,
                            Index                     nvals,
                            BinaryOpT                 dup,
                            char*                     dat_name) {
//  if (row_indices == NULL || col_indices == NULL || values == NULL)
//    return GrB_NULL_POINTER;
//
//  if (row_indices->size() == 0 && col_indices->size() == 0 && values->size() == 0 && dat_name == NULL)
//    return GrB_NO_VALUE;
//
//  if (dat_name == NULL || (*row_indices).size() > 0)
    return block_matrix_.build(row_indices, col_indices, values, nvals, dup, dat_name);

//    return block_matrix_.build(dat_name);
}

template <typename T>
Info BlockMatrix<T>::setElement(Index row_index, Index col_index) {
    return block_matrix_.setElement(row_index, col_index);
}

template <typename T>
Info BlockMatrix<T>::extractElement(T*    val,
                                    Index row_index,
                                    Index col_index) {
  if (val == NULL) return GrB_NULL_POINTER;
  return block_matrix_.extractElement(val, row_index, col_index);
}

template <typename T>
Info BlockMatrix<T>::extractTuples(std::vector<Index>* row_indices,
                                   std::vector<Index>* col_indices,
                                   std::vector<T>*     values,
                                   Index*              n) {
  if (row_indices == NULL || col_indices == NULL || values == NULL || n == NULL)
    return GrB_NULL_POINTER;
  return block_matrix_.extractTuples(row_indices, col_indices, values, n);
}

template <typename T>
Info BlockMatrix<T>::extractTuples(std::vector<T>* values,
                              Index*          n) {
  if (values == NULL) return GrB_NULL_POINTER;
  return block_matrix_.extractTuples(values, n);
}

// handy method
template <typename T>
void BlockMatrix<T>::operator=(const BlockMatrix& rhs) {
    block_matrix_.dup(&rhs.block_matrix_);
}

template <typename T>
const T BlockMatrix<T>::operator[](Index ind) { // why not BlockMatrix<T> ??
    return block_matrix_[ind];
}

template <typename T>
Info BlockMatrix<T>::print(bool force_update) {
    return block_matrix_.print(force_update);
}

template <typename T>
Info BlockMatrix<T>::check() {
    return block_matrix_.check();
}

template <typename T>
Info BlockMatrix<T>::setNrows(Index nrows) {
    return block_matrix_.setNrows(nrows);
}

template <typename T>
Info BlockMatrix<T>::setNcols(Index ncols) {
    return block_matrix_.setNcols(ncols);
}

template <typename T>
Info BlockMatrix<T>::resize(Index nrows, Index ncols) {
  return block_matrix_.resize(nrows, ncols);
}

template <typename T>
Info BlockMatrix<T>::setStorage(Storage mat_type) { // in graphblas/types.hpp
  return block_matrix_.setStorage(mat_type);
}

template <typename T>
Info BlockMatrix<T>::getStorage(Storage* mat_type) const {
  if (mat_type == NULL) return GrB_NULL_POINTER;
  return block_matrix_.getStorage(mat_type);
}

template <typename T>
template <typename U>
Info BlockMatrix<T>::fill(Index axis, Index nvals, U start) {
  return block_matrix_.fill(axis, nvals, start);
}

template <typename T>
template <typename U>
Info BlockMatrix<T>::fillAscending(Index axis, Index nvals, U start) {
  return block_matrix_.fillAscending(axis, nvals, start);
}

}// namespace graphblas

#endif  // GRAPHBLAS_BLOCK_MATRIX_HPP_