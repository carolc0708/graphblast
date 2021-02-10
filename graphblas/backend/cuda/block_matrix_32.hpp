#ifndef GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_HPP_
#define GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_HPP_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <cstring> // memset
#include <cassert>
#include <cstdio>

#include <vector>
#include <iostream>
#include <algorithm>

#include "graphblas/util.hpp"

////////////////////// CUDA ERROR /////////////////////////////////////////

static void CudaCheckCore(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cudaGetErrorString(code), file, line);
      exit(code);
   }
}

#define CudaCheck( test ) { CudaCheckCore((test), __FILE__, __LINE__); }
#define CudaCheckAfterCall() { CudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

////////////////////// CUDA SPARSE ERROR ///////////////////////////////////

static const char * cusparseGetErrorString(cusparseStatus_t error)
{
    // Read more at: http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar
    switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "The operation completed successfully.";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
               "To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
                "To correct: prior to the function call, deallocate previously allocated memory as much as possible.";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
            "To correct: ensure that all the parameters being passed have valid values.";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
            "To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
            "To correct: prior to the function call, unbind any previously bound textures.";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
                "To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
                "To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
    }

    return "<unknown>";
}
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file, int line) {
   if (code != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr,"Cuda Error %d : %s %s %d\n", code, cusparseGetErrorString(code), file, line);
      exit(code);
   }
}

#define CudaSparseCheck( test ) { CudaSparseCheckCore((test), __FILE__, __LINE__); }



namespace graphblas {
namespace backend {

template <typename T>
class SparseMatrix; // we need to rely on this to build blockmatrix

template <typename T>
class BlockMatrix32 {
 public:
    // default constructor
    BlockMatrix32()
        : nrows_(0), ncols_(0), nvals_(0), ncapacity_(0), nempty_(0),
        h_bcsrRowPtr_(NULL), h_bcsrColInd_(NULL), h_bcsrVal_(NULL),
        h_bcscColPtr_(NULL), h_bcscRowInd_(NULL), h_bcscVal_(NULL),
        d_bcsrRowPtr_(NULL), d_bcsrColInd_(NULL), d_bcsrVal_(NULL),
        d_bcscColPtr_(NULL), d_bcscRowInd_(NULL), d_bcscVal_(NULL),
        need_update_(0), symmetric_(0) {
        format_ = getEnv("GRB_SPARSE_MATRIX_FORMAT", GrB_BLOCK_MATRIX_32_BCSRBCSC);

        // cusparse handle
        cudaStream_t streamId;

        streamId = 0;
        cusparseHandle = 0;
        CudaSparseCheck(cusparseCreate(&cusparseHandle));
        CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
    }

    explicit BlockMatrix32(Index nrows, Index ncols)
        : nrows_(nrows), ncols_(ncols), nvals_(0), ncapacity_(0), nempty_(0),
        h_bcsrRowPtr_(NULL), h_bcsrColInd_(NULL), h_bcsrVal_(NULL),
        h_bcscColPtr_(NULL), h_bcscRowInd_(NULL), h_bcscVal_(NULL),
        d_bcsrRowPtr_(NULL), d_bcsrColInd_(NULL), d_bcsrVal_(NULL),
        d_bcscColPtr_(NULL), d_bcscRowInd_(NULL), d_bcscVal_(NULL),
        need_update_(0), symmetric_(0) {
        format_ = getEnv("GRB_SPARSE_MATRIX_FORMAT", GrB_BLOCK_MATRIX_32_BCSRBCSC);

        // cusparse handle
        cudaStream_t streamId;

        streamId = 0;
        cusparseHandle = 0;
        CudaSparseCheck(cusparseCreate(&cusparseHandle));
        CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
    }

    // default destructor
    ~BlockMatrix32();

    // C API methods
    Info nnew(Index nrows, Index ncols);
    Info dup(const BlockMatrix32* rhs);
    Info clear();     // 1 way to free: (1) clear
    Info nrows(Index* nrows_t) const;
    Info ncols(Index* ncols_t) const;
    Info nvals(Index* nvals_t) const;
    template <typename BinaryOpT>
    Info build(const std::vector<Index>* row_indices,
               const std::vector<Index>* col_indices,
               const std::vector<T>*     values,
               Index                     nvals,
               BinaryOpT                 dup,
               char*                     dat_name);
    Info setElement(Index row_index,
              Index col_index);
    Info extractElement(T*    val,
                  Index row_index,
                  Index col_index);
    Info extractTuples(std::vector<Index>* row_indices,
                 std::vector<Index>* col_indices,
                 std::vector<T>*     values,
                 Index*              n);
    Info extractTuples(std::vector<T>* values,
                 Index*          n);

    // handy method
    const T operator[](Index ind);
    Info print(bool force_update);
    Info check();
    Info setNrows(Index nrows);
    Info setNcols(Index ncols);
    Info setNvals(Index nvals);
    Info getFormat(SparseMatrixFormat* format) const;
    Info getSymmetry(bool* symmetry) const;
    Info resize(Index nrows, Index ncols);
    template <typename U>
    Info fill(Index axis, Index nvals, U start);
    template <typename U>
    Info fillAscending(Index axis, Index nvals, U start);

 private:
    Info allocateCpu();
    Info allocateGpu();
    Info allocate();  // 3 ways to allocate: (1) dup, (2) build, (3) spgemm
                    //                     (4) fill,(5) fillAscending
    Info printBCSR(const char* str);  // private method for pretty printing
    Info printBCSC(const char* str);
    Info cpuToGpu();
    Info gpuToCpu(bool force_update = false);

    Info syncCpu();   // synchronizes CSR and CSC representations

    // block csr specific
    void csr2bcsr(SparseMatrix<T>* A, const Index blocksize);

 private:
    const int kcap_ratio_ = 1;
    const int kresize_ratio_ = 1;
    //const T kcap_ratio_    = 1.2f;  // Note: nasty bug if this is set to 1.f!
    //const T kresize_ratio_ = 1.2f;

    Index nrows_;
    Index ncols_;
    Index nvals_;     // 3 ways to set: (1) dup (2) build (3) nnew
    Index ncapacity_;
    Index nempty_; // non-used variable

    // block csr specific
    Index nblocks_;
    Index nblockrow_;
    Index blocksize_;

    Index* h_bcsrRowPtr_;  // CSR format
    Index* h_bcsrColInd_;
    T*     h_bcsrVal_;
    Index* h_bcscColPtr_;  // CSC format
    Index* h_bcscRowInd_;
    T*     h_bcscVal_;

    Index* d_bcsrRowPtr_;  // GPU CSR format
    Index* d_bcsrColInd_;
    T*     d_bcsrVal_;
    Index* d_bcscColPtr_;  // GPU CSC format
    Index* d_bcscRowInd_;
    T*     d_bcscVal_;

    // GPU variables
    bool need_update_;
    bool symmetric_;

    SparseMatrixFormat format_;

    // cusparse handle
    cusparseHandle_t cusparseHandle;

}; // class BlockMatrix32

template <typename T>
BlockMatrix32<T>::~BlockMatrix32() {
    if (h_bcsrRowPtr_) free(h_bcsrRowPtr_);
    if (h_bcsrColInd_) free(h_bcsrColInd_);
    if (h_bcsrVal_   ) free(h_bcsrVal_);
    if (d_bcsrRowPtr_) CUDA_CALL(cudaFree(d_bcsrRowPtr_));
    if (d_bcsrColInd_) CUDA_CALL(cudaFree(d_bcsrColInd_));
    if (d_bcsrVal_   ) CUDA_CALL(cudaFree(d_bcsrVal_   ));

    if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
        if (h_bcscColPtr_) free(h_bcscColPtr_);
        if (h_bcscRowInd_) free(h_bcscRowInd_);
        if (h_bcscVal_   ) free(h_bcscVal_);

        if (!symmetric_) {
            if (d_bcscColPtr_) CUDA_CALL(cudaFree(d_bcscColPtr_));
            if (d_bcscRowInd_) CUDA_CALL(cudaFree(d_bcscRowInd_));
            if (d_bcscVal_   ) CUDA_CALL(cudaFree(d_bcscVal_   ));
        }
    }

    // destroy sparse handle
    CudaSparseCheck(cusparseDestroy(cusparseHandle));
}

template <typename T>
Info BlockMatrix32<T>::nnew(Index nrows, Index ncols) { // assume squared
  nrows_ = nrows;
  ncols_ = ncols;

  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::dup(const BlockMatrix32* rhs) {
    if (nrows_ != rhs->nrows_) return GrB_DIMENSION_MISMATCH;
    if (ncols_ != rhs->ncols_) return GrB_DIMENSION_MISMATCH;
    nvals_ = rhs->nvals_;
    symmetric_ = rhs->symmetric_;
    format_ = rhs->format_;
    nblocks_ = rhs->nblocks_;
    nblockrow_ = rhs->nblockrow_;
    blocksize_ = rhs->blocksize_;


    CHECK(allocate());

    CUDA_CALL(cudaMemcpy(d_bcsrRowPtr_, rhs->d_bcsrRowPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(d_bcsrColInd_, rhs->d_bcsrColPtr_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaMemcpy(d_bcsrVal_, rhs->d_bcsrVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToDevice));

    if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
        CUDA_CALL(cudaMemcpy(d_bcscVal_, rhs->d_bcscVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToDevice));
        if (!symmetric_ && format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
            CUDA_CALL(cudaMemcpy(d_bcscColPtr_, rhs->d_bcscColPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaMemcpy(d_bcscRowInd_, rhs->d_bcscRowInd_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToDevice));
        }
    }

    need_update_ = true;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::clear() {
    nvals_ = 0;
    ncapacity_ = 0;

    if (h_bcsrRowPtr_) free(h_bcsrRowPtr_);
    if (h_bcsrColInd_) free(h_bcsrColInd_);
    if (h_bcsrVal_    ) free(h_bcsrVal_);
    if (d_bcsrRowPtr_) CUDA_CALL(cudaFree(d_bcsrRowPtr_));
    if (d_bcsrColInd_) CUDA_CALL(cudaFree(d_bcsrColInd_));
    if (d_bcsrVal_    ) CUDA_CALL(cudaFree(d_bcsrVal_));

    h_bcsrRowPtr_ = NULL;
    h_bcsrColInd_ = NULL;
    h_bcsrVal_    = NULL;
    d_bcsrRowPtr_ = NULL;
    d_bcsrColInd_ = NULL;
    d_bcsrVal_    = NULL;

    if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
        if (h_bcscColPtr_) free(h_bcscColPtr_);
        if (h_bcscRowInd_) free(h_bcscRowInd_);
        if (h_bcscVal_   ) free(h_bcscVal_);
        if (d_bcscVal_   ) CUDA_CALL(cudaFree(d_bcscVal_));

        if (!symmetric_) {
            if (d_bcscColPtr_) CUDA_CALL(cudaFree(d_bcscColPtr_));
            if (d_bcscRowInd_) CUDA_CALL(cudaFree(d_bcscRowInd_));
        }
    }

    return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix32<T>::nrows(Index* nrows_t) const {
    *nrows_t = nrows_;
    return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix32<T>::ncols(Index* ncols_t) const {
    *ncols_t = ncols_;
    return GrB_SUCCESS;
}

template <typename T>
inline Info BlockMatrix32<T>::nvals(Index* nvals_t) const {
    *nvals_t = nvals_;
    return GrB_SUCCESS;
}

template <typename T>
template <typename BinaryOpT>
Info BlockMatrix32<T>::build(const std::vector<Index>* row_indices,
                             const std::vector<Index>* col_indices,
                             const std::vector<T>*     values,
                             Index                     nvals,
                             BinaryOpT                 dup,
                             char*                     dat_name) {

    /* read in the matrix as csr first */
    // Initialize sparse matrix A
    SparseMatrix<T> A(nrows_, ncols_);
    Info err = A.build(row_indices, col_indices, values, nvals, dup, dat_name);
    //A.print(false);

    /* and then transform it into bcsr, set bcsr metadata */
    csr2bcsr(&A, 32);

    /* transfer some metadata from csr */
    A.nvals(&nvals_); // set bcsr nvals
    ncapacity_ = nvals_ * kcap_ratio_;
    A.getSymmetry(&symmetric_); //set symmetric_

    //std::cout << nvals_ << ", " << ncapacity_ << ", " << (symmetric_?"true":"false") << std::endl;

    /* copy device memory to host */
    // allocate cpu first
    CHECK(allocateCpu());

    CUDA_CALL(cudaMemcpy(h_bcsrRowPtr_, d_bcsrRowPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_bcsrColInd_, d_bcsrColInd_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_bcsrVal_, d_bcsrVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CALL(cudaMemcpy(h_bcscColPtr_, d_bcscColPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_bcscRowInd_, d_bcscRowInd_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_bcscVal_, d_bcscVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToHost));

    //A.~SparseMatrix();

    return err;
 }

template <typename T>
Info BlockMatrix32<T>::setElement(Index row_index, Index col_index) {
  std::cout << "BlockMatrix32 setElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::extractElement(T* val, Index row_index, Index col_index) {
  std::cout << "BlockMatrix32 extractElement\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::extractTuples(std::vector<Index>* row_indices,
                                     std::vector<Index>* col_indices,
                                     std::vector<T>*     values,
                                     Index*              n) { // (TODO) may have some problem
  CHECK(gpuToCpu());
  row_indices->clear();
  col_indices->clear();
  values->clear();

  if (*n > nvals_) {
    std::cout << "Error: Too many tuples requested!\n";
    return GrB_UNINITIALIZED_OBJECT;
  }

  if (*n < nvals_) {
    std::cout << "Error: Insufficient space!\n";
    return GrB_INSUFFICIENT_SPACE;
  }

  Index count = 0;
  for (Index row = 0; row < nrows_; row++) {
    for (Index ind = h_bcsrRowPtr_[row]; ind < h_bcsrRowPtr_[row+1]; ind++) {
      if (h_bcsrVal_[ind] != 0 && count < *n) {
        count++;
        row_indices->push_back(row);
        col_indices->push_back(h_bcsrColInd_[ind]);
        values->push_back(h_bcsrVal_[ind]);
      }
    }
  }

  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::extractTuples(std::vector<T>* values, Index* n) {
  std::cout << "BlockMatrix32 extractTuples to dense vector\n";
  std::cout << "Error: Feature not implemented yet!\n";
  return GrB_SUCCESS;
}

template <typename T>
const T BlockMatrix32<T>::operator[](Index ind) {
    CHECKVOID(gpuToCpu(true));
    if (ind >= nvals_) std::cout << "Error: index out of bounds!\n";
    return h_bcsrColInd_[ind];
}

template <typename T>
Info BlockMatrix32<T>::print(bool force_update) {
  CHECK(gpuToCpu(force_update));
  printArray("bcsrColInd", h_bcsrColInd_, std::min(nblocks_, 40));
  printArray("bcsrRowPtr", h_bcsrRowPtr_, std::min(nblockrow_+1, 40));
  printArray("bcsrVal",    h_bcsrVal_,    nblocks_); // (TODO) revise this print format
  //CHECK(printBCSR("pretty print")); // also this
//  if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
//    if (!h_bcscRowInd_ || !h_bcscColPtr_ || !h_bcscVal_)
//      //syncCpu();
//    printArray("bcscRowInd", h_bcscRowInd_, std::min(nblocks_, 40));
//    printArray("bcscColPtr", h_bcscColPtr_, std::min(nblockrow_+1, 40));
//    printArray("bcscVal",    h_bcscVal_,    nblocks_); // (TODO) revise this print format
//    //CHECK(printBCSC("pretty print")); // and this
//  }
  return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::check() {
    CHECK(gpuToCpu());
    std::cout << "Begin check:\n";

    // Check csrRowPtr is monotonically increasing
    for (Index row = 0; row < nrows_; row++)
        assert(h_bcsrRowPtr_[row+1] >= h_bcsrRowPtr_[row]);


    // Check that: 1) there are no -1's in ColInd
    //             2) monotonically increasing
    for (Index row = 0; row < nrows_; row++) {
        Index row_start = h_bcsrRowPtr_[row];
        Index row_end = h_bcsrRowPtr_[row+1];
        Index p_end = h_bcsrRowPtr_[row+1];

        for (Index col = row_start; col < row_end-1; col++) {
            assert(h_bcsrColInd_[col] != -1);
            assert(h_bcsrColInd_[col+1] >= h_bcsrColInd_[col]);
            assert(h_bcsrColInd_[col] > 0);
        }
        for (Index col = row_end; col < p_end; col++)
            assert(h_bcsrColInd_[col] == -1);
    }

    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::setNrows(Index nrows) {
    nrows_ = nrows;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::setNcols(Index ncols) {
    ncols_ = ncols;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::setNvals(Index nvals) {
    nvals_ = nvals;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::getFormat(SparseMatrixFormat* format) const {
    *format = format_;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::getSymmetry(bool* symmetry) const {
    *symmetry = symmetric_;
    *symmetry = false;
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::resize(Index nrows, Index ncols) {
    if (nrows <= nrows_) nrows_ = nrows;
    else return GrB_PANIC;
    if (ncols <= ncols_) ncols_ = ncols;
    else return GrB_PANIC;

    return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info BlockMatrix32<T>::fill(Index axis, Index nvals, U start) { //(TODO) should check this
    CHECK(setNvals(nvals));
    CHECK(allocate());

    if (axis == 0) {
        for (Index i = 0; i < nvals; i++)
            h_bcsrRowPtr_[i] = (Index) start;
    } else if (axis == 1) {
        for (Index i = 0; i < nvals; i++)
            h_bcsrColInd_[i] = (Index) start;
    } else if (axis == 2) {
        for (Index i = 0; i < nvals; i++)
            h_bcsrVal_[i] = (T) start;
    }

    CHECK(cpuToGpu());
    return GrB_SUCCESS;
}

template <typename T>
template <typename U>
Info BlockMatrix32<T>::fillAscending(Index axis, Index nvals, U start) { //(TODO) should check this
    CHECK(setNvals(nvals));
    CHECK(allocate());

    if (axis == 0) {
        for (Index i = 0; i < nvals; i++)
            h_bcsrRowPtr_[i] = i + (Index) start;
    } else if (axis == 1) {
        for (Index i = 0; i < nvals; i++)
            h_bcsrColInd_[i] = i + (Index) start;
    } else if (axis == 2) {
        for (Index i = 0; i < nvals; i++)
            h_bcscVal_[i] = (T)i + start;
    }

    CHECK(cpuToGpu());
    return GrB_SUCCESS;
}

template <typename T>
Info BlockMatrix32<T>::allocateCpu() {
    // allocate
    ncapacity_ = kcap_ratio_ * nblocks_; // (TODO) not adopting capacity currently

    // host malloc
    if (nblockrow_ > 0 && h_bcsrRowPtr_ == NULL)
        h_bcsrRowPtr_ = reinterpret_cast<Index*>(malloc((nblockrow_+1) * sizeof(Index)));
    if (nblocks_ > 0 && h_bcsrColInd_ == NULL)
        h_bcsrColInd_ = reinterpret_cast<Index*>(malloc(nblocks_ * sizeof(Index)));
    if (nblocks_ > 0 && h_bcsrVal_ == NULL)
        h_bcsrVal_ = reinterpret_cast<T*>(malloc(nblocks_ * (blocksize_ * blocksize_) * sizeof(T)));

    if (nblockrow_ > 0 && h_bcscColPtr_ == NULL) {
        h_bcscColPtr_ = reinterpret_cast<Index*>(malloc((nblockrow_+1) * sizeof(Index)));
        std::cout << "Allocate " << nblockrow_+1 << std::endl;
    } else std::cout << "Do not allocate " << nblockrow_+1 << " " << h_bcscColPtr_ << std::endl;

    if (nblocks_ > 0 && h_bcscRowInd_ == NULL) {
        h_bcscRowInd_ = reinterpret_cast<Index*>(malloc(nblocks_ * sizeof(Index)));
        std::cout << "Allocate " << nblocks_ << std::endl;
    } else {
        std::cout << "Do not allocate " << nblocks_ << " " << h_bcscRowInd_ << std::endl;
    }

    if (nblocks_ > 0 && h_bcscVal_ == NULL) {
        h_bcscVal_ = reinterpret_cast<T*>(malloc(nblocks_ * (blocksize_ * blocksize_) * sizeof(T)));
        std::cout << "Allocate " << nblocks_ * (blocksize_ * blocksize_) << std::endl;
    } else {
        std::cout << "Do not allocate " << nblocks_ * (blocksize_ * blocksize_) << " " << h_bcscVal_ << std::endl;
    }

    return GrB_SUCCESS;
}

// allocategpu
template <typename T>
Info BlockMatrix32<T>::allocateGpu() { // (TODO) not adopting capacity currently
    // GPU malloc
    if (nrows_ > 0 && d_bcsrRowPtr_ == NULL)
        CUDA_CALL(cudaMalloc(&d_bcsrRowPtr_, (nblockrow_+1) * sizeof(Index)));
    if (nvals_ > 0 && d_bcsrColInd_ == NULL)
        CUDA_CALL(cudaMalloc(&d_bcsrColInd_, nblocks_ * sizeof(Index)));
    if (nvals_ > 0 && d_bcsrVal_ == NULL) {
        CUDA_CALL(cudaMalloc(&d_bcsrVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T)));
        printMemory("bcsrVal");
    }

    if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
        if (nvals_ > 0 && d_bcscVal_ == NULL) {
            CUDA_CALL(cudaMalloc(&d_bcscVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T)));
            printMemory("bcscVal");
            if (!symmetric_) {
                if (nrows_ > 0 && d_bcscColPtr_ == NULL)
                    CUDA_CALL(cudaMalloc(&d_bcscColPtr_, (nblockrow_+1) * sizeof(Index)));
                if (nvals_ > 0 && d_bcscRowInd_ == NULL)
                    CUDA_CALL(cudaMalloc(&d_bcscRowInd_, nblocks_ * sizeof(Index)));
            }
         }
    }

    return GrB_SUCCESS;
}

// allocate
template <typename T>
Info BlockMatrix32<T>::allocate() {
    CHECK(allocateCpu());
    CHECK(allocateGpu());
    return GrB_SUCCESS;
}

// print bcsr
template <typename T>
Info BlockMatrix32<T>::printBCSR(const char* str) {
    Index row_length = std::min(20, nrows_);
    Index col_length = std::min(20, ncols_);
    std::cout << str << ":\n";

    for (Index row = 0; row < row_length; row++) {
        Index col_start = h_bcsrRowPtr_[row];
        Index col_end   = h_bcsrRowPtr_[row+1];
        for (Index col = 0; col < col_length; col++) {
            Index col_ind = h_bcsrColInd_[col_start];
            if (col_start < col_end && col_ind == col && h_bcsrVal_[col_start] > 0) {
                std::cout << "x ";
                col_start++;
            } else {
                std::cout << "0 ";
            }
        }
        std::cout << std::endl;
    }

    return GrB_SUCCESS;
}

// print bcsc
template <typename T>
Info BlockMatrix32<T>::printBCSC(const char* str) {
    Index row_length = std::min(20, nrows_);
    Index col_length = std::min(20, ncols_);
    std::cout << str << ":\n";

    for (Index row = 0; row < col_length; row++) {
        Index col_start = h_bcscColPtr_[row];
        Index col_end   = h_bcscColPtr_[row+1];
        for (Index col = 0; col < row_length; col++) {
          Index col_ind = h_bcscRowInd_[col_start];
          if (col_start < col_end && col_ind == col && h_bcscVal_[col_start] > 0) {
            std::cout << "x ";
            col_start++;
          } else {
            std::cout << "0 ";
          }
        }
        std::cout << std::endl;
    }
    return GrB_SUCCESS;
}

// copy graph to gpu
template <typename T>
Info BlockMatrix32<T>::cpuToGpu() {
    CHECK(allocateGpu());

    CUDA_CALL(cudaMemcpy(d_bcsrRowPtr_, h_bcsrRowPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bcsrColInd_, h_bcsrColInd_, nblocks_ * sizeof(Index), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_bcsrVal_,    h_bcsrVal_,    nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyHostToDevice));

    if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
        CUDA_CALL(cudaMemcpy(d_bcscVal_, h_bcscVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyHostToDevice));

        if (!symmetric_) {
          CUDA_CALL(cudaMemcpy(d_bcscColPtr_, h_bcscColPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyHostToDevice));
          CUDA_CALL(cudaMemcpy(d_bcscRowInd_, h_bcscRowInd_, nblocks_* sizeof(Index), cudaMemcpyHostToDevice));
        } else {
          d_bcscColPtr_ = d_bcsrRowPtr_;
          d_bcscRowInd_ = d_bcsrColInd_;
        }
    }

    return GrB_SUCCESS;
}

// copy graph to cpu
template <typename T>
Info BlockMatrix32<T>::gpuToCpu (bool force_update) {
    if (need_update_ || force_update) {
        CUDA_CALL(cudaMemcpy(h_bcsrRowPtr_, d_bcsrRowPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_bcsrColInd_, d_bcsrColInd_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(h_bcsrVal_, d_bcsrVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToHost));

        if (format_ == GrB_BLOCK_MATRIX_32_BCSRBCSC) {
            // Must account for combination of:
            // 1) CSRCSC
            // 2) sparse matrix being output of matrix-matrix multiply
            // In this case, the CSC copy does not exist, which causes an error.
            if (d_bcscVal_ && d_bcscColPtr_ && d_bcscRowInd_ && h_bcscVal_ && h_bcscColPtr_ && h_bcscRowInd_) {
                CUDA_CALL(cudaMemcpy(h_bcscVal_, d_bcscVal_, nblocks_ * (blocksize_ * blocksize_) * sizeof(T), cudaMemcpyDeviceToHost));
                if (!symmetric_) {
                    CUDA_CALL(cudaMemcpy(h_bcscColPtr_, d_bcscColPtr_, (nblockrow_+1) * sizeof(Index), cudaMemcpyDeviceToHost));
                    CUDA_CALL(cudaMemcpy(h_bcscRowInd_, d_bcscRowInd_, nblocks_ * sizeof(Index), cudaMemcpyDeviceToHost));
                }
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());
    }
    need_update_ = false;
    return GrB_SUCCESS;
}

// synchronizes csr with csc
template <typename T>
Info BlockMatrix32<T>::syncCpu() { // not valid right now
  CHECK(allocateCpu());
  if (h_bcsrRowPtr_ && h_bcsrColInd_ && h_bcsrVal_ &&
      h_bcscColPtr_ && h_bcscRowInd_ && h_bcscVal_)
    csr2csc(h_bcscColPtr_, h_bcscRowInd_, h_bcscVal_,
            h_bcsrRowPtr_, h_bcsrColInd_, h_bcsrVal_, nrows_, ncols_);
  else
    return GrB_INVALID_OBJECT;
  return GrB_SUCCESS;
}

// csr to bcsr
template <typename T>
void BlockMatrix32<T>::csr2bcsr(SparseMatrix<T>* A, const Index blocksize) {
    // metadata
    blocksize_ = blocksize;
    nblockrow_ = (nrows_ + blocksize_ - 1) / blocksize_;
    cudaMalloc((void **)&d_bcsrRowPtr_, sizeof(Index) * (nblockrow_ + 1));

    // create cusparsematdescr -- csr
    cusparseMatDescr_t csr_descr = 0;
    CudaSparseCheck(cusparseCreateMatDescr(&csr_descr));
    cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO);

    // create cusparsematdescr -- bcsr
    cusparseMatDescr_t bcsr_descr = 0;
    CudaSparseCheck(cusparseCreateMatDescr(&bcsr_descr));
    cusparseSetMatType(bcsr_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bcsr_descr,CUSPARSE_INDEX_BASE_ZERO);

    // should separte two different descr, although content are the same

    // count nnz blocks
    Index base, nbNnzBlocks;
    Index* nnzTotalDevHostPtr = &nbNnzBlocks;

    cusparseXcsr2bsrNnz(cusparseHandle, CUSPARSE_DIRECTION_COLUMN, nrows_, ncols_, csr_descr,
                    A->d_csrRowPtr_, A->d_csrColInd_, blocksize_,
                    bcsr_descr, d_bcsrRowPtr_, nnzTotalDevHostPtr);


    if (NULL != nnzTotalDevHostPtr){
        nbNnzBlocks = *nnzTotalDevHostPtr;
    } else{
        cudaMemcpy(&nbNnzBlocks, d_bcsrRowPtr_+nblockrow_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, d_bcsrRowPtr_, sizeof(int), cudaMemcpyDeviceToHost);
        nbNnzBlocks -= base;
    }

    nblocks_ = nbNnzBlocks;

    //std::cout << "nblocks_: " << nblocks_ << ", blocksize_: " << blocksize_ << ", nblockrow_: " << nblockrow_ << std::endl;


    CudaCheck(cudaMalloc((void**)&d_bcsrColInd_, sizeof(Index) * nbNnzBlocks));
    CudaCheck(cudaMalloc((void**)&d_bcsrVal_, sizeof(T) * (blocksize_ * blocksize_) * nbNnzBlocks)); // out of memory error
    cusparseScsr2bsr(cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                     nrows_, ncols_, csr_descr, A->d_csrVal_, A->d_csrRowPtr_,
                     A->d_csrColInd_, blocksize_, bcsr_descr, d_bcsrVal_, d_bcsrRowPtr_, d_bcsrColInd_);

    //std::cout << "nblocks_: " << nblocks_ << ", blocksize_: " << blocksize_ << ", nblockrow_: " << nblockrow_ << std::endl;

    // copy d_csr to d_csc
    CudaCheck(cudaMalloc((void **)&d_bcscColPtr_, sizeof(Index) * (nblockrow_ + 1)));
    CudaCheck(cudaMemcpy(d_bcscColPtr_, d_bcsrRowPtr_, sizeof(Index) * (nblockrow_ + 1), cudaMemcpyDeviceToDevice));
    CudaCheck(cudaMalloc((void **)&d_bcscRowInd_, sizeof(Index) * nblocks_));
    CudaCheck(cudaMemcpy(d_bcscRowInd_, d_bcsrColInd_, sizeof(Index) * nblocks_, cudaMemcpyDeviceToDevice));
    CudaCheck(cudaMalloc((void **)&d_bcscVal_, sizeof(T) * (blocksize_ * blocksize_) * nblocks_));
    CudaCheck(cudaMemcpy(d_bcscVal_, d_bcsrVal_, sizeof(T) * (blocksize_ * blocksize_) * nblocks_, cudaMemcpyDeviceToDevice));

    // free descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    csr_descr = 0;
    cusparseDestroyMatDescr(bcsr_descr);
    bcsr_descr = 0;
    cusparseDestroy(cusparseHandle);
    cusparseHandle = 0;
}


} // namespace backend
} // namespace graphblas

#endif // GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_HPP_