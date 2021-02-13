#ifndef GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_SPMV_HPP_
#define GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_SPMV_HPP_

#include <cuda.h>
#include <cusparse.h>

#include <moderngpu.cuh>
#include <cub.cuh>

#include <iostream>
#include <string>

#include "graphblas/backend/cuda/kernels/kernels.hpp"

#define BCSR

namespace graphblas {
namespace backend {

template <typename W, typename a, typename U, typename M,
          typename BinaryOpT, typename SemiringT>
Info spmv(DenseVector<W>*        w,
          const Vector<M>*       mask,
          BinaryOpT              accum,
          SemiringT              op,
          const BlockMatrix32<a>* A,
          const DenseVector<U>*  u,
          Descriptor*            desc) { // w = Au. * mask

    // Get descriptor parameters for SCMP, REPL, TRAN
    Desc_value scmp_mode, repl_mode, inp0_mode, inp1_mode;
    CHECK(desc->get(GrB_MASK, &scmp_mode));
    CHECK(desc->get(GrB_OUTP, &repl_mode));
    CHECK(desc->get(GrB_INP0, &inp0_mode));
    CHECK(desc->get(GrB_INP1, &inp1_mode));


    std::string accum_type = typeid(accum).name();
    // TODO(@ctcyang): add accum and replace support
    // -have masked variants as separate kernel
    // -have scmp as template parameter
    // -accum and replace as parts in flow
    bool use_mask  = (mask != NULL);
    bool use_accum = (accum_type.size() > 1); // output overwrite w or accumulated to w
    bool use_scmp  = (scmp_mode == GrB_SCMP); // structural complement of mask
    bool use_repl  = (repl_mode == GrB_REPLACE); // do not/do clear output before writing mask indices
    bool use_tran  = (inp0_mode == GrB_TRAN || inp1_mode == GrB_TRAN); // do not/do transpose first/second input param

    if (desc->debug()) {
        std::cout << "Executing Spmv\n";
        printState(use_mask, use_accum, use_scmp, use_repl, use_tran);
    }

    // Transpose (default is CSR):
    const Index* A_csrRowPtr = (use_tran) ? A->d_bcscColPtr_ : A->d_bcsrRowPtr_;
    const Index* A_csrColInd = (use_tran) ? A->d_bcscRowInd_ : A->d_bcsrColInd_;
    const a*     A_csrVal    = (use_tran) ? A->d_bcscVal_    : A->d_bcsrVal_;
    const Index  A_nrows     = (use_tran) ? A->ncols_       : A->nrows_;

    if (desc->debug()) {
        std::cout << "bcscColPtr: " << A->d_bcscColPtr_ << std::endl;
        std::cout << "bcscRowInd: " << A->d_bcscRowInd_ << std::endl;
        std::cout << "bcscVal:    " << A->d_bcscVal_    << std::endl;

        std::cout << "bcsrRowPtr: " << A->d_bcsrRowPtr_ << std::endl;
        std::cout << "bcsrColInd: " << A->d_bcsrColInd_ << std::endl;
        std::cout << "bcsrVal:    " << A->d_bcsrVal_    << std::endl;
    }

    // Get descriptor parameters for nthreads
    Desc_value ta_mode, tb_mode, nt_mode;
    CHECK(desc->get(GrB_TA, &ta_mode));
    CHECK(desc->get(GrB_TB, &tb_mode));
    CHECK(desc->get(GrB_NT, &nt_mode)); // seems to be 8 (in graphblas/types.hpp)

    const int ta = static_cast<int>(ta_mode);
    const int tb = static_cast<int>(tb_mode);
    const int nt = static_cast<int>(nt_mode);

    /*!
    * /brief atomicAdd() 3+5  = 8
    *        atomicSub() 3-5  =-2
    *        atomicMin() 3,5  = 3
    *        atomicMax() 3,5  = 5
    *        atomicOr()  3||5 = 1
    *        atomicXor() 3^^5 = 0
    */
    auto add_op = extractAdd(op);
    int functor = add_op(3, 5);

    if (desc->debug()) {
        std::cout << "Fused mask: " << desc->fusedmask() << std::endl;
        std::cout << "Functor:    " << functor << std::endl;
    }

    if (desc->struconly() && functor != 1)
        std::cout << "Warning: Using structure-only mode and not using logical or "
            << "semiring may result in unintended behaviour. Is this intended?\n";

    if (false){//use_mask && desc->fusedmask() && functor == 1) {
        // Mask type
        // 1) Dense mask
        // 2) Sparse mask TODO(@ctcyang)
        // 3) Uninitialized
        Storage mask_vec_type;
        CHECK(mask->getStorage(&mask_vec_type));

        if (mask_vec_type == GrB_DENSE) { // only dense mask is support
            dim3 NT, NB;
            NT.x = nt;
            NT.y = 1;
            NT.z = 1;
            NB.x = (A_nrows+nt-1)/nt;
            NB.y = 1;
            NB.z = 1;

            int variant = 0;
            variant |= use_scmp          ? 4 : 0; // structural complement of mask
            variant |= desc->earlyexit() ? 2 : 0;
            variant |= desc->opreuse()   ? 1 : 0;

            // switch variant ...
            switch (variant) {
                case 0:
                    spmvDenseMaskedOrKernel<false, false, false><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 1:
                    spmvDenseMaskedOrKernel<false, false, true><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 2:
                    spmvDenseMaskedOrKernel<false, true, false><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 3:
                    spmvDenseMaskedOrKernel<false, true, true><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 4:
                    spmvDenseMaskedOrKernel<true, false, false><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 5:
                    spmvDenseMaskedOrKernel<true, false, true><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 6:
                    spmvDenseMaskedOrKernel<true, true, false><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                case 7:
                    spmvDenseMaskedOrKernel<true, true, true><<<NB, NT>>>(
                      w->d_val_, mask->dense_.d_val_, NULL, op.identity(),
                      extractMul(op), extractAdd(op), A_nrows, A->nvals_,
                      A_csrRowPtr, A_csrColInd, A_csrVal, u->d_val_);
                break;
                default:
                break;
            }
            if (desc->debug())
            printDevice("w_val", w->d_val_, A_nrows);

        } else if (mask_vec_type == GrB_SPARSE) {
            std::cout << "DeVec Sparse Mask logical_or Spmv\n";
            std::cout << "Error: Feature not implemented yet!\n";

        } else return GrB_UNINITIALIZED_OBJECT;

    } else { // not (use_mask && desc->fusedmask() && functor == 1)
        Index* w_ind;
        W*     w_val;

        if (use_accum) {
            CHECK(desc->resize(A_nrows * sizeof(W), "buffer"));
            w_val = reinterpret_cast<W*>(desc->d_buffer_);
        } else w_val = w->d_val_;

///
// template<typename MatrixIt, typename ColsIt, typename CsrIt, typename VecIt,
//	typename DestIt, typename T, typename MulOp, typename AddOp>
// MGPU_HOST void SpmvCsrBinary(MatrixIt matrix_global, ColsIt cols_global, int nz,
//	CsrIt csr_global, int numRows, VecIt vec_global, bool supportEmpty,
//	DestIt dest_global, T identity, MulOp mulOp, AddOp addOp,
//	CudaContext& context)

#ifdef ORIG
        // cannot be use by BlockMatrix32 type
        mgpu::SpmvCsrBinary(A_csrVal, A_csrColInd, A->nvals_, A_csrRowPtr, A_nrows,
                            u->d_val_, true, w_val, op.identity(), extractMul(op), extractAdd(op),
                            *(desc->d_context_) );
        dim3 NT, NB;
        NT.x = nt;
        NT.y = 1;
        NT.z = 1;
        NB.x = (A_nrows+nt-1)/nt;
        NB.y = 1;
        NB.z = 1;
        w->nvals_ = u->nvals_;
        if (desc->debug()) {
          std::cout << w->nvals_ << " nnz in vector w\n";
          printDevice("w_val", w_val, A_nrows);
        }
        if (use_mask) {
          if (use_scmp)
            assignDenseDenseMaskedKernel<false, true, true><<<NB, NT>>>(
                w_val, w->nvals_, mask->dense_.d_val_, extractAdd(op),
                op.identity(), reinterpret_cast<Index*>(NULL), A_nrows);
          else
            assignDenseDenseMaskedKernel< true, true, true><<<NB, NT>>>(
                w_val, w->nvals_, mask->dense_.d_val_, extractAdd(op),
                op.identity(), reinterpret_cast<Index*>(NULL), A_nrows);
        }
        if (use_accum) {
          if (desc->debug()) {
            std::cout << "Doing eWiseAdd accumulate:\n";
            printDevice("w_val", w->d_val_, A_nrows);
          }
          eWiseAddDenseDenseKernel<<<NB, NT>>>(w->d_val_, NULL, extractAdd(op),
              w->d_val_, w_val, A_nrows);
        }
#endif

#ifdef BCSR
        // no mask support at this time
        std::cout << "cusparseBSRMV()" << std::endl;
//        std::cout << "use_accum: " << (use_accum?"true":"false") << std::endl; // false
//        std::cout << "use_mask: " << (use_mask?"true":"false") << std::endl; // true
//        std::cout << "use_scmp: " << (use_scmp?"true":"false") << std::endl; // true

        cusparseBSRMV(w, A, u, use_accum);

        dim3 NT, NB;
        NT.x = nt;
        NT.y = 1;
        NT.z = 1;
        NB.x = (A_nrows+nt-1)/nt;
        NB.y = 1;
        NB.z = 1;
        w->nvals_ = u->nvals_;

        assignDenseDenseMaskedKernel<false, true, true><<<NB, NT>>>(
        w_val, w->nvals_, mask->dense_.d_val_, extractAdd(op),
        op.identity(), reinterpret_cast<Index*>(NULL), A_nrows);
#endif

        if (desc->debug())
          printDevice("w_val", w->d_val_, A_nrows);
        // TODO(@ctcyang): add semiring inputs to CUB
        /*size_t temp_storage_bytes = 0;
        cub::DeviceSpmv::CsrMV(desc->d_temp_, temp_storage_bytes, A->d_csrVal_,
            A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
            A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());
        desc->resize( temp_storage_bytes, "temp" );
        cub::DeviceSpmv::CsrMV(desc->d_temp_, desc->d_temp_size_, A->d_csrVal_,
            A->d_csrRowPtr_, A->d_csrColInd_, u->d_val_, w->d_val_,
            A->nrows_, A->ncols_, A->nvals_, 1.f, op->identity());*/
      }
          w->need_update_ = true;
          return GrB_SUCCESS;

} // Info spmv

template <typename W, typename a, typename U>
void cusparseBSRMV(DenseVector<W>* w, const BlockMatrix32<a>* A, const DenseVector<U>* u, bool use_accum) { // w = α ∗ op ( A ) ∗ u + β ∗ w
    // Suppose that A is m x n sparse matrix represented by CSR format,
    // hx is a host vector of size n, and hy is also a host vector of size m.
    // m and n are not multiple of blockDim.
    // step 1: transform CSR to BSR with column-major order --> has been done in matrix build
    // metadata
    cudaStream_t streamId = 0;
    cusparseHandle_t cusparseHandle = 0; //cusparseHandle_t
    CudaSparseCheck(cusparseCreate(&cusparseHandle));
    CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));

    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseMatDescr_t bcsr_descr = 0;
    CudaSparseCheck(cusparseCreateMatDescr(&bcsr_descr));
    cusparseSetMatType(bcsr_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bcsr_descr,CUSPARSE_INDEX_BASE_ZERO);


    // step 2: allocate vector x and vector y large enough for bsrmv
    //cudaMalloc((void**)&x, sizeof(a)*(A->nblockrow_ * A->blocksize_));
    //cudaMalloc((void**)&y, sizeof(a)*(A->nblockrow_ * A->blocksize_));
    //cudaMemcpy(x, hx, sizeof(a) * A->ncols_, cudaMemcpyHostToDevice); // hx: size = ncols_ <-- u
    //cudaMemcpy(y, hy, sizeof(a) * A->nrows_, cudaMemcpyHostToDevice); // hy: size = nrows_

    float alpha, beta;
    if (use_accum) { alpha = 1.f; beta = 1.f;}
    else { alpha = 1.f; beta = 0.f; }

    // step 3: perform bsrmv
    // cusparseSbsrmv() : default spmv
    // cusparseSbsrxmv(): support mask, but A should be in bsrx format (need bsrEndPtr)
    printDevice("w_val", w->d_val_, A->nrows_);
    printDevice("u_val", w->d_val_, A->nrows_);
    cusparseSbsrmv(cusparseHandle, dirA, transA, A->nblockrow_, A->nblockrow_, A->nvals_, &alpha,
       bcsr_descr, A->d_bcsrVal_, A->d_bcsrRowPtr_, A->d_bcsrColInd_, A->blocksize_, u->d_val_, &beta, w->d_val_);
    printDevice("w_val", w->d_val_, A->nrows_);
    printDevice("u_val", u->d_val_, A->nrows_);
} //cusparseBSRMV


} // namespace backend
} // namespace graphblas




#endif //GRAPHBLAS_BACKEND_CUDA_BLOCK_MATRIX_32_SPMV_HPP_