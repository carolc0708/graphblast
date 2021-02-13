#define GRB_USE_CUDA
#define private public

#include <vector>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <map>

#include <cstdio>
#include <cstdlib>

//#include "graphblas/graphblas.hpp"
#include "graphblas/backend.hpp"
#include "graphblas/mmio.hpp"
#include "graphblas/types.hpp"
#include "graphblas/stddef.hpp"
#include "graphblas/util.hpp"
#include "graphblas/dimension.hpp"
#include "graphblas/descriptor.hpp"
#include "graphblas/vector.hpp"
#include "graphblas/matrix.hpp"
#include "graphblas/block_matrix.hpp" // should be at here
#include "graphblas/operations.hpp"

#define __GRB_BACKEND_HEADER <graphblas/backend/__GRB_BACKEND_ROOT/__GRB_BACKEND_ROOT.hpp>
#include __GRB_BACKEND_HEADER
#undef __GRB_BACKEND_HEADER

#include "test/test.hpp"

#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE dup_suite

//#include <boost/test/included/unit_test.hpp>
#include <boost/program_options.hpp>

void testBlockMatrixBuild(char const* mtx)
{
    std::vector<graphblas::Index> row_indices;
    std::vector<graphblas::Index> col_indices;
    std::vector<float> values;
    graphblas::Index nrows, ncols, nvals;
    graphblas::Info err;
    graphblas::Descriptor desc;
    char* dat_name;

    // Read in sparse matrix
    readMtx(mtx, &row_indices, &col_indices, &values, &nrows, &ncols, &nvals, 0, false, &dat_name);

    //std::cout << __GRB_BACKEND_ROOT << std::endl;

    // Initialize sparse matrix A
    BlockMatrix<float> A(nrows, ncols);
    err = A.build(&row_indices, &col_indices, &values, nvals, GrB_NULL, dat_name);
    //std::cout << err << std::endl;
    //A.print(false);
}

int main() {

    //testBlockMatrixBuild ("data/small/small.mtx");

    //testBlockMatrixBuild ("/root/workspace/data/topc-datasets/soc-orkut.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/soc-LiveJournal1.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/hollywood-2009.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/indochina-2004.mtx");

    //testBlockMatrixBuild ("/root/workspace/data/coAuthorsCiteseer.mtx");
    testBlockMatrixBuild ("/root/workspace/data/coAuthorsDBLP.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/cit-Patents.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/com-Orkut.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/Journals.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/G43.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/ship_003.mtx");

    //testBlockMatrixBuild ("/root/workspace/data/rgg_n_2_24_s0.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/road_usa.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/road_central.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/belgium_osm.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/roadNet-CA.mtx");
    //testBlockMatrixBuild ("/root/workspace/data/delaunay_n24.mtx");

    // csr
//    std::vector<int> ver {2997166, 4847571, 1139905, 7414866, 227320, 299067, 3774768, 3072441, 124,
//    1000, 121728, 16777216, 23947347, 14081816, 1441295, 1971281, 16777216};
//    std::vector<int> edg = {212698418, 68475391, 112751422, 191606827, 1628268, 1955352, 16518947,
//    234370166, 11944, 19980, 7964306, 265114400, 57708624, 33866826, 3099940, 5533214, 100663202};

    // bsr 32
//    std::vector<std::size_t> br = {93662, 151487, 35623, 231715, 7104, 9346, 117962, 96014, 4, 32,
//    3804, 524288, 748355, 440057, 45041, 61603, 524288};
//    std::vector<std::size_t> nb = {141651235, 35482774, 14954753, 3473463, 390958, 690040, 15768304,
//    157180673, 16, 1022, 92578, 120457723, 14698015, 17085353, 492817, 1019157, 7688118};

    // bsr 64
//    std::vector<std::size_t> br = {46831, 75744, 17812, 115858, 3552, 4673, 58981, 48007, 2, 16,
//    1902, 262144, 374178, 220029, 22521, 30802, 262144};
//    std::vector<std::size_t> nb = {116452367, 30708026, 9985446, 2001026, 364441, 646987, 15297781, 129124673,
//    4, 256, 46020, 37241387, 9517315, 12682228, 387949, 665060, 4225982};

//    for(int i =0; i<ver.size(); i++) {
//        std::cout << i << " ";
//        //std::cout << (ver[i] + 2 * edg[i]) * sizeof(int) /(1024.0 * 1024.0 * 1024.0) << std::endl;
//        std::cout << nb[i] * sizeof(float) * 64 * 64 / (1024.0*1024.0*1024.0) << ", ";
//        std::cout << nb[i] * sizeof(float) * 64 * 64 / (1024.0*1024.0) << ", ";
//        std::cout << nb[i] * sizeof(float) * 64 * 64 / (1024.0) << ", ";
//        std::cout << ((br[i] + nb[i]) * sizeof(int) + nb[i] * sizeof(double) * 64) / (1024.0*1024.0*1024.0) << ", ";
//        std::cout << ((br[i] + nb[i]) * sizeof(int) + nb[i] * sizeof(double) * 64) / (1024.0*1024.0) << ", ";
//        std::cout << ((br[i] + nb[i]) * sizeof(int) + nb[i] * sizeof(double) * 64) / (1024.0) << std::endl;
//    }



    return 0;
}
