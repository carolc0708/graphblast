#define GRB_USE_CUDA
#define private public

#include <iostream>
#include <algorithm>
#include <string>

#include <cstdio>
#include <cstdlib>

// #include <cuda_profiler_api.h>

#include <boost/program_options.hpp>

#include "graphblas/graphblas.hpp" // operation.hpp included
#include "graphblas/block_matrix.hpp"
#include "graphblas/block_matrix_operations.hpp"
#include "graphblas/algorithm/bfs.hpp" // for bfsCpu
#include "graphblas/algorithm/block_matrix_32_bfs.hpp"
#include "test/test.hpp"

bool debug_;
bool memory_;

// execute only gpu ver this time
int main(int argc, char** argv) {
  std::vector<graphblas::Index> row_indices;
  std::vector<graphblas::Index> col_indices;
  std::vector<float> values;
  graphblas::Index nrows, ncols, nvals;

  // Parse arguments
  bool debug;
  bool transpose;
  bool mtxinfo;
  int  directed;
  int  niter;
  int  source;
  char* dat_name;
  po::variables_map vm;

  // Read in sparse matrix
  if (argc < 2) {
    fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
    exit(1);
  } else {
    parseArgs(argc, argv, &vm);
    debug     = vm["debug"    ].as<bool>();
    transpose = vm["transpose"].as<bool>();
    mtxinfo   = vm["mtxinfo"  ].as<bool>();
    directed  = vm["directed" ].as<int>();
    niter     = vm["niter"    ].as<int>();
    source    = vm["source"   ].as<int>();

    // This is an imperfect solution, because this should happen in
    // desc.loadArgs(vm) instead of application code!
    // TODO(@ctcyang): fix this
    readMtx(argv[argc-1], &row_indices, &col_indices, &values, &nrows, &ncols,
        &nvals, directed, mtxinfo, &dat_name);
  }

  // Descriptor desc
  graphblas::Descriptor desc;
  CHECK(desc.loadArgs(vm));
  if (transpose)
    CHECK(desc.toggle(graphblas::GrB_INP1));


  // bcsr matrix
  graphblas::BlockMatrix<float> ba(nrows, ncols);
  CHECK(ba.build(&row_indices, &col_indices, &values, nvals, GrB_NULL,
      dat_name));
  CHECK(ba.nrows(&nrows));
  CHECK(ba.ncols(&ncols));
  CHECK(ba.nvals(&nvals));
  if (debug) CHECK(ba.print());

  // Vector v
  graphblas::Vector<float> v(nrows);

  // Warmup
  CpuTimer warmup;
  warmup.Start();
  graphblas::algorithm::bfs(&v, &ba, source, &desc); // <--- here
  warmup.Stop();

  // verify warmup
  std::vector<float> h_bfs_gpu;
  CHECK(v.extractTuples(&h_bfs_gpu, &nrows));
  //VERIFY_LIST(h_bfs_cpu, h_bfs_gpu, nrows);

  // Benchmark
//  graphblas::Vector<float> y(nrows);
//  CpuTimer vxm_gpu;
//  // cudaProfilerStart();
//  vxm_gpu.Start();
//  float tight = 0.f;
//  float val;
//  for (int i = 0; i < niter; i++) {
//    val = graphblas::algorithm::bfs(&y, &ba, source, &desc); // <-- here
//    tight += val;
//  }
//  // cudaProfilerStop();
//  vxm_gpu.Stop();

  float flop = 0;
  std::cout << "warmup, " << warmup.ElapsedMillis() << ", " <<
    flop/warmup.ElapsedMillis()/1000000.0 << "\n";
  //float elapsed_vxm = vxm_gpu.ElapsedMillis();
  //std::cout << "tight, " << tight/niter << "\n";
  //std::cout << "vxm, " << elapsed_vxm/niter << "\n";

//  if (niter) {
//    std::vector<float> h_bfs_gpu2;
//    CHECK(y.extractTuples(&h_bfs_gpu2, &nrows));
//    VERIFY_LIST(h_bfs_cpu, h_bfs_gpu2, nrows);
//  }

  return 0;
}
