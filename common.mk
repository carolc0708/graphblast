CUDA_ARCH = 35
ARCH = -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}
OPTIONS = -O3 -use_fast_math -w -std=c++11

MGPU_DIR = ext/moderngpu/include/
CUB_DIR = ext/cub/cub/
BOOST_DIR = /root/workspace/boost_1_58_0/
GRB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

BOOST_DEPS = /root/workspace/boost_1_58_0/stage/lib/
GRB_DEPS = ext/moderngpu/src/mgpucontext.cu \
           ext/moderngpu/src/mgpuutil.cpp

LIBS = -L${BOOST_DEPS} \
			 -lboost_program_options \
       -lcublas \
			 -lcusparse \
			 -lcurand
