// Copyright 2019 ETH Zürich, Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_buffer.cuh>

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"

namespace vis {

template <int block_width>
__global__ void CountValidSurfelsCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u32> count_buffer) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  int is_valid = 0;
  if (surfel_index < surfels_size) {
    is_valid = __float_as_int(surfels(kSurfelX, surfel_index)) != 0x7fffffff;
  }
  
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage int_storage;
  u32 count_in_block = BlockReduceInt(int_storage).Sum(is_valid);
  if (threadIdx.x == 0 && count_in_block > 0) {
    atomicAdd(&count_buffer(0, 0), static_cast<u32>(count_in_block));
  }
}

void CallCountValidSurfelsCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u32>& count_buffer) {
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      CountValidSurfelsCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      surfels_size,
      surfels,
      count_buffer);
}

}
