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


#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <libvis/libvis.h>

#include <libvis/cuda/cuda_buffer.cuh>

namespace vis {

// Adds a residual (given its Jacobian and "raw" (unweighted) residual value and
// its weight) to the matrix H and the vector b of the Gauss-Newton update
// equation H*x = b. Does nothing if valid == false. Stores the entries of H
// sequentially row by row (respectively column by column since the matrix is
// symmetric) while leaving out redundant entries. The required size is:
// size * (size + 1) / 2
template <int size, int block_width, int block_height>
__forceinline__ __device__ void AccumulateGaussNewtonHAndB(
    bool valid,
    float raw_residual,
    float residual_weight,
    float* jacobian,
    CUDABuffer_<float>& H_buffer,
    CUDABuffer_<float>& b_buffer,
    typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* float_storage) {
  typedef typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  
  // TODO: Would it be faster to use a few different shared memory buffers for the reduce operations to avoid some of the __syncthreads()?
  
  // jacobian.tranpose() * weight * jacobian:
  {
    float* write_ptr = H_buffer.address();
    #pragma unroll
    for (int row = 0; row < size; ++ row) {
      #pragma unroll
      for (int col = row; col < size; ++ col) {
        const float jacobian_sq_i = residual_weight * jacobian[row] * jacobian[col];
        __syncthreads();  // Required before re-use of shared memory.
        const float block_sum =
            BlockReduceFloat(*float_storage).Sum(valid ? jacobian_sq_i : 0.f);
        if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
          atomicAdd(write_ptr, block_sum);
          ++ write_ptr;
        }
      }
    }
  }
  
  const float weighted_raw_residual = residual_weight * raw_residual;
  
  // jacobian.transpose() * weight * point_residual:
  {
    #pragma unroll
    for (int i = 0; i < size; ++ i) {
      const float b_i = weighted_raw_residual * jacobian[i];
      __syncthreads();  // Required before re-use of shared memory.
      const float block_sum =
          BlockReduceFloat(*float_storage).Sum(valid ? b_i : 0);
      if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
        atomicAdd(&b_buffer(0, i), block_sum);
      }
    }
  }
}

}
