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

#include "badslam/kernel_supporting_surfels.h"

#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <math_constants.h>

#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/kernels.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

template <int block_width, bool merge_surfels>
__global__ void DetermineSupportingSurfelsCUDAKernel(
    float cell_merge_dist_squared,
    float cos_surfel_merge_normal_threshold,
    SurfelProjectionParameters surfel_proj,
    SupportingSurfelBuffers supporting_surfels,
    CUDABuffer_<u32> deleted_count_buffer) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  int deleted = 0;
  
  SurfelProjectionResultXY r;
  if (SurfelProjectsToAssociatedPixel(surfel_index, surfel_proj, &r)) {
    // Set the supporting surfel entry only if it was previously empty
    r.px /= surfel_proj.depth_params.sparse_surfel_cell_size;
    r.py /= surfel_proj.depth_params.sparse_surfel_cell_size;
    for (int i = 0; i < kMergeBufferCount; ++ i) {
      u32 sup_index = atomicCAS(&supporting_surfels.b[i](r.py, r.px), kInvalidIndex, surfel_index);
      
      if (sup_index == kInvalidIndex) {
        break;
      } else if (merge_surfels) {
        // The entry was not replaced since another surfel was entered at this
        // pixel previously. Test whether the two surfels should be merged.
        float3 sup_normal = SurfelGetNormal(surfel_proj.surfels, sup_index);
        float3 this_normal = SurfelGetNormal(surfel_proj.surfels, surfel_index);
        
        if (Dot(sup_normal, this_normal) > cos_surfel_merge_normal_threshold) {
          float3 sup_position = SurfelGetPosition(surfel_proj.surfels, sup_index);
          float sup_radius_sq = SurfelGetRadiusSquared(surfel_proj.surfels, sup_index);
          float3 this_position = SurfelGetPosition(surfel_proj.surfels, surfel_index);
          float this_radius_sq = SurfelGetRadiusSquared(surfel_proj.surfels, surfel_index);
          
          float min_radius_sq = ::min(sup_radius_sq, this_radius_sq);
          if (SquaredDistance(sup_position, this_position) <
                  min_radius_sq * cell_merge_dist_squared) {
            surfel_proj.surfels(kSurfelX, surfel_index) = CUDART_NAN_F;
            deleted = 1;
          }
        }
      }
    }
  }
  
  // Update surfel_count if surfels merged
  if (merge_surfels) {
    typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceInt;
    __shared__ typename BlockReduceInt::TempStorage int_storage;
    u32 deleted_in_block = BlockReduceInt(int_storage).Sum(deleted);
    if (threadIdx.x == 0 && deleted_in_block > 0) {
      atomicAdd(&deleted_count_buffer(0, 0), static_cast<u32>(deleted_in_block));
    }
  }
}

void CallDetermineSupportingSurfelsCUDAKernel(
    cudaStream_t stream,
    bool merge_surfels,
    float cell_merge_dist_squared,
    float cos_surfel_merge_normal_threshold,
    SurfelProjectionParameters surfel_proj,
    SupportingSurfelBuffers supporting_surfels,
    CUDABuffer_<u32> deleted_count_buffer) {
  COMPILE_OPTION(merge_surfels,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          DetermineSupportingSurfelsCUDAKernel,
          1024,
          surfel_proj.surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _merge_surfels),
          /* kernel parameters */
          cell_merge_dist_squared,
          cos_surfel_merge_normal_threshold,
          surfel_proj,
          supporting_surfels,
          deleted_count_buffer));
}

}
