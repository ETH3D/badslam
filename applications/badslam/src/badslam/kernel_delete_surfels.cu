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
#include <math_constants.h>

#include <libvis/cuda/cuda_auto_tuner.h>
#include "badslam/cuda_util.cuh"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

template <bool update_radii>
__global__ void ResetSurfelAccumForSurfelDeletionAndRadiusUpdateCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
    if (update_radii) {
      surfels(kSurfelAccum2, surfel_index) = CUDART_INF_F;
    }
  }
}

void CallResetSurfelAccumForSurfelDeletionAndRadiusUpdateCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    bool update_radii) {
  if (update_radii) {
    CUDA_AUTO_TUNE_1D(
        ResetSurfelAccumForSurfelDeletionAndRadiusUpdateCUDAKernel<true>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        surfels_size,
        surfels);
  } else {
    CUDA_AUTO_TUNE_1D(
        ResetSurfelAccumForSurfelDeletionAndRadiusUpdateCUDAKernel<false>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        surfels_size,
        surfels);
  }
  CUDA_CHECK();
}


template <bool update_radii>
__global__ void CountObservationsAndFreeSpaceViolationsCUDAKernel(
    SurfelProjectionParameters s,
    CUDABuffer_<u16> radius_buffer) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  SurfelProjectionResultXYFreeSpace r;
  if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
    // TODO: Use accum buffers as u16 buffers here for probably slightly more speed?
    s.surfels(kSurfelAccum0, surfel_index) += 1.f;
    
    if (update_radii) {
      float stored_radius_squared = s.surfels(kSurfelAccum2, surfel_index);
      float measured_radius_squared = __half2float(__ushort_as_half(radius_buffer(r.py, r.px)));
      s.surfels(kSurfelAccum2, surfel_index) = min(stored_radius_squared, measured_radius_squared);
    }
  } else if (surfel_index < s.surfels_size && r.is_free_space_violation) {
    s.surfels(kSurfelAccum1, surfel_index) += 1.f;
  }
}

void CallCountObservationsAndFreeSpaceViolationsCUDAKernel(
    cudaStream_t stream,
    const SurfelProjectionParameters& s,
    const CUDABuffer_<u16>& radius_buffer,
    bool update_radii) {
  if (update_radii) {
    CUDA_AUTO_TUNE_1D(
        CountObservationsAndFreeSpaceViolationsCUDAKernel<true>,
        1024,
        s.surfels_size,
        0, stream,
        /* kernel parameters */
        s,
        radius_buffer);
  } else {
    CUDA_AUTO_TUNE_1D(
        CountObservationsAndFreeSpaceViolationsCUDAKernel<false>,
        1024,
        s.surfels_size,
        0, stream,
        /* kernel parameters */
        s,
        radius_buffer);
  }
  CUDA_CHECK();
}


template <int block_width, bool update_radii>
__global__ void MarkDeletedSurfelsCUDAKernel(
    int min_observation_count,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u32> deleted_count_buffer) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  int deleted = 0;
  if (surfel_index < surfels_size) {
    float observation_count = surfels(kSurfelAccum0, surfel_index);
    if (observation_count < min_observation_count ||
        surfels(kSurfelAccum1, surfel_index) > observation_count) {
      // Mark the surfel as deleted.
      if (__float_as_int(surfels(kSurfelX, surfel_index)) != 0x7fffffff) {
        surfels(kSurfelX, surfel_index) = CUDART_NAN_F;
        deleted = 1;
      }
    } else if (update_radii) {
      SurfelSetRadiusSquared(&surfels, surfel_index, surfels(kSurfelAccum2, surfel_index));
    }
    
//     // DEBUG
//     u8 g = 255.99f * ::min(1.f, (observation_count / 30.f));
//     u8 r = 255.99f * ::min(1.f, (surfels(kSurfelAccum1, surfel_index) / 30.f));
//     SurfelSetColor(&surfels, surfel_index, r, g, 0);
  }
  
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceInt;
  __shared__ typename BlockReduceInt::TempStorage int_storage;
  u32 deleted_in_block = BlockReduceInt(int_storage).Sum(deleted);
  if (threadIdx.x == 0 && deleted_in_block > 0) {
    atomicAdd(&deleted_count_buffer(0, 0), static_cast<u32>(deleted_in_block));
  }
}

void CallMarkDeletedSurfelsCUDAKernel(
    cudaStream_t stream,
    int min_observation_count,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    CUDABuffer_<u32>* deleted_count_buffer,
    bool update_radii) {
  if (update_radii) {
    CUDA_AUTO_TUNE_1D_TEMPLATED(
        MarkDeletedSurfelsCUDAKernel,
        1024,
        surfels_size,
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, true),
        /* kernel parameters */
        min_observation_count,
        surfels_size,
        surfels,
        *deleted_count_buffer);
  } else {
    CUDA_AUTO_TUNE_1D_TEMPLATED(
        MarkDeletedSurfelsCUDAKernel,
        1024,
        surfels_size,
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, false),
        /* kernel parameters */
        min_observation_count,
        surfels_size,
        surfels,
        *deleted_count_buffer);
  }
  CUDA_CHECK();
}

}
