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
#include <libvis/cuda/cuda_auto_tuner.h>

#include "badslam/cuda_util.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

__global__ void SetSurfelInactiveKernel(
    u32 surfels_size,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    // TODO: Operate on int (4 surfels) instead of u8 for improved speed?
    active_surfels(0, surfel_index) &= ~kSurfelActiveFlag;
  }
}

void CallSetSurfelInactiveKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<u8>& active_surfels) {
  CUDA_AUTO_TUNE_1D(
      SetSurfelInactiveKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      active_surfels);
  CUDA_CHECK();
}


__global__ void DetermineActiveSurfelsKernel(
    SurfelProjectionParameters s,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < s.surfels_size) {
    if (active_surfels(0, surfel_index) & kSurfelActiveFlag) {
      // Nothing to do, the surfel is already set to active.
      return;
    }
    
    SurfelProjectionResultXY r;
    if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
      active_surfels(0, surfel_index) = kSurfelActiveFlag;
    }
  }
}

void CallDetermineActiveSurfelsKernel(
    cudaStream_t stream,
    const SurfelProjectionParameters& s,
    const CUDABuffer_<u8>& active_surfels) {
  CUDA_AUTO_TUNE_1D(
      DetermineActiveSurfelsKernel,
      1024,
      s.surfels_size,
      0, stream,
      /* kernel parameters */
      s,
      active_surfels);
  CUDA_CHECK();
}

}
