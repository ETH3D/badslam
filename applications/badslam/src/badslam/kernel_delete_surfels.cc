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

#include "badslam/kernel_delete_surfels.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void DeleteSurfelsAndUpdateRadiiCUDAImpl(
    bool update_radii,
    cudaStream_t stream,
    int min_observation_count,
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32* surfel_count,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABufferPtr<u32>* deleted_count_buffer) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // Reset accumulation fields (TODO: Do this after an iteration and assume they stay reset to reduce the number of kernel calls?)
  CallResetSurfelAccumForSurfelDeletionAndRadiusUpdateCUDAKernel(
      stream,
      surfels_size,
      surfels->ToCUDA(),
      update_radii);
  
  if (!*deleted_count_buffer) {
    deleted_count_buffer->reset(new CUDABuffer<u32>(1, 1));
  }
  (*deleted_count_buffer)->Clear(0, stream);
  
  // Count observations and free-space violations (of frames that have been
  // active, and frames having co-visibility with them)
  for (const shared_ptr<Keyframe>& keyframe : keyframes) {
    if (!keyframe) {
      continue;
    }
    
    CallCountObservationsAndFreeSpaceViolationsCUDAKernel(
        stream,
        CreateSurfelProjectionParameters(camera, depth_params, surfels_size, *surfels, keyframe.get()),
        keyframe->radius_buffer().ToCUDA(),
        update_radii);
  }
  
  // Delete surfels with less than min_observation_count observations, or
  // more free-space violations than valid observations.
  CallMarkDeletedSurfelsCUDAKernel(
      stream,
      min_observation_count,
      surfels_size,
      surfels->ToCUDA(),
      &(*deleted_count_buffer)->ToCUDA(),
      update_radii);
  
  // Update surfel_count
  u32 deleted_count;
  (*deleted_count_buffer)->DownloadAsync(stream, &deleted_count);
  cudaStreamSynchronize(stream);
  *surfel_count -= deleted_count;
}

void DeleteSurfelsAndUpdateRadiiCUDA(
    cudaStream_t stream,
    int min_observation_count,
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32* surfel_count,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABufferPtr<u32>* deleted_count_buffer) {
  DeleteSurfelsAndUpdateRadiiCUDAImpl(
      /*update_radii*/ true,
      stream,
      min_observation_count,
      camera,
      depth_params,
      keyframes,
      surfel_count,
      surfels_size,
      surfels,
      deleted_count_buffer);
}

// void DeleteSurfelsCUDA(
//     cudaStream_t stream,
//     int min_observation_count,
//     const PinholeCamera4f& camera,
//     const DepthParameters& depth_params,
//     const vector<shared_ptr<Keyframe>>& keyframes,
//     u32* surfel_count,
//     u32 surfels_size,
//     CUDABuffer<float>* surfels) {
//   DeleteSurfelsAndUpdateRadiiCUDAImpl(
//       /*update_radii*/ false,
//       stream,
//       min_observation_count,
//       camera,
//       depth_params,
//       keyframes,
//       surfel_count,
//       surfels_size,
//       surfels,
//       deleted_count_buffer);
// }

}
