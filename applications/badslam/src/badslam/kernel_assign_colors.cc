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

#include "badslam/kernel_assign_colors.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void AssignColorsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    CUDABuffer<float>* surfels) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // Reset accumulation fields (TODO: Do this after an iteration and assume they stay reset to reduce the number of kernel calls?)
  CallResetSurfelForColorAssignmentKernel(
      stream,
      surfels_size,
      surfels->ToCUDA());
  
  // Accumulate color observations
  for (const shared_ptr<Keyframe>& keyframe : keyframes) {
    if (!keyframe) {
      continue;
    }
    
    CallAccumulateColorObservationsCUDAKernel(
        stream,
        surfels_size,
        CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, *surfels, keyframe.get()),
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        keyframe->color_texture());
  }
  
  // Assign colors
  CallAssignColorsCUDAKernel(
      stream,
      surfels_size,
      surfels->ToCUDA());
}

void AssignDescriptorColorsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    CUDABuffer<float>* surfels) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // Reset accumulation fields (TODO: Do this after an iteration and assume they stay reset to reduce the number of kernel calls?)
  CallResetSurfelForDescriptorColorAssignmentKernel(
      stream,
      surfels_size,
      surfels->ToCUDA());
  
  // Accumulate color observations
  for (const shared_ptr<Keyframe>& keyframe : keyframes) {
    if (!keyframe) {
      continue;
    }
    
    CallAccumulateDescriptorColorObservationsCUDAKernel(
        stream,
        surfels_size,
        CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, *surfels, keyframe.get()),
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        CreatePixelCornerProjector(color_camera),
        keyframe->color_texture());
  }
  
  // Assign colors
  CallAssignDescriptorColorsCUDAKernel(
      stream,
      surfels_size,
      surfels->ToCUDA());
}

}
