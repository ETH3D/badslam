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

#include "badslam/kernel_opt_geometry.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void UpdateSurfelNormalsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    const CUDABuffer<u8>& active_surfels) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // Accumulate normals by projecting the surfels into all active and covis-active keyframes.
  CallResetSurfelAccum0to3CUDAKernel(
      stream,
      surfels_size,
      surfels.ToCUDA(),
      active_surfels.ToCUDA());
  
  for (const shared_ptr<Keyframe>& keyframe : keyframes) {
    if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
      continue;
    }
    
    CallAccumulateSurfelNormalOptimizationCoeffsCUDAKernel(
        stream,
        CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, keyframe.get()),
        keyframe->global_R_frame_cuda(),
        active_surfels.ToCUDA());
  }
  
  // Solve for the normal updates.
  CallUpdateSurfelNormalCUDAKernel(
      stream,
      surfels_size,
      surfels.ToCUDA(),
      active_surfels.ToCUDA());
}


void OptimizeGeometryIterationCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    const CUDABuffer<u8>& active_surfels) {
  CHECK(use_depth_residuals || use_descriptor_residuals);
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // TODO: Instead of always checking active_surfels, try using an index
  //       list of all active surfels to only run on the active ones, might be faster
  
  // TODO: Can put the accum reset for the second step at the end of the update
  //       kernel of the first step. However, make sure to mention clearly that
  //       this has to be adapted if the order of the steps is changed.
  
  
  // --- Normals ---
  
  // Accumulate normals by projecting the surfels into all active and covis-active keyframes.
  CallResetSurfelAccum0to3CUDAKernel(
      stream,
      surfels_size,
      surfels.ToCUDA(),
      active_surfels.ToCUDA());
  
  for (const shared_ptr<Keyframe>& keyframe : keyframes) {
    if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
      continue;
    }
    
    CallAccumulateSurfelNormalOptimizationCoeffsCUDAKernel(
        stream,
        CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, keyframe.get()),
        keyframe->global_R_frame_cuda(),
        active_surfels.ToCUDA());
    CUDA_CHECK();
  }
  
  // Solve for the normal updates.
  CallUpdateSurfelNormalCUDAKernel(
      stream,
      /* kernel parameters */
      surfels_size,
      surfels.ToCUDA(),
      active_surfels.ToCUDA());
  CUDA_CHECK();
  
  
  if (!use_descriptor_residuals) {
    // --- Position ---
    
    // Reset accumulation fields (TODO: Do this after an iteration and assume they stay reset to reduce the number of kernel calls?)
    CallResetSurfelAccum0to1CUDAKernel(
        stream,
        surfels_size,
        surfels.ToCUDA(),
        active_surfels.ToCUDA());
    
    // Accumulate H and b for surfel positions by projecting the surfels into all keyframes.
    for (const shared_ptr<Keyframe>& keyframe : keyframes) {
      if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
        continue;
      }
      
      CallAccumulateSurfelPositionOptimizationCoeffsFromDepthResidualCUDAKernel(
          stream,
          CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, keyframe.get()),
          CreatePixelCenterUnprojector(depth_camera),
          CreateDepthToColorPixelCorner(depth_camera, color_camera),
          color_camera.parameters()[0],
          color_camera.parameters()[1],
          keyframe->color_texture(),
          active_surfels.ToCUDA());
    }
    
    // Solve for the surfel position updates.
    CallUpdateSurfelPositionCUDAKernel(
        stream,
        surfels_size,
        surfels.ToCUDA(),
        active_surfels.ToCUDA());
  } else {  // if (use_descriptor_residuals)
    // --- Position and descriptors (jointly) ---
    
    CallResetSurfelAccumCUDAKernel(
        stream,
        surfels_size,
        surfels.ToCUDA(),
        active_surfels.ToCUDA());
    
    for (const shared_ptr<Keyframe>& keyframe : keyframes) {
      if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
        continue;
      }
      
      AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel(
          stream,
          CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, keyframe.get()),
          CreatePixelCenterUnprojector(depth_camera),
          CreateDepthToColorPixelCorner(depth_camera, color_camera),
          CreatePixelCornerProjector(color_camera),
          keyframe->color_texture(),
          active_surfels.ToCUDA(),
          use_depth_residuals);
    }
    
    CallUpdateSurfelPositionAndDescriptorCUDAKernel(
        stream,
        surfels_size,
        surfels.ToCUDA(),
        active_surfels.ToCUDA());
  }
}

}
