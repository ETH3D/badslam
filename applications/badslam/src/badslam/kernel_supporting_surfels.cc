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

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void DetermineSupportingSurfelsCUDAImpl(
    bool merge_surfels,
    float merge_dist_factor,
    u32* surfel_count,
    cudaStream_t stream,
    const PinholeCamera4f& camera,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABuffer<u32>** supporting_surfels,
    CUDABufferPtr<u32>* deleted_count_buffer) {
  CUDA_CHECK();
  
  SupportingSurfelBuffers supporting_surfel_buffers;
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfel_buffers.b[i] = supporting_surfels[i]->ToCUDA();
  }
  
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels[i]->Clear(kInvalidIndex, stream);
  }
  
  if (surfels_size == 0) {
    return;
  }
  
  if (merge_surfels) {
    if (!*deleted_count_buffer) {
      deleted_count_buffer->reset(new CUDABuffer<u32>(1, 1));
    }
    (*deleted_count_buffer)->Clear(0, stream);
    
    float cell_merge_dist_squared =
        depth_params.sparse_surfel_cell_size * depth_params.sparse_surfel_cell_size *
        merge_dist_factor * merge_dist_factor;
    
    // This is set to: cosf(M_PI / 180.f * kSurfelMergeNormalThreshold);
    float cos_surfel_merge_normal_threshold =
        cos_normal_compatibility_threshold;
    
    CallDetermineSupportingSurfelsCUDAKernel(
        stream,
        merge_surfels,
        cell_merge_dist_squared,
        cos_surfel_merge_normal_threshold,
        CreateSurfelProjectionParameters(
            camera, depth_params, surfels_size, *surfels, depth_buffer, normals_buffer, frame_T_global),
        supporting_surfel_buffers,
        (*deleted_count_buffer)->ToCUDA());
    
    // Update surfel_count
    u32 deleted_count;
    (*deleted_count_buffer)->DownloadAsync(stream, &deleted_count);
    cudaStreamSynchronize(stream);
    *surfel_count -= deleted_count;
  } else {
    CallDetermineSupportingSurfelsCUDAKernel(
        stream,
        merge_surfels,
        /*cell_merge_dist_squared*/ 0,
        /*cos_surfel_merge_normal_threshold*/ 0,
        CreateSurfelProjectionParameters(
            camera, depth_params, surfels_size, *surfels, depth_buffer, normals_buffer, frame_T_global),
        supporting_surfel_buffers,
        CUDABuffer_<u32>());
  }
  CUDA_CHECK();
}

void DetermineSupportingSurfelsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& camera,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABuffer<u32>** supporting_surfels) {
  DetermineSupportingSurfelsCUDAImpl(
      /*merge_surfels*/ false,
      /*merge_dist_factor*/ 0,
      nullptr,
      stream,
      camera,
      frame_T_global,
      depth_params,
      depth_buffer,
      normals_buffer,
      surfels_size,
      surfels,
      supporting_surfels,
      nullptr);
}

void DetermineSupportingSurfelsAndMergeSurfelsCUDA(
    cudaStream_t stream,
    float merge_dist_factor,
    const PinholeCamera4f& camera,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABuffer<u32>** supporting_surfels,
    u32* surfel_count,
    CUDABufferPtr<u32>* deleted_count_buffer) {
  DetermineSupportingSurfelsCUDAImpl(
      /*merge_surfels*/ true,
      merge_dist_factor,
      surfel_count,
      stream,
      camera,
      frame_T_global,
      depth_params,
      depth_buffer,
      normals_buffer,
      surfels_size,
      surfels,
      supporting_surfels,
      deleted_count_buffer);
}

}
