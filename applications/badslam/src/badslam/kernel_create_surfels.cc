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

#include "badslam/kernel_create_surfels.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void CreateSurfelsForKeyframeCUDA(
    cudaStream_t stream,
    int sparse_surfel_cell_size,
    bool filter_new_surfels,
    int min_observation_count,
    int keyframe_id,
    const vector<shared_ptr<Keyframe>>& keyframes,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const CUDAMatrix3x4& global_T_frame,
    const CUDAMatrix3x4& frame_T_global,
    const vector<CUDAMatrix3x4>& covis_T_frame,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    const CUDABuffer<u16>& radius_buffer,
    const CUDABuffer<uchar4>& color_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer<u32>** supporting_surfels,
    void** new_surfels_temp_storage,
    usize* new_surfels_temp_storage_bytes,
    CUDABuffer<u8>* new_surfel_flag_vector,
    CUDABuffer<u32>* new_surfel_indices,
    u32 surfels_size,
    u32 /*surfel_count*/,
    u32* new_surfel_count,
    CUDABuffer<float>* surfels) {
  CUDA_CHECK();
  
  // The first kernel marks in a sequential (non-pitched) vector whether a new
  // surfel is created for the corresponding pixel or not.
  // TODO regarding using supporting_surfels[0] only:
  //      Here we only check whether any surfel is there, so only use the first buffer.
  //      This means the other buffers don't need to be computed at all.
  //      However, alternatively the same check as when merging surfels should be used
  //      which could still create a surfel if there is another surfel within the same pixel.
  //      In this case, all buffers would be needed.
  // ATTENTION: This modifies supporting_surfels[0], but it does not update it
  //            properly.
  CallCreateSurfelsForKeyframeCUDASerializingKernel(
      stream,
      sparse_surfel_cell_size,
      depth_buffer.ToCUDA(),
      color_buffer.ToCUDA(),
      supporting_surfels[0]->ToCUDA(),
      new_surfel_flag_vector->ToCUDA());
  
  *new_surfel_count = CreateSurfelsForKeyframeCUDA_CountNewSurfels(
      stream,
      depth_buffer.width() * depth_buffer.height(),
      new_surfels_temp_storage,
      new_surfels_temp_storage_bytes,
      &new_surfel_flag_vector->ToCUDA(),
      &new_surfel_indices->ToCUDA());
  if (*new_surfel_count == 0) {
    return;
  }
  
  // If desired, filter out new surfels which are not observed by the minimum
  // observation count or which have more free-space violations than valid
  // observations.
  if (filter_new_surfels) {
    // Write list of image pixel indices with new surfels. Initialize
    // observation count to 1, free-space violation count to 0.
    u16* observation_vector = reinterpret_cast<u16*>(reinterpret_cast<u8*>(surfels->ToCUDA().address()) + kSurfelAccum0 * surfels->ToCUDA().pitch());
    u16* free_space_violation_vector = reinterpret_cast<u16*>(reinterpret_cast<u8*>(surfels->ToCUDA().address()) + kSurfelAccum1 * surfels->ToCUDA().pitch());
    u32* new_surfel_index_list = reinterpret_cast<u32*>(reinterpret_cast<u8*>(surfels->ToCUDA().address()) + kSurfelAccum2 * surfels->ToCUDA().pitch());
    CallWriteNewSurfelIndexAndInitializeObservationsCUDAKernel(
        stream,
        depth_buffer.width() * depth_buffer.height(),
        new_surfel_flag_vector->ToCUDA(),
        new_surfel_indices->ToCUDA(),
        observation_vector,
        free_space_violation_vector,
        new_surfel_index_list);
    
    // For all keyframes having co-visibility with this keyframe, accumulate
    // observations.
    const shared_ptr<Keyframe>& keyframe = keyframes[keyframe_id];
    for (usize i = 0; i < keyframe->co_visibility_list().size(); ++ i) {
      const shared_ptr<Keyframe>& co_visible_keyframe = keyframes[keyframe->co_visibility_list()[i]];
      
      CallCountObservationsForNewSurfelsCUDAKernel(
          stream,
          *new_surfel_count,
          new_surfel_index_list,
          observation_vector,
          free_space_violation_vector,
          depth_params,
          CreatePixelCenterUnprojector(depth_camera),
          depth_buffer.ToCUDA(),
          normals_buffer.ToCUDA(),
          covis_T_frame[i],
          CreatePixelCornerProjector(depth_camera),
          co_visible_keyframe->depth_buffer().ToCUDA(),
          co_visible_keyframe->normals_buffer().ToCUDA());
    }
    
    // Remove filtered surfels.
    CallFilterNewSurfelsCUDAKernel(
        stream,
        min_observation_count,
        *new_surfel_count,
        new_surfel_index_list,
        observation_vector,
        free_space_violation_vector,
        new_surfel_flag_vector->ToCUDA());
    
    // Repeat new surfel counting to adapt the indices and new_surfel_count.
    *new_surfel_count = CreateSurfelsForKeyframeCUDA_CountNewSurfels(
        stream,
        depth_buffer.width() * depth_buffer.height(),
        new_surfels_temp_storage,
        new_surfels_temp_storage_bytes,
        &new_surfel_flag_vector->ToCUDA(),
        &new_surfel_indices->ToCUDA());
    if (*new_surfel_count == 0) {
      return;
    }
  }
  
  // Append the new surfels at the end of the surfel list.
  if (surfels_size + *new_surfel_count > surfels->width()) {
    LOG(ERROR) << "Maximum surfel count exceeded! Retry with higher --max_surfel_count parameter value.";
    return;
  }
  
  CallCreateSurfelsForKeyframeCUDACreationAppendKernel(
      stream,
      CreatePixelCenterUnprojector(depth_camera),
      CreateDepthToColorPixelCorner(depth_camera, color_camera),
      CreatePixelCornerProjector(color_camera),
      global_T_frame,
      frame_T_global,
      depth_params,
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      radius_buffer.ToCUDA(),
      color_texture,
      new_surfel_flag_vector->ToCUDA(),
      new_surfel_indices->ToCUDA(),
      surfels_size,
      surfels->ToCUDA());
}

}
