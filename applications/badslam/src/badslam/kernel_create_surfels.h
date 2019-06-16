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

#include <cuda_runtime.h>
#include <libvis/cuda/cuda_buffer.cuh>
#include <libvis/libvis.h>

#include "badslam/surfel_projection.cuh"

namespace vis {

void CallCreateSurfelsForKeyframeCUDASerializingKernel(
    cudaStream_t stream,
    int sparse_surfel_cell_size,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<uchar4>& color_buffer,
    const CUDABuffer_<u32>& supporting_surfels,
    const CUDABuffer_<u8>& new_surfel_flag_vector);

void CallWriteNewSurfelIndexAndInitializeObservationsCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<u8>& new_surfel_flag_vector,
    const CUDABuffer_<u32>& new_surfel_indices,
    u16* observation_vector,
    u16* free_space_violation_vector,
    u32* new_surfel_index_list);

void CallCreateSurfelsForKeyframeCUDACreationAppendKernel(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    const CUDAMatrix3x4& global_T_frame,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    const CUDABuffer_<u16>& radius_buffer,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u8>& new_surfel_flag_vector,
    const CUDABuffer_<u32>& new_surfel_indices,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels);

void CallCountObservationsForNewSurfelsCUDAKernel(
    cudaStream_t stream,
    int new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    const DepthParameters& depth_params,
    const PixelCenterUnprojector& unprojector,
    const CUDABuffer_<u16>& new_surfel_depth_buffer,
    const CUDABuffer_<u16>& new_surfel_normals_buffer,
    const CUDAMatrix3x4& covis_T_frame,
    const PixelCornerProjector& projector,
    const CUDABuffer_<u16>& covis_depth_buffer,
    const CUDABuffer_<u16>& covis_normals_buffer);

void CallFilterNewSurfelsCUDAKernel(
    cudaStream_t stream,
    u16 min_observation_count,
    u32 new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    const CUDABuffer_<u8>& new_surfel_flag_vector);

u32 CreateSurfelsForKeyframeCUDA_CountNewSurfels(
    cudaStream_t stream,
    u32 pixel_count,
    void** new_surfels_temp_storage,
    usize* new_surfels_temp_storage_bytes,
    CUDABuffer_<u8>* new_surfel_flag_vector,
    CUDABuffer_<u32>* new_surfel_indices);

}
