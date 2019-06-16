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

void CallResetSurfelAccum0to3CUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels);

void CallAccumulateSurfelNormalOptimizationCoeffsCUDAKernel(
    cudaStream_t stream,
    SurfelProjectionParameters s,
    CUDAMatrix3x3 global_R_frame,
    CUDABuffer_<u8> active_surfels);

void CallUpdateSurfelNormalCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels);

void CallResetSurfelAccum0to1CUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels);

void CallAccumulateSurfelPositionOptimizationCoeffsFromDepthResidualCUDAKernel(
    cudaStream_t stream,
    SurfelProjectionParameters s,
    PixelCenterUnprojector depth_unprojector,
    DepthToColorPixelCorner depth_to_color,
    float color_fx,
    float color_fy,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u8> active_surfels);

void CallUpdateSurfelPositionCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels);

void CallResetSurfelAccumCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels);

void AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel(
    cudaStream_t stream,
    const SurfelProjectionParameters& s,
    const PixelCenterUnprojector& depth_unprojector,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u8>& active_surfels,
    bool use_depth_residuals);

void CallUpdateSurfelPositionAndDescriptorCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels);

}
