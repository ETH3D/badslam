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

void CallAccumulatePoseEstimationCoeffsCUDAKernel(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCenterProjector& color_center_projector,
    const PixelCornerProjector& color_corner_projector,
    const PixelCenterUnprojector& depth_unprojector,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer);

void CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterProjector& color_center_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer,
    CUDABuffer_<float>* debug_residual_image);

void CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY(
    cudaStream_t stream,
    bool debug,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterProjector& color_center_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer,
    const CUDABuffer_<float>& H_buffer,
    const CUDABuffer_<float>& b_buffer,
    CUDABuffer_<float>* debug_residual_image);

void CallComputeCostAndResidualCountFromImagesCUDAKernel_GradMag(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer);

void ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PixelCornerProjector& depth_projector,
    const PixelCenterUnprojector& depth_unprojector,
    float baseline_fx,
    const DepthToColorPixelCorner& depth_to_color,
    float threshold_factor,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer_<float>& surfel_depth,
    const CUDABuffer_<u16>& surfel_normals,
    const CUDABuffer_<u8>& surfel_color,
    const CUDABuffer_<float>& frame_depth,
    const CUDABuffer_<u16>& frame_normals,
    cudaTextureObject_t frame_color,
    const CUDABuffer_<u32>& residual_count_buffer,
    const CUDABuffer_<float>& residual_buffer);

}
