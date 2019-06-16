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
#include <libvis/opengl.h>

#include "badslam/cuda_matrix.cuh"
#include "badslam/surfel_projection.cuh"

namespace vis {

// Applies a bilateral filter to a depth image, and removes depth values larger
// than max_depth.
void BilateralFilteringAndDepthCutoffCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    u16 max_depth,
    float raw_to_float_depth,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth);

// Estimates normal vectors for pixels of a depth image using centered finite
// differences. If not all 4-neighbors have a depth value for a given pixel, it
// is discarded (i.e., removed from the depth image).
void ComputeNormalsCUDA(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth,
    CUDABuffer_<u16>* normals_buffer);

// Estimates radii for depth measurements. The radius is defined as the minimum
// distance of the 3D point represented by a pixel to one of the 3D points from
// its 4-neighborhood in the image. If not all 4-neighbors have a depth value,
// the center pixel is discarded (i.e., removed from the depth image).
void ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    float raw_to_float_depth,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<u16>* radius_buffer,
    CUDABuffer_<u16>* out_depth);

// Computes the minimum and maximum depth value of the given depth image.
void ComputeMinMaxDepthCUDA(
    cudaStream_t stream,
    const CUDABuffer_<u16>& depth_buffer,
    float raw_to_float_depth,
    const CUDABuffer_<float>& init_buffer,
    CUDABuffer_<float>* result_buffer,
    float* keyframe_min_depth,
    float* keyframe_max_depth);
}
