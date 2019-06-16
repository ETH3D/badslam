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

void CallAccumulateIntrinsicsCoefficientsCUDAKernel(
    cudaStream_t stream,
    bool optimize_color_intrinsics,
    bool optimize_depth_intrinsics,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    const PixelCenterUnprojector& depth_center_unprojector,
    float color_fx,
    float color_fy,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u32>& observation_count,
    const CUDABuffer_<float>& depth_A,
    const CUDABuffer_<float>& depth_B,
    const CUDABuffer_<float>& depth_D,
    const CUDABuffer_<float>& depth_b1,
    const CUDABuffer_<float>& depth_b2,
    const CUDABuffer_<float>& color_H,
    const CUDABuffer_<float>& color_b);

void CallComputeIntrinsicsIntermediateMatricesCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<float>& A,
    const CUDABuffer_<float>& B,
    const CUDABuffer_<float>& D,
    const CUDABuffer_<float>& b1,
    const CUDABuffer_<float>& b2);

void CallSolveForPixelIntrinsicsUpdateCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<u32>& observation_count,
    const CUDABuffer_<float>& B,
    const CUDABuffer_<float>& D,
    const CUDABuffer_<float>& x1,
    const CUDABuffer_<float>& cfactor_buffer);

}
