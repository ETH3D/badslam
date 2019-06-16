// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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
#include <curand_kernel.h>

#include "libvis/cuda/cuda_buffer.cuh"
#include "libvis/cuda/cuda_matrix.cuh"
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/libvis.h"

namespace vis {

constexpr int kPatchMatchStereo_MatchMetric_SSD = 0;
constexpr int kPatchMatchStereo_MatchMetric_ZNCC = 1;
constexpr int kPatchMatchStereo_MatchMetric_Census = 2;

void InitPatchMatchCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    float max_normal_2d_length,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    float inv_min_depth,
    float inv_max_depth,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    CUDABuffer_<curandState>* random_states,
    CUDABuffer_<float>* lambda,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

void PatchMatchMutationStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    float max_normal_2d_length,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    float step_range,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

// void PatchMatchOptimizationStepCUDA(
//     cudaStream_t stream,
//     int match_metric,
//     int context_radius,
//     float max_normal_2d_length,
//     cudaTextureObject_t reference_unprojection_lookup,
//     const CUDABuffer_<u8>& reference_image,
//     cudaTextureObject_t reference_texture,
//     const CUDAMatrix3x4& stereo_tr_reference,
//     const PixelCornerProjector_& stereo_camera,
//     const cudaTextureObject_t stereo_image,
//     CUDABuffer_<float>* inv_depth_map,
//     CUDABuffer_<char2>* normals,
//     CUDABuffer_<float>* costs,
//     CUDABuffer_<curandState>* random_states,
//     CUDABuffer_<float>* lambda);

void PatchMatchPropagationStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    CUDABuffer_<curandState>* random_states,
    float second_best_min_distance_factor = 0,
    CUDABuffer_<float>* best_inv_depth_map = nullptr);

void PatchMatchDiscreteRefinementStepCUDA(
    cudaStream_t stream,
    int match_metric,
    int context_radius,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    int num_steps,
    float range_factor,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs);

void PatchMatchLeftRightConsistencyCheckCUDA(
    cudaStream_t stream,
    int context_radius,
    float lr_consistency_factor_threshold,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDAMatrix3x4& stereo_tr_reference,
    const PixelCornerProjector_& stereo_camera,
    const CUDABuffer_<float>& lr_consistency_inv_depth,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out);

void PatchMatchFilterOutliersCUDA(
    cudaStream_t stream,
    int context_radius,
    float min_inv_depth,
    float required_range_min_depth,
    float required_range_max_depth,
    cudaTextureObject_t reference_unprojection_lookup,
    const CUDABuffer_<u8>& reference_image,
    cudaTextureObject_t reference_texture,
    const CUDAMatrix3x4& stereo_tr_reference,
    const CUDAMatrix3x4& reference_tr_stereo,
    const PixelCornerProjector_& stereo_camera,
    const cudaTextureObject_t stereo_image,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<char2>* normals,
    CUDABuffer_<float>* costs,
    float cost_threshold,
    float epipolar_gradient_threshold,
    float min_cos_angle,
    CUDABuffer_<float>* second_best_costs,
    float second_best_min_cost_factor);

void MedianFilterDepthMap3x3CUDA(
    cudaStream_t stream,
    int context_radius,
    CUDABuffer_<float>* inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out,
    CUDABuffer_<float>* costs,
    CUDABuffer_<float>* costs_out,
    CUDABuffer_<float>* second_best_costs,
    CUDABuffer_<float>* second_best_costs_out);

void BilateralFilterCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out);

void FillHolesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& inv_depth_map,
    CUDABuffer_<float>* inv_depth_map_out);

}
