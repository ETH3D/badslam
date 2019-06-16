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
#include <libvis/camera.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/libvis.h>
#include <libvis/opengl.h>
#include <libvis/sophus.h>

#include "badslam/cuda_matrix.cuh"
#include "badslam/kernels.cuh"
#include "badslam/surfel_projection.cuh"

namespace vis {

class Keyframe;


struct PoseEstimationHelperBuffers {
  __host__ PoseEstimationHelperBuffers()
      : residual_count_buffer(1, 1),
        residual_buffer(1, 1),
        H_buffer(1, (6 * (6 + 1)) / 2),
        b_buffer(1, 6) {}
  
  CUDABuffer<u32> residual_count_buffer;
  CUDABuffer<float> residual_buffer;
  CUDABuffer<float> H_buffer;
  CUDABuffer<float> b_buffer;
};

struct IntrinsicsOptimizationHelperBuffers {
  __host__ IntrinsicsOptimizationHelperBuffers(
      int pixel_count,
      int sparse_pixel_count,
      int a_rows)
      : observation_count(1, sparse_pixel_count),
        depth_A(1, a_rows * (a_rows + 1) / 2),
        depth_B(a_rows, pixel_count),
        depth_D(1, pixel_count),
        depth_b1(1, a_rows),
        depth_b2(1, pixel_count),
        color_H(1, (4 * (4 + 1)) / 2),
        color_b(1, 4) {}
  
  // Observation count for each pixel in the sparsified image
  CUDABuffer<u32> observation_count;
  // One half (including the diagonal) of the 3x3 submatrix A of the approximate Hessian matrix in Gauss-Newton
  CUDABuffer<float> depth_A;
  // Top-right off-diagonal block (transpose of bottom-left block)
  CUDABuffer<float> depth_B;
  // Bottom-right block on the diagonal
  CUDABuffer<float> depth_D;
  // Right hand side vector corresponding to the rows of A
  CUDABuffer<float> depth_b1;
  // Right hand side vector corresponding to the rows of D
  CUDABuffer<float> depth_b2;
  
  CUDABuffer<float> color_H;
  CUDABuffer<float> color_b;
};


// ### Public functions ###

void DetermineSupportingSurfelsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& camera,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABuffer<u32>** supporting_surfels);

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
    CUDABufferPtr<u32>* deleted_count_buffer);

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
    u32 surfel_count,
    u32* new_surfel_count,
    CUDABuffer<float>* surfels);

void UpdateVisualizationBuffersCUDA(
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer_resource,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    bool visualize_normals,
    bool visualize_descriptors,
    bool visualize_radii);

void AccumulatePoseEstimationCoeffsCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    const CUDAMatrix3x4& frame_T_global_estimate,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    bool debug,
    u32* residual_count,
    float* residual_sum,
    float* H,
    float* b,
    PoseEstimationHelperBuffers* helper_buffers);

void DebugSetBufferToColorTexture(
    cudaStream_t stream,
    cudaTextureObject_t texture,
    CUDABuffer<uchar4>* buffer);

void AccumulatePoseEstimationCoeffsFromImagesCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    float baseline_fx,
    float threshold_factor,
    const CUDABuffer<float>& downsampled_depth,
    const CUDABuffer<u16>& downsampled_normals,
    cudaTextureObject_t downsampled_color,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer<float>& surfel_depth,
    const CUDABuffer<u16>& surfel_normals,
    const CUDABuffer<uchar>& surfel_color,
    u32* residual_count,
    float* residual_sum,
    float* H,
    float* b,
    bool debug,
    CUDABuffer<float>* debug_residual_image,
    PoseEstimationHelperBuffers* helper_buffers,
    bool use_gradmag);

void ComputeCostAndResidualCountFromImagesCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    float baseline_fx,
    float threshold_factor,
    const CUDABuffer<float>& downsampled_depth,
    const CUDABuffer<u16>& downsampled_normals,
    cudaTextureObject_t downsampled_color,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer<float>& surfel_depth,
    const CUDABuffer<u16>& surfel_normals,
    const CUDABuffer<uchar>& surfel_color,
    u32* residual_count,
    float* residual_sum,
    PoseEstimationHelperBuffers* helper_buffers,
    bool use_gradmag);

void UpdateSurfelNormalsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    const CUDABuffer<u8>& active_surfels);

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
    const CUDABuffer<u8>& active_surfels);

void OptimizeIntrinsicsCUDA(
    cudaStream_t stream,
    bool optimize_depth_intrinsics,
    bool optimize_color_intrinsics,
    const vector<shared_ptr<Keyframe>>& keyframes,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    PinholeCamera4f* out_color_camera,
    PinholeCamera4f* out_depth_camera,
    float* a,
    CUDABufferPtr<float>* cfactor_buffer,
    IntrinsicsOptimizationHelperBuffers* buffers);

void UpdateSurfelActivationCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABuffer<u8>* active_surfels);

void DeleteSurfelsAndUpdateRadiiCUDA(
    cudaStream_t stream,
    int min_observation_count,
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32* surfel_count,
    u32 surfels_size,
    CUDABuffer<float>* surfels,
    CUDABufferPtr<u32>* deleted_count_buffer);

// void DeleteSurfelsCUDA(
//     cudaStream_t stream,
//     int min_observation_count,
//     const PinholeCamera4f& camera,
//     const DepthParameters& depth_params,
//     const vector<shared_ptr<Keyframe>>& keyframes,
//     u32* surfel_count,
//     u32 surfels_size,
//     CUDABuffer<float>* surfels);

void CompactSurfelsCUDA(
    cudaStream_t stream,
    void** free_spots_temp_storage,
    usize* free_spots_temp_storage_bytes,
    u32 surfel_count,
    u32* surfels_size,
    CUDABuffer_<float>* surfels,
    CUDABuffer_<u8>* active_surfels = nullptr);

void AssignColorsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    CUDABuffer<float>* surfels);

void AssignDescriptorColorsCUDA(
    cudaStream_t stream,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const vector<shared_ptr<Keyframe>>& keyframes,
    u32 surfels_size,
    CUDABuffer<float>* surfels);

// // Renders the first surfel to each pixel which projects to that pixel.
// // No z-buffering is performed, however surfels which are associated with their
// // pixel in the depth image are preferred.
// void RenderSurfelsForPoseEstimationCUDA(
//     cudaStream_t stream,
//     CUDABuffer<float>* surfel_depth,
//     CUDABuffer<u16>* surfel_normal,
//     CUDABuffer<uchar4>* surfel_color,
//     const PinholeCamera4f& camera,
//     const DepthParameters& depth_params,
//     const CUDABuffer<u16>& frame_depth_buffer,
//     const CUDABuffer<u16>& frame_normals_buffer,
//     const SE3f& frame_T_global,
//     u32 surfels_size,
//     const CUDABuffer<float>& surfels,
//     bool debug);

// Converts raw u16 depth to calibrated float depth, and maps the color image
// to the depth intrinsics to fit to the depth image.
void CalibrateDepthAndTransformColorToDepthCUDA(
    cudaStream_t stream,
    const DepthToColorPixelCorner& depth_to_color,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* out_depth,
    CUDABuffer_<uchar>* out_color);

// CalibrateDepthAndTransformColorToDepthCUDA() with depth calibration only
void CalibrateDepthCUDA(
    cudaStream_t stream,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<float>* out_depth);

// For the 1st downsampling iteration, converts from u16 raw depth to float calibrated depth.
void CalibrateAndDownsampleImagesCUDA(
    cudaStream_t stream,
    bool downsample_color,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* downsampled_depth,
    CUDABuffer_<u16>* downsampled_normals,
    CUDABuffer_<uchar>* downsampled_color,
    bool debug);

// For the 2nd and further downsampling iterations.
void DownsampleImagesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* downsampled_depth,
    CUDABuffer_<u16>* downsampled_normals,
    CUDABuffer_<uchar>* downsampled_color,
    bool debug);

// void DownsampleImagesConsistentlyCUDA(
//     cudaStream_t stream,
//     const CUDABuffer_<float>& comparison_depth_buffer,
//     const CUDABuffer_<u16>& comparison_normals_buffer,
//     const CUDABuffer_<float>& depth_buffer,
//     const CUDABuffer_<u16>& normals_buffer,
//     cudaTextureObject_t color_texture,
//     CUDABuffer_<float>* downsampled_depth,
//     CUDABuffer_<u16>* downsampled_normals,
//     CUDABuffer_<uchar>* downsampled_color,
//     bool debug);

void DebugVerifySurfelCount(
    cudaStream_t stream,
    u32 surfel_count,
    u32 surfels_size,
    const CUDABuffer<float>& surfels);


void PCGInitCUDA(
    cudaStream_t stream,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCenterUnprojector& depth_unprojector,
    const PixelCornerProjector& color_projector,
    cudaTextureObject_t color_texture,
    u32 kf_pose_unknown_index,
    u32 surfel_unknown_start_index,
    bool optimize_poses,
    bool optimize_geometry,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    bool optimize_depth_intrinsics,
    bool optimize_color_intrinsics,
    u32 depth_intrinsics_unknown_start_index,
    u32 color_intrinsics_unknown_start_index,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    u32 surfels_size);

void PCGInit2CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    u32 a_unknown_index,
    float a,
    const CUDABuffer_<PCGScalar>& pcg_r,
    const CUDABuffer_<PCGScalar>& pcg_M,
    CUDABuffer_<PCGScalar>* pcg_delta,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n);

void PCGStep1CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCenterUnprojector& depth_unprojector,
    const PixelCornerProjector& color_projector,
    cudaTextureObject_t color_texture,
    u32 kf_pose_unknown_index,
    u32 surfel_unknown_start_index,
    bool optimize_poses,
    bool optimize_geometry,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    bool optimize_depth_intrinsics,
    bool optimize_color_intrinsics,
    u32 depth_intrinsics_unknown_start_index,
    u32 a_unknown_index,
    u32 color_intrinsics_unknown_start_index,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_alpha_d,
    u32 surfels_size);

void PCGStep2CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    u32 a_unknown_index,
    const CUDABuffer_<PCGScalar>& pcg_r,
    const CUDABuffer_<PCGScalar>& pcg_M,
    CUDABuffer_<PCGScalar>* pcg_delta,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_alpha_d,
    CUDABuffer_<PCGScalar>* pcg_beta_n);

void PCGStep3CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_beta_n);

void PCGDebugVerifyResultCUDA(
    cudaStream_t stream,
    u32 unknown_count,
    u32 a_unknown_index,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_delta);

void UpdateSurfelsFromPCGDeltaCUDA(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float>* surfels,
    bool use_descriptor_residuals,
    u32 surfel_unknown_start_index,
    const CUDABuffer_<PCGScalar>& pcg_delta);

void UpdateCFactorsFromPCGDeltaCUDA(
    cudaStream_t stream,
    CUDABuffer_<float>* cfactor_buffer,
    u32 cfactor_unknown_start_index,
    const CUDABuffer_<PCGScalar>& pcg_delta);

}
