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

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <math_constants.h>

#include "badslam/cost_function.cuh"
#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/gauss_newton.cuh"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"


namespace vis {

__forceinline__ __device__ void ComputeRawDepthResidualAndJacobian(
    const PixelCenterUnprojector& unprojector,
    int px,
    int py,
    float pixel_calibrated_depth,
    float depth_residual_inv_stddev,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    float* raw_residual,
    float* jacobian) {
  float3 local_unproj;
  ComputeRawDepthResidual(unprojector, px, py, pixel_calibrated_depth,
                          depth_residual_inv_stddev,
                          surfel_local_position, surfel_local_normal,
                          &local_unproj, raw_residual);
  
  // Compute Jacobian of residual.
  
//   // Old version for exp(hat(T)) * global_T_frame:
//   jacobian[0] = surfel_global_normal.x;
//   jacobian[1] = surfel_global_normal.y;
//   jacobian[2] = surfel_global_normal.z;
//   jacobian[3] = surfel_global_position.y * surfel_global_normal.z - surfel_global_position.z * surfel_global_normal.y;
//   jacobian[4] = -surfel_global_position.x * surfel_global_normal.z + surfel_global_position.z * surfel_global_normal.x;
//   jacobian[5] = surfel_global_position.x * surfel_global_normal.y - surfel_global_position.y * surfel_global_normal.x;
  
  // New version for global_T_frame * exp(hat(T)):
//   jacobian[0] = gtf.row0.x*surfel_global_normal.x + gtf.row1.x*surfel_global_normal.y + gtf.row2.x*surfel_global_normal.z;
//   jacobian[1] = gtf.row0.y*surfel_global_normal.x + gtf.row1.y*surfel_global_normal.y + gtf.row2.y*surfel_global_normal.z;
//   jacobian[2] = gtf.row0.z*surfel_global_normal.x + gtf.row1.z*surfel_global_normal.y + gtf.row2.z*surfel_global_normal.z;
//   jacobian[3] = - surfel_global_normal.x*(gtf.row0.y*local_unproj.z - gtf.row0.z*local_unproj.y)
//                 - surfel_global_normal.y*(gtf.row1.y*local_unproj.z - gtf.row1.z*local_unproj.y)
//                 - surfel_global_normal.z*(gtf.row2.y*local_unproj.z - gtf.row2.z*local_unproj.y);
//   jacobian[4] =   surfel_global_normal.x*(gtf.row0.x*local_unproj.z - gtf.row0.z*local_unproj.x)
//                 + surfel_global_normal.y*(gtf.row1.x*local_unproj.z - gtf.row1.z*local_unproj.x)
//                 + surfel_global_normal.z*(gtf.row2.x*local_unproj.z - gtf.row2.z*local_unproj.x);
//   jacobian[5] = - surfel_global_normal.x*(gtf.row0.x*local_unproj.y - gtf.row0.y*local_unproj.x)
//                 - surfel_global_normal.y*(gtf.row1.x*local_unproj.y - gtf.row1.y*local_unproj.x)
//                 - surfel_global_normal.z*(gtf.row2.x*local_unproj.y - gtf.row2.y*local_unproj.x);
  
  // Simplified form of the new version above by rotating all the vectors into
  // the local frame (which does not change the values of the dot products),
  // i.e., multiplying by frame_tr_global from the left side:
  jacobian[0] = depth_residual_inv_stddev * surfel_local_normal.x;
  jacobian[1] = depth_residual_inv_stddev * surfel_local_normal.y;
  jacobian[2] = depth_residual_inv_stddev * surfel_local_normal.z;
  jacobian[3] = depth_residual_inv_stddev * (-surfel_local_normal.y * local_unproj.z + surfel_local_normal.z * local_unproj.y);
  jacobian[4] = depth_residual_inv_stddev * ( surfel_local_normal.x * local_unproj.z - surfel_local_normal.z * local_unproj.x);
  jacobian[5] = depth_residual_inv_stddev * (-surfel_local_normal.x * local_unproj.y + surfel_local_normal.y * local_unproj.x);
}

__forceinline__ __device__ void ComputeRawDescriptorResidualAndJacobian(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    const float3& ls,  // surfel_local_position
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2,
    float* jacobian_1,
    float* jacobian_2) {
  ComputeRawDescriptorResidual(color_texture, pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, raw_residual_1, raw_residual_2);
  
  float grad_x_fx_1;
  float grad_y_fy_1;
  float grad_x_fx_2;
  float grad_y_fy_2;
  DescriptorJacobianWrtProjectedPosition(
      color_texture, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
  grad_x_fx_1 *= color_center_projector.fx;
  grad_x_fx_2 *= color_center_projector.fx;
  grad_y_fy_1 *= color_center_projector.fy;
  grad_y_fy_2 *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
  jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
  jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
  jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
  jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;
  
  jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
  jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
  jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
  jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

__forceinline__ __device__ void ComputeRawDescriptorResidualAndJacobianWithFloatTexture(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    const float3& ls,  // surfel_local_position
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2,
    float* jacobian_1,
    float* jacobian_2) {
  ComputeRawDescriptorResidualWithFloatTexture(color_texture, pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, raw_residual_1, raw_residual_2);
  
  float grad_x_fx_1;
  float grad_y_fy_1;
  float grad_x_fx_2;
  float grad_y_fy_2;
  DescriptorJacobianWrtProjectedPositionWithFloatTexture(
      color_texture, pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
  grad_x_fx_1 *= color_center_projector.fx;
  grad_x_fx_2 *= color_center_projector.fx;
  grad_y_fy_1 *= color_center_projector.fy;
  grad_y_fy_2 *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
  jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
  jacobian_1[2] = (ls.x * grad_x_fx_1 + ls.y * grad_y_fy_1) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian_1[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
  jacobian_1[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
  jacobian_1[5] = -(ls.x * grad_y_fy_1 - ls.y * grad_x_fx_1) * inv_ls_z;
  
  jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
  jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
  jacobian_2[2] = (ls.x * grad_x_fx_2 + ls.y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
  jacobian_2[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
  jacobian_2[5] = -(ls.x * grad_y_fy_2 - ls.y * grad_x_fx_2) * inv_ls_z;
}

__forceinline__ __device__ void ComputeRawColorResidualAndJacobian(
    const PixelCenterProjector& color_center_projector,
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float3& ls,  // surfel_local_position
    float surfel_gradmag,
    float* raw_residual,
    float* jacobian) {
  ComputeRawColorResidual(color_texture, pxy, surfel_gradmag, raw_residual);
  
  float grad_x_fx;
  float grad_y_fy;
  ColorJacobianWrtProjectedPosition(
      color_texture, pxy, &grad_x_fx, &grad_y_fy);
  grad_x_fx *= color_center_projector.fx;
  grad_y_fy *= color_center_projector.fy;
  
  float inv_ls_z = 1.f / ls.z;
  float ls_z_sq = ls.z * ls.z;
  float inv_ls_z_sq = inv_ls_z * inv_ls_z;
  
  jacobian[0] = -grad_x_fx * inv_ls_z;
  jacobian[1] = -grad_y_fy * inv_ls_z;
  jacobian[2] = (ls.x * grad_x_fx + ls.y * grad_y_fy) * inv_ls_z_sq;
  
  float ls_x_y = ls.x * ls.y;
  
  jacobian[3] =  ((ls.y * ls.y + ls_z_sq) * grad_y_fy + ls_x_y * grad_x_fx) * inv_ls_z_sq;
  jacobian[4] = -((ls.x * ls.x + ls_z_sq) * grad_x_fx + ls_x_y * grad_y_fy) * inv_ls_z_sq;
  jacobian[5] = -(ls.x * grad_y_fy - ls.y * grad_x_fx) * inv_ls_z;
}

template <int block_width, int block_height>
__forceinline__ __device__ void AccumulatePoseResidualAndCount(
    bool visible,
    float residual,
    CUDABuffer_<u32>& residual_count_buffer,
    CUDABuffer_<float>& residual_buffer,
    typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* float_storage,
    typename cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* int_storage) {
  typedef typename cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef typename cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  
  __syncthreads();  // Required before re-use of shared memory.
  int num_valid_residuals = BlockReduceInt(*int_storage).Sum(visible ? 1 : 0);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    //residual_count_buffer[blockIdx.x] = num_valid_residuals;
    atomicAdd(&residual_count_buffer(0, 0), static_cast<u32>(num_valid_residuals));
  }
  
  __syncthreads();  // Required before re-use of shared memory.
  const float residual_sum =
      BlockReduceFloat(*float_storage).Sum(visible ? residual : 0.f);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    atomicAdd(&residual_buffer(0, 0), residual_sum);
  }
}


template <int block_width, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterProjector color_center_projector,
    PixelCornerProjector color_corner_projector,
    PixelCenterUnprojector depth_unprojector,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &visible, &r)) {
    return;
  }
  
  float jacobian[6];
  float raw_residual;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  // TODO: Would it be faster to do the accumulation only once, while summing
  //       both residual types at the same time?
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    float3 surfel_local_normal = s.frame_T_global.Rotate(r.surfel_normal);  // TODO: Could be gotten from surfel association instead of computing it twice
    
    float depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, surfel_local_normal, s.depth_params.baseline_fx);
    
    ComputeRawDepthResidualAndJacobian(
        depth_unprojector,
        r.px,
        r.py,
        r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position,
        surfel_local_normal,
        &raw_residual,
        jacobian);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual,
        ComputeDepthResidualWeight(raw_residual),
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  // --- Descriptor residual ---
  if (use_descriptor_residuals) {
    float raw_residual_2;
    float jacobian_2[6];
    
    float2 color_pxy;
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      float2 t1_pxy, t2_pxy;
      ComputeTangentProjections(
          r.surfel_global_position,
          r.surfel_normal,
          SurfelGetRadiusSquared(s.surfels, surfel_index),
          s.frame_T_global,
          color_corner_projector,
          &t1_pxy,
          &t2_pxy);
      ComputeRawDescriptorResidualAndJacobian(
          color_center_projector,
          color_texture,
          color_pxy,
          t1_pxy, t2_pxy,
          r.surfel_local_position,
          s.surfels(kSurfelDescriptor1, surfel_index),
          s.surfels(kSurfelDescriptor2, surfel_index),
          &raw_residual,
          &raw_residual_2,
          jacobian,
          jacobian_2);
    } else {
      visible = false;
    }
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual,
        ComputeDescriptorResidualWeight(raw_residual),
        jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_residual_2,
        ComputeDescriptorResidualWeight(raw_residual_2),
        jacobian_2,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_residual),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

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
    const CUDABuffer_<float>& b_buffer) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          AccumulatePoseEstimationCoeffsCUDAKernel,
          256,
          s.surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _debug, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          s,
          depth_to_color,
          color_center_projector,
          color_corner_projector,
          depth_unprojector,
          color_texture,
          residual_count_buffer,
          residual_buffer,
          H_buffer,
          b_buffer));
  CUDA_CHECK();
}


template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  float descriptor_jacobian_1[6];
  float descriptor_jacobian_2[6];
  float raw_descriptor_residual_1;
  float raw_descriptor_residual_2;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // TODO: De-duplicate this with the identical code below in this file
                  // Compute descriptor in surfel image
                  const float intensity = 1 / 255.f * surfel_color(y, x);
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));
                  
                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    ComputeRawDescriptorResidualAndJacobianWithFloatTexture(
                        color_center_projector,
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_local_position,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2,
                        descriptor_jacobian_1,
                        descriptor_jacobian_2);
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(raw_descriptor_residual_1) : 0)) :  // NOTE: Using the 1st residual only
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor),
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_1,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_1, threshold_factor),
        descriptor_jacobian_1,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual_2,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_2, threshold_factor),
        descriptor_jacobian_2,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_descriptor_residual_1, threshold_factor),  // NOTE: Using the 1st residual only
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

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
    CUDABuffer_<float>* debug_residual_image) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
    CUDA_AUTO_TUNE_2D_TEMPLATED(
        AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY,
        32, 32,
        surfel_depth.width(), surfel_depth.height(),
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
        /* kernel parameters */
        depth_projector,
        color_center_projector,
        depth_unprojector,
        baseline_fx,
        depth_to_color,
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth,
        surfel_normals,
        surfel_color,
        frame_depth,
        frame_normals,
        frame_color,
        residual_count_buffer,
        residual_buffer,
        H_buffer,
        b_buffer,
        debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}


template <int block_width, int block_height, bool debug, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag(
    PixelCornerProjector depth_projector,
    PixelCenterProjector color_center_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer,
    CUDABuffer_<float> H_buffer,
    CUDABuffer_<float> b_buffer,
    CUDABuffer_<float> debug_residual_image) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float depth_jacobian[6];
  float raw_depth_residual;
  float descriptor_jacobian[6];
  float raw_descriptor_residual;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                ComputeRawDepthResidualAndJacobian(
                    depth_unprojector,
                    px,
                    py,
                    pixel_calibrated_depth,
                    depth_residual_inv_stddev,
                    surfel_local_position,
                    surfel_local_normal,
                    &raw_depth_residual,
                    depth_jacobian);
              }
              
              if (use_descriptor_residuals) {
                float2 color_pxy;
                if (TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                  ComputeRawColorResidualAndJacobian(
                      color_center_projector,
                      frame_color,
                      color_pxy,
                      surfel_local_position,
                      surfel_color(y, x),
                      &raw_descriptor_residual,
                      descriptor_jacobian);
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Write residual debug image?
  if (debug && x < surfel_depth.width() && y < surfel_depth.height()) {
    debug_residual_image(y, x) =
        visible ?
        ((use_depth_residuals ? ComputeWeightedDepthResidual(raw_depth_residual) : 0) +
         (use_descriptor_residuals ? ComputeWeightedDescriptorResidual(raw_descriptor_residual) : 0)) :
        CUDART_NAN_F;
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_depth_residual,
        ComputeDepthResidualWeight(raw_depth_residual, threshold_factor),
        depth_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
  
  if (use_descriptor_residuals) {
    AccumulateGaussNewtonHAndB<6, block_width, block_height>(
        visible,
        raw_descriptor_residual,
        ComputeDescriptorResidualWeight(raw_descriptor_residual, threshold_factor),
        descriptor_jacobian,
        H_buffer,
        b_buffer,
        &temp_storage.float_storage);
    
    if (debug) {
      AccumulatePoseResidualAndCount<block_width, block_height>(
          visible,
          ComputeWeightedDescriptorResidual(raw_descriptor_residual, threshold_factor),
          residual_count_buffer,
          residual_buffer,
          &temp_storage.float_storage,
          &temp_storage.int_storage);
    }
  }
}

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
    CUDABuffer_<float>* debug_residual_image) {
  COMPILE_OPTION_3(debug, use_depth_residuals, use_descriptor_residuals,
    CUDA_AUTO_TUNE_2D_TEMPLATED(
        AccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag,
        32, 32,
        surfel_depth.width(), surfel_depth.height(),
        0, stream,
        TEMPLATE_ARGUMENTS(block_width, block_height, _debug, _use_depth_residuals, _use_descriptor_residuals),
        /* kernel parameters */
        depth_projector,
        color_center_projector,
        depth_unprojector,
        baseline_fx,
        depth_to_color,
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth,
        surfel_normals,
        surfel_color,
        frame_depth,
        frame_normals,
        frame_color,
        residual_count_buffer,
        residual_buffer,
        H_buffer,
        b_buffer,
        debug_residual_image ? *debug_residual_image : CUDABuffer_<float>()));
}


template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  float raw_descriptor_residual_1;
  float raw_descriptor_residual_2;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                if (x < surfel_depth.width() - 1 &&  // NOTE: These conditions are only necessary since we compute descriptors in the input image and always go right / down
                    y < surfel_depth.height() - 1) {
                  // Compute descriptor in surfel image
                  const float intensity = 1 / 255.f * surfel_color(y, x);
                  const float t1_intensity = 1 / 255.f * surfel_color(y, x + 1);
                  const float t2_intensity = 1 / 255.f * surfel_color(y + 1, x);
                  
                  float surfel_descriptor_1 = (180.f * (t1_intensity - intensity));
                  float surfel_descriptor_2 = (180.f * (t2_intensity - intensity));
                  
                  // Transform the two offset points to the target / estimate frame.
                  // In order not to require depth estimates at both offset pixels,
                  // we estimate their depth using the center pixel's normal.
                  float3 surfel_normal = U16ToImageSpaceNormal(surfel_normals(y, x));
                  const float plane_d =
                      (depth_unprojector.nx(x) * surfel_calibrated_depth) * surfel_normal.x +
                      (depth_unprojector.ny(y) * surfel_calibrated_depth) * surfel_normal.y + surfel_calibrated_depth * surfel_normal.z;
                  
                  float x_plus_1_depth = plane_d / (depth_unprojector.nx(x + 1) * surfel_normal.x + depth_unprojector.ny(y) * surfel_normal.y + surfel_normal.z);
                  float3 x_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x + 1, y, x_plus_1_depth);
                  float2 pxy_t1 = depth_projector.Project(x_plus_1_local_position);
                  int t1_px = static_cast<int>(pxy_t1.x);
                  int t1_py = static_cast<int>(pxy_t1.y);
                  if (pxy_t1.x < 0 || pxy_t1.y < 0 ||
                      // t1_px < 0 || t1_py < 0 ||
                      t1_px >= frame_depth.width() || t1_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float y_plus_1_depth = plane_d / (depth_unprojector.nx(x) * surfel_normal.x + depth_unprojector.ny(y + 1) * surfel_normal.y + surfel_normal.z);
                  float3 y_plus_1_local_position = estimate_frame_T_surfel_frame * depth_unprojector.UnprojectPoint(x, y + 1, y_plus_1_depth);
                  float2 pxy_t2 = depth_projector.Project(y_plus_1_local_position);
                  int t2_px = static_cast<int>(pxy_t2.x);
                  int t2_py = static_cast<int>(pxy_t2.y);
                  if (pxy_t2.x < 0 || pxy_t2.y < 0 ||
                      // t2_px < 0 || t2_py < 0 ||
                      t2_px >= frame_depth.width() || t2_py >= frame_depth.height()) {
                    visible = false;
                  }
                  
                  float2 color_pxy, color_pxy_t1, color_pxy_t2;
                  if (visible &&
                      x_plus_1_local_position.z > 0 &&
                      y_plus_1_local_position.z > 0 &&
                      TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy) &&
                      TransformDepthToColorPixelCorner(pxy_t1, depth_to_color, &color_pxy_t1) &&
                      TransformDepthToColorPixelCorner(pxy_t2, depth_to_color, &color_pxy_t2)) {
                    ComputeRawDescriptorResidualWithFloatTexture(
                        frame_color,
                        color_pxy,
                        color_pxy_t1,
                        color_pxy_t2,
                        surfel_descriptor_1,
                        surfel_descriptor_2,
                        &raw_descriptor_residual_1,
                        &raw_descriptor_residual_2);
                  } else {
                    visible = false;
                  }
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    // TODO: It should be possible to merge these two calls and directly accumulate the sum (also use 2 for the residual count then).
    //       It should even be possible to merge it with the depth residual call as well in case both residual types are used.
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_1, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual_2, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
}

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
    const CUDABuffer_<float>& residual_buffer) {
  COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_2D_TEMPLATED(
          ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY,
          32, 32,
          surfel_depth.width(), surfel_depth.height(),
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          depth_projector,
          depth_unprojector,
          baseline_fx,
          depth_to_color,
          threshold_factor,
          estimate_frame_T_surfel_frame,
          surfel_depth,
          surfel_normals,
          surfel_color,
          frame_depth,
          frame_normals,
          frame_color,
          residual_count_buffer,
          residual_buffer));
}


template <int block_width, int block_height, bool use_depth_residuals, bool use_descriptor_residuals>
__global__ void ComputeCostAndResidualCountFromImagesCUDAKernel_GradMag(
    PixelCornerProjector depth_projector,
    PixelCenterUnprojector depth_unprojector,
    float baseline_fx,
    DepthToColorPixelCorner depth_to_color,
    float threshold_factor,
    CUDAMatrix3x4 estimate_frame_T_surfel_frame,
    CUDABuffer_<float> surfel_depth,
    CUDABuffer_<u16> surfel_normals,
    CUDABuffer_<u8> surfel_color,
    CUDABuffer_<float> frame_depth,
    CUDABuffer_<u16> frame_normals,
    cudaTextureObject_t frame_color,
    CUDABuffer_<u32> residual_count_buffer,
    CUDABuffer_<float> residual_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  bool visible = false;
  float raw_depth_residual;
  float raw_descriptor_residual;
  
  if (x < surfel_depth.width() && y < surfel_depth.height()) {
    float surfel_calibrated_depth = surfel_depth(y, x);
    if (surfel_calibrated_depth > 0) {
      float3 surfel_local_position;
      if (estimate_frame_T_surfel_frame.MultiplyIfResultZIsPositive(depth_unprojector.UnprojectPoint(x, y, surfel_calibrated_depth), &surfel_local_position)) {
        int px, py;
        float2 pxy;
        if (ProjectSurfelToImage(
            frame_depth.width(), frame_depth.height(),
            depth_projector,
            surfel_local_position,
            &px, &py,
            &pxy)) {
          float pixel_calibrated_depth = frame_depth(py, px);
          if (pixel_calibrated_depth > 0) {
            float3 surfel_local_normal;
            if (IsAssociatedWithPixel<false>(
                surfel_local_position,
                surfel_normals,
                x,
                y,
                estimate_frame_T_surfel_frame,
                frame_normals,
                px,
                py,
                pixel_calibrated_depth,
                threshold_factor * kDepthResidualDefaultTukeyParam,
                baseline_fx,
                depth_unprojector,
                nullptr,
                &surfel_local_normal)) {
              visible = true;
              
              if (use_depth_residuals) {
                float depth_residual_inv_stddev =
                    ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(px), depth_unprojector.ny(py), pixel_calibrated_depth, surfel_local_normal, baseline_fx);
                
                float3 local_unproj;
                ComputeRawDepthResidual(depth_unprojector, px, py, pixel_calibrated_depth,
                                        depth_residual_inv_stddev,
                                        surfel_local_position, surfel_local_normal,
                                        &local_unproj, &raw_depth_residual);
              }
              
              if (use_descriptor_residuals) {
                float2 color_pxy;
                if (TransformDepthToColorPixelCorner(pxy, depth_to_color, &color_pxy)) {
                  ComputeRawColorResidual(frame_color, color_pxy, surfel_color(y, x), &raw_descriptor_residual);
                } else {
                  visible = false;
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Early exit?
  __shared__ int have_visible;
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    have_visible = 0;
  }
  __syncthreads();
  
  if (visible) {
    have_visible = 1;
  }
  __syncthreads();
  if (have_visible == 0) {
    return;
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  typedef cub::BlockReduce<int, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceInt;
  __shared__ union {
    typename BlockReduceFloat::TempStorage float_storage;
    typename BlockReduceInt::TempStorage int_storage;
  } temp_storage;
  
  if (use_depth_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDepthResidual(raw_depth_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
  
  if (use_descriptor_residuals) {
    AccumulatePoseResidualAndCount<block_width, block_height>(
        visible,
        ComputeWeightedDescriptorResidual(raw_descriptor_residual, threshold_factor),
        residual_count_buffer,
        residual_buffer,
        &temp_storage.float_storage,
        &temp_storage.int_storage);
  }
}

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
    const CUDABuffer_<float>& residual_buffer) {
  COMPILE_OPTION_2(use_depth_residuals, use_descriptor_residuals,
      CUDA_AUTO_TUNE_2D_TEMPLATED(
          ComputeCostAndResidualCountFromImagesCUDAKernel_GradMag,
          32, 32,
          surfel_depth.width(), surfel_depth.height(),
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, block_height, _use_depth_residuals, _use_descriptor_residuals),
          /* kernel parameters */
          depth_projector,
          depth_unprojector,
          baseline_fx,
          depth_to_color,
          threshold_factor,
          estimate_frame_T_surfel_frame,
          surfel_depth,
          surfel_normals,
          surfel_color,
          frame_depth,
          frame_normals,
          frame_color,
          residual_count_buffer,
          residual_buffer));
}

}
