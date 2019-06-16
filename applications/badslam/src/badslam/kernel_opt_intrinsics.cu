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
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"


namespace vis {

constexpr int kARows = 4 + 1;

template <int block_width, bool optimize_color_intrinsics, bool optimize_depth_intrinsics>
__global__ void AccumulateIntrinsicsCoefficientsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCornerProjector color_corner_projector,
    PixelCenterUnprojector depth_center_unprojector,
    float color_fx, float color_fy,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u32> observation_count,
    CUDABuffer_<float> depth_A,
    CUDABuffer_<float> depth_B,
    CUDABuffer_<float> depth_D,
    CUDABuffer_<float> depth_b1,
    CUDABuffer_<float> depth_b2,
    CUDABuffer_<float> color_H,
    CUDABuffer_<float> color_b) {
  constexpr int block_height = 1;
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Parameters: fx_inv, fy_inv, cx_inv, cy_inv, a_0, a_1 (all global), cfactor (per sparsification pixel)
  float depth_jacobian[kARows + 1] = {0, 0, 0, 0, 0};
  float raw_depth_residual = 0;
  
  // Parameters: fx_inv, fy_inv, cx_inv, cy_inv
  float descriptor_jacobian_1[4] = {0, 0, 0, 0};
  float raw_descriptor_residual_1 = 0;
  float descriptor_jacobian_2[4] = {0, 0, 0, 0};
  float raw_descriptor_residual_2 = 0;
  
  int sparse_pixel_index = -1;
  
  SurfelProjectionResult6 r;
  if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
    float nx = depth_center_unprojector.nx(r.px);
    float ny = depth_center_unprojector.ny(r.py);
    
    if (optimize_depth_intrinsics) {
      int sparse_px = r.px / s.depth_params.sparse_surfel_cell_size;
      int sparse_py = r.py / s.depth_params.sparse_surfel_cell_size;
      float cfactor = s.depth_params.cfactor_buffer(sparse_py, sparse_px);
      
      float raw_inv_depth = 1.0f / (s.depth_params.raw_to_float_depth * s.depth_buffer(r.py, r.px));  // TODO: SurfelProjectsToAssociatedPixel() also reads that value, could be gotten from there
      float exp_inv_depth = expf(- s.depth_params.a * raw_inv_depth);
      float corrected_inv_depth = cfactor * exp_inv_depth + raw_inv_depth;
      if (fabs(corrected_inv_depth) > 1e-4f) {  // NOTE: Corresponds to 1000 meters
        float3 local_surfel_normal = s.frame_T_global.Rotate(r.surfel_normal);
        float dot = Dot(make_float3(nx, ny, 1), local_surfel_normal);
        
        float depth_residual_inv_stddev =
            ComputeDepthResidualInvStddevEstimate(nx, ny, r.pixel_calibrated_depth, local_surfel_normal, s.depth_params.baseline_fx);
        
        float jac_base = depth_residual_inv_stddev * dot * exp_inv_depth / (corrected_inv_depth * corrected_inv_depth);
        
        // Depth residual derivative wrt. ...
        // cx_inv (Attention: notice the indexing order!)
        depth_jacobian[2] = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row0.x, s.frame_T_global.row0.y, s.frame_T_global.row0.z));
        // cy_inv
        depth_jacobian[3] = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row1.x, s.frame_T_global.row1.y, s.frame_T_global.row1.z));
        // fx_inv
        depth_jacobian[0] = r.px * depth_jacobian[2];
        // fy_inv
        depth_jacobian[1] = r.py * depth_jacobian[3];
//         // a_0
//         depth_jacobian[4] = -cfactor * jac_base;
        // a
        depth_jacobian[4] = cfactor * raw_inv_depth * jac_base;
        // cfactor
        depth_jacobian[5] = -jac_base;
        
        float3 local_unproj = make_float3(r.pixel_calibrated_depth * nx, r.pixel_calibrated_depth * ny, r.pixel_calibrated_depth);
        ComputeRawDepthResidual(
            depth_residual_inv_stddev, r.surfel_local_position, local_surfel_normal, local_unproj, &raw_depth_residual);
        
        sparse_pixel_index = sparse_px + sparse_py * s.depth_params.cfactor_buffer.width();
      }
    }
    
    if (optimize_color_intrinsics) {
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
        float grad_x_1;
        float grad_y_1;
        float grad_x_2;
        float grad_y_2;
        DescriptorJacobianWrtProjectedPosition(
            color_texture, color_pxy, t1_pxy, t2_pxy, &grad_x_1, &grad_y_1, &grad_x_2, &grad_y_2);
        
        descriptor_jacobian_1[0] = grad_x_1 * nx;
        descriptor_jacobian_1[1] = grad_y_1 * ny;
        descriptor_jacobian_1[2] = grad_x_1;
        descriptor_jacobian_1[3] = grad_y_1;
        
        descriptor_jacobian_2[0] = grad_x_2 * nx;
        descriptor_jacobian_2[1] = grad_y_2 * ny;
        descriptor_jacobian_2[2] = grad_x_2;
        descriptor_jacobian_2[3] = grad_y_2;
        
        float surfel_descriptor_1 = s.surfels(kSurfelDescriptor1, surfel_index);
        float surfel_descriptor_2 = s.surfels(kSurfelDescriptor2, surfel_index);
        ComputeRawDescriptorResidual(
            color_texture, color_pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, &raw_descriptor_residual_1, &raw_descriptor_residual_2);
      }
    }
  }
  
  // TODO: Would it be faster to use a few different shared memory buffers (instead of only a single one) for the reduce operations to avoid some of the __syncthreads()?
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage float_storage;
  
  if (optimize_depth_intrinsics) {
    const float depth_weight = ComputeDepthResidualWeight(raw_depth_residual);
    
    // depth_jacobian.tranpose() * depth_jacobian (top-left part A), as well as
    // depth_jacobian.transpose() * raw_depth_residual (top rows corresponding to A):
    AccumulateGaussNewtonHAndB<kARows, block_width, block_height>(
        sparse_pixel_index >= 0,
        raw_depth_residual,
        depth_weight,
        depth_jacobian,
        depth_A,
        depth_b1,
        &float_storage);
    
    if (sparse_pixel_index >= 0) {
      // depth_jacobian.tranpose() * depth_jacobian (top-right part B):
      #pragma unroll
      for (int i = 0; i < kARows; ++ i) {
        const float depth_jacobian_sq_i = depth_weight * depth_jacobian[/*row*/ i] * depth_jacobian[/*col*/ kARows];
        atomicAdd(&depth_B(i, sparse_pixel_index), depth_jacobian_sq_i);
      }
      
      // depth_jacobian.tranpose() * depth_jacobian (diagonal-only part D):
      const float depth_jacobian_sq_i = depth_weight * depth_jacobian[/*row*/ kARows] * depth_jacobian[/*col*/ kARows];
      atomicAdd(&depth_D(0, sparse_pixel_index), depth_jacobian_sq_i);
      
      // depth_jacobian.transpose() * point_residual (bottom row corresponding to D):
      const float b_pose_i = depth_weight * raw_depth_residual * depth_jacobian[kARows];
      atomicAdd(&depth_b2(0, sparse_pixel_index), b_pose_i);
      
      // observation count:
      atomicAdd(&observation_count(0, sparse_pixel_index), 1);
    }
  }
  
  if (optimize_color_intrinsics) {
    AccumulateGaussNewtonHAndB<4, block_width, block_height>(
        raw_descriptor_residual_1 != 0,
        raw_descriptor_residual_1,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_1),
        descriptor_jacobian_1,
        color_H,
        color_b,
        &float_storage);
    AccumulateGaussNewtonHAndB<4, block_width, block_height>(
        raw_descriptor_residual_2 != 0,
        raw_descriptor_residual_2,
        ComputeDescriptorResidualWeight(raw_descriptor_residual_2),
        descriptor_jacobian_2,
        color_H,
        color_b,
        &float_storage);
  }
}

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
    const CUDABuffer_<float>& color_b) {
  COMPILE_OPTION_2(optimize_color_intrinsics, optimize_depth_intrinsics,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          AccumulateIntrinsicsCoefficientsCUDAKernel,
          1024,
          s.surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _optimize_color_intrinsics, _optimize_depth_intrinsics),
          /* kernel parameters */
          s,
          depth_to_color,
          color_corner_projector,
          depth_center_unprojector,
          color_fx,
          color_fy,
          color_texture,
          observation_count,
          depth_A,
          depth_B,
          depth_D,
          depth_b1,
          depth_b2,
          color_H,
          color_b));
  CUDA_CHECK();
}


template <int block_width>
__global__ void ComputeIntrinsicsIntermediateMatricesCUDAKernel(
    u32 pixel_count,
    CUDABuffer_<float> A,
    CUDABuffer_<float> B,
    CUDABuffer_<float> D,
    CUDABuffer_<float> b1,
    CUDABuffer_<float> b2) {
  unsigned int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // TODO: Would it be faster to use a few different shared memory buffers for the reduce operations to avoid some of the __syncthreads()?
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage float_storage;
  
  u8 weight = 1;
  if (pixel_index >= pixel_count) {
    weight = 0;
    pixel_index = pixel_count - 1;
  }
  
  const float D_inverse = 1.0f / D(0, pixel_index);
  if (!(D_inverse < 1e12f)) {
    weight = 0;
    D(0, pixel_index) = CUDART_NAN_F;
  }
  
//   if (pixel_index >= 5 * 640 + 320 - 14 && pixel_index <= 5 * 640 + 320 + 14) {
//     printf("px index %i D: %f, D_inverse: %f\n", pixel_index, D(0, pixel_index), D_inverse);
//   }
  
  // D^(-1) b2 [Dx1], exclusive access by this thread
  float D_inv_b2;
  if (weight > 0) {
    D_inv_b2 = D_inverse * b2(0, pixel_index);
    // Store in D
    D(0, pixel_index) = D_inv_b2;
  }
  
  // B D^(-1) B^T [AxA], concurrent access
  // TODO: load all the B(:, pixel_index) into variables for better performance?
  int index = 0;
  #pragma unroll
  for (int row = 0; row < kARows; ++ row) {
    #pragma unroll
    for (int col = row; col < kARows; ++ col) {
      float B_D_inv_B_i = B(row, pixel_index) * D_inverse * B(col, pixel_index);
      
      // Accumulate on A(0, index) (subtract from it)
      __syncthreads();  // Required before re-use of shared memory.
      const float block_sum =
          BlockReduceFloat(float_storage).Sum(weight ? B_D_inv_B_i : 0);
      if (threadIdx.x == 0) {
        atomicAdd(&A(0, index), -1.f * block_sum);
        ++ index;
      }
    }
  }
  
  // B D^(-1) b2 [Ax1], concurrent access
  #pragma unroll
  for (int row = 0; row < kARows; ++ row) {
    float B_D_inv_b2_i = B(row, pixel_index) * D_inv_b2;
    
    // Accumulate on b1(0, row) (subtract from it)
    __syncthreads();  // Required before re-use of shared memory.
    const float block_sum =
        BlockReduceFloat(float_storage).Sum(weight ? B_D_inv_b2_i : 0);
    if (threadIdx.x == 0) {
      atomicAdd(&b1(0, row), -1.f * block_sum);
    }
  }
  
  // D^(-1) B^T [DxA], exclusive access by this thread
  if (weight > 0) {
    #pragma unroll
    for (int row = 0; row < kARows; ++ row) {
      float D_inv_B_T_i = D_inverse * B(row, pixel_index);
      
      // Store in B
      B(row, pixel_index) = D_inv_B_T_i;
    }
  }
}

void CallComputeIntrinsicsIntermediateMatricesCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<float>& A,
    const CUDABuffer_<float>& B,
    const CUDABuffer_<float>& D,
    const CUDABuffer_<float>& b1,
    const CUDABuffer_<float>& b2) {
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      ComputeIntrinsicsIntermediateMatricesCUDAKernel,
      1024,
      pixel_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      pixel_count,
      A,
      B,
      D,
      b1,
      b2);
  CUDA_CHECK();
}


template <int block_width>
__global__ void SolveForPixelIntrinsicsUpdateCUDAKernel(
    u32 pixel_count,
    CUDABuffer_<u32> observation_count,
    CUDABuffer_<float> B,
    CUDABuffer_<float> D,
    CUDABuffer_<float> x1,
    CUDABuffer_<float> cfactor_buffer) {
  unsigned int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ float x1_shared[kARows];
  if (threadIdx.x < kARows) {
    x1_shared[threadIdx.x] = x1(0, threadIdx.x);
  }
  
  u8 weight = 1;
  if (pixel_index >= pixel_count) {
    weight = 0;
    pixel_index = pixel_count - 1;
  }
  
  // x2 = (D^(-1) b2) (stored in D) - (D^(-1) B^T x1) (D^(-1) B^T stored in B)
  float offset = D(0, pixel_index);
  
  __syncthreads();  // Make x1_shared available
  
  if (::isnan(offset)) {
    offset = 0;
  } else {
    #pragma unroll
    for (int row = 0; row < kARows; ++ row) {
      offset -= B(row, pixel_index) * x1_shared[row];
    }
  }
  
  int y = pixel_index / cfactor_buffer.width();
  int x = pixel_index - y * cfactor_buffer.width();
  float cfactor = 0;
  if (weight > 0 /*&& observation_count(0, pixel_index) >= 10*/) {
    cfactor = cfactor_buffer(y, x) - offset;
    
    // Reset pixels which do not have any observation anymore to avoid having
    // outlier values stick around
    if (observation_count(0, pixel_index) == 0) {
      weight = 0;
      cfactor = 0;
    }
    
    cfactor_buffer(y, x) = cfactor;
  }
}

void CallSolveForPixelIntrinsicsUpdateCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<u32>& observation_count,
    const CUDABuffer_<float>& B,
    const CUDABuffer_<float>& D,
    const CUDABuffer_<float>& x1,
    const CUDABuffer_<float>& cfactor_buffer) {
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      SolveForPixelIntrinsicsUpdateCUDAKernel,
      1024,
      pixel_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      pixel_count,
      observation_count,
      B,
      D,
      x1,
      cfactor_buffer);
  CUDA_CHECK();
}

}
