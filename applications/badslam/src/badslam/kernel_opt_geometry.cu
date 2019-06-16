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

#include "badslam/cost_function.cuh"
#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

__global__ void ResetSurfelAccumCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
    surfels(kSurfelAccum2, surfel_index) = 0;
    surfels(kSurfelAccum3, surfel_index) = 0;
    surfels(kSurfelAccum4, surfel_index) = 0;
    surfels(kSurfelAccum5, surfel_index) = 0;
    surfels(kSurfelAccum6, surfel_index) = 0;
    surfels(kSurfelAccum7, surfel_index) = 0;
    surfels(kSurfelAccum8, surfel_index) = 0;
  }
}

void CallResetSurfelAccumCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels) {
  CUDA_AUTO_TUNE_1D(
      ResetSurfelAccumCUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}


__global__ void ResetSurfelAccum0to3CUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
    surfels(kSurfelAccum2, surfel_index) = 0;
    surfels(kSurfelAccum3, surfel_index) = 0;
  }
}

void CallResetSurfelAccum0to3CUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels) {
  CUDA_AUTO_TUNE_1D(
      ResetSurfelAccum0to3CUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}


template<bool use_depth_residuals>
__global__ void AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel(
    SurfelProjectionParameters s,
    PixelCenterUnprojector depth_unprojector,
    DepthToColorPixelCorner depth_to_color,
    PixelCornerProjector color_corner_projector,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < s.surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    SurfelProjectionResult6 r;
    if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
      float3 rn = s.frame_T_global.Rotate(r.surfel_normal);
      
      // --- Depth residual change wrt. position change ---
      if (use_depth_residuals) {
        float depth_residual_inv_stddev =
            ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, rn, s.depth_params.baseline_fx);
        
        const float depth_jacobian = -depth_residual_inv_stddev;
        
        float3 local_unproj;
        float raw_depth_residual;
        ComputeRawDepthResidual(
            depth_unprojector, r.px, r.py, r.pixel_calibrated_depth,
            depth_residual_inv_stddev,
            r.surfel_local_position, rn, &local_unproj, &raw_depth_residual);
        
        const float depth_weight = ComputeDepthResidualWeight(raw_depth_residual);
        
        // Accumulate:
        s.surfels(kSurfelAccum0, surfel_index) += depth_weight * depth_jacobian * depth_jacobian;
        s.surfels(kSurfelAccum6, surfel_index) += depth_weight * raw_depth_residual * depth_jacobian;
      }
      // --------------------------------------------------
      
      
      float2 color_pxy;
      if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
        // --- Descriptor residual ---
        float2 t1_pxy, t2_pxy;
        ComputeTangentProjections(
            r.surfel_global_position,
            r.surfel_normal,
            SurfelGetRadiusSquared(s.surfels, surfel_index),
            s.frame_T_global,
            color_corner_projector,
            &t1_pxy,
            &t2_pxy);
        const float surfel_descriptor_1 = s.surfels(kSurfelDescriptor1, surfel_index);
        const float surfel_descriptor_2 = s.surfels(kSurfelDescriptor2, surfel_index);
        float raw_descriptor_residual_1;
        float raw_descriptor_residual_2;
        ComputeRawDescriptorResidual(
            color_texture, color_pxy, t1_pxy, t2_pxy, surfel_descriptor_1, surfel_descriptor_2, &raw_descriptor_residual_1, &raw_descriptor_residual_2);
        // ---------------------------
        
        // --- Descriptor residual change wrt. position change ---
        float grad_x_1;
        float grad_y_1;
        float grad_x_2;
        float grad_y_2;
        DescriptorJacobianWrtProjectedPosition(
            color_texture, color_pxy, t1_pxy, t2_pxy, &grad_x_1, &grad_y_1, &grad_x_2, &grad_y_2);
        
        const float term1 = -color_corner_projector.fx * (rn.x*r.surfel_local_position.z - rn.z*r.surfel_local_position.x);
        const float term2 = -color_corner_projector.fy * (rn.y*r.surfel_local_position.z - rn.z*r.surfel_local_position.y);
        const float term3 = 1.f / (r.surfel_local_position.z * r.surfel_local_position.z);
        float jacobian_wrt_position_1 = -(grad_x_1 * term1 + grad_y_1 * term2) * term3;
        float jacobian_wrt_position_2 = -(grad_x_2 * term1 + grad_y_2 * term2) * term3;
        // -------------------------------------------------------
        
        // --- Descriptor residual change wrt. descriptor change ---
        constexpr float jacobian_wrt_descriptor = -1.f;
        // ---------------------------------------------------------
        
        // Accumulate:
        // kSurfelAccum0: H(0, 0)
        // kSurfelAccum1: H(0, 1) = H(1, 0)
        // kSurfelAccum2: H(0, 2) = H(2, 0)
        // kSurfelAccum3: H(1, 1)
        // kSurfelAccum4: H(1, 2) = H(2, 1)
        // kSurfelAccum5: H(2, 2)
        // kSurfelAccum6: b(0)
        // kSurfelAccum7: b(1)
        // kSurfelAccum8: b(2)
        const float weight_1 = ComputeDescriptorResidualWeight(raw_descriptor_residual_1);
        const float weighted_raw_residual_1 = weight_1 * raw_descriptor_residual_1;
        
        const float weight_2 = ComputeDescriptorResidualWeight(raw_descriptor_residual_2);
        const float weighted_raw_residual_2 = weight_2 * raw_descriptor_residual_2;
        
        // Residual 1 (and some parts of 2, where accumulating onto the same variable)
        s.surfels(kSurfelAccum0, surfel_index) += weight_1 * jacobian_wrt_position_1 * jacobian_wrt_position_1 +
                                                  weight_2 * jacobian_wrt_position_2 * jacobian_wrt_position_2;  // from residual 2
        s.surfels(kSurfelAccum1, surfel_index) += weight_1 * jacobian_wrt_position_1 * jacobian_wrt_descriptor;
        s.surfels(kSurfelAccum3, surfel_index) += weight_1 * jacobian_wrt_descriptor * jacobian_wrt_descriptor;
        s.surfels(kSurfelAccum6, surfel_index) += weighted_raw_residual_1 * jacobian_wrt_position_1 +
                                                  weighted_raw_residual_2 * jacobian_wrt_position_2;  // from residual 2
        s.surfels(kSurfelAccum7, surfel_index) += weighted_raw_residual_1 * jacobian_wrt_descriptor;
        
        // Residual 2
        s.surfels(kSurfelAccum2, surfel_index) += weight_2 * jacobian_wrt_position_2 * jacobian_wrt_descriptor;
        s.surfels(kSurfelAccum5, surfel_index) += weight_2 * jacobian_wrt_descriptor * jacobian_wrt_descriptor;
        s.surfels(kSurfelAccum8, surfel_index) += weighted_raw_residual_2 * jacobian_wrt_descriptor;
      }
    }
  }
}

void AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel(
    cudaStream_t stream,
    const SurfelProjectionParameters& s,
    const PixelCenterUnprojector& depth_unprojector,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u8>& active_surfels,
    bool use_depth_residuals) {
  if (use_depth_residuals) {
    CUDA_AUTO_TUNE_1D(
        AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel<true>,
        512,
        s.surfels_size,
        0, stream,
        /* kernel parameters */
        s,
        depth_unprojector,
        depth_to_color,
        color_corner_projector,
        color_texture,
        active_surfels);
  } else {
    CUDA_AUTO_TUNE_1D(
        AccumulateSurfelPositionAndDescriptorOptimizationCoeffsCUDAKernel<false>,
        512,
        s.surfels_size,
        0, stream,
        /* kernel parameters */
        s,
        depth_unprojector,
        depth_to_color,
        color_corner_projector,
        color_texture,
        active_surfels);
  }
  CUDA_CHECK();
}


__global__ void UpdateSurfelPositionAndDescriptorCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    float H_0_0 = surfels(kSurfelAccum0, surfel_index);
    float H_0_1 = surfels(kSurfelAccum1, surfel_index);
    float H_0_2 = surfels(kSurfelAccum2, surfel_index);
    float H_1_1 = surfels(kSurfelAccum3, surfel_index);
    float H_1_2 = surfels(kSurfelAccum4, surfel_index);
    float H_2_2 = surfels(kSurfelAccum5, surfel_index);
    
    // Make sure that the matrix is positive definite
    // (instead of only semi-positive definite).
    constexpr float kEpsilon = 1e-6f;
    H_0_0 += kEpsilon;
    H_1_1 += kEpsilon;
    H_2_2 += kEpsilon;
    
    // Perform in-place Cholesky decomposition of H
    H_0_0 = sqrtf(H_0_0);
    H_0_1 = H_0_1 / H_0_0;
    H_1_1 = sqrtf(H_1_1 - H_0_1 * H_0_1);
    H_0_2 = H_0_2 / H_0_0;
    H_1_2 = (H_1_2 - H_0_2 * H_0_1) / H_1_1;
    H_2_2 = sqrtf(H_2_2 - H_0_2 * H_0_2 - H_1_2 * H_1_2);
    
    // Solve H * x = b for x.
    //
    // (H_0_0     0     0)   (H_0_0 H_0_1 H_0_2)   (x0)   (b0)
    // (H_0_1 H_1_1     0) * (    0 H_1_1 H_1_2) * (x1) = (b1)
    // (H_0_2 H_1_2 H_2_2)   (    0     0 H_2_2)   (x2)   (b2)
    //
    // Naming the result of the second multiplication y, we get:
    //
    // (H_0_0     0     0)   (y0)   (b0)
    // (H_0_1 H_1_1     0) * (y1) = (b1)
    // (H_0_2 H_1_2 H_2_2)   (y2)   (b2)
    // 
    // and:
    // 
    // (H_0_0 H_0_1 H_0_2)   (x0)   (y0)
    // (    0 H_1_1 H_1_2) * (x1) = (y1)
    // (    0     0 H_2_2)   (x2) = (y2)
    
    const float b0 = surfels(kSurfelAccum6, surfel_index);
    const float b1 = surfels(kSurfelAccum7, surfel_index);
    const float b2 = surfels(kSurfelAccum8, surfel_index);
    
    float y0 = b0 / H_0_0;
    float y1 = (b1 - H_0_1 * y0) / H_1_1;
    float y2 = (b2 - H_0_2 * y0 - H_1_2 * y1) / H_2_2;
    
    float x2 = y2 / H_2_2;
    float x1 = (y1 - H_1_2 * x2) / H_1_1;
    float x0 = (y0 - H_0_2 * x2 - H_0_1 * x1) / H_0_0;
    
    if (x0 != 0) {
      // Update surfel position
      float3 global_position = SurfelGetPosition(surfels, surfel_index);
      float3 surfel_normal = SurfelGetNormal(surfels, surfel_index);
      SurfelSetPosition(&surfels, surfel_index, global_position - x0 * surfel_normal);
    }
    
    if (x1 != 0) {
      float surfel_descriptor_1 = surfels(kSurfelDescriptor1, surfel_index);
      surfel_descriptor_1 -= x1;
      surfels(kSurfelDescriptor1, surfel_index) = ::max(-180.f, ::min(180.f, surfel_descriptor_1));
    }
    
    if (x2 != 0) {
      float surfel_descriptor_2 = surfels(kSurfelDescriptor2, surfel_index);
      surfel_descriptor_2 -= x2;
      surfels(kSurfelDescriptor2, surfel_index) = ::max(-180.f, ::min(180.f, surfel_descriptor_2));
    }
    
    // Reset accum fields for normal optimization.
    // surfels(kSurfelAccum0, surfel_index) = 0;
    // surfels(kSurfelAccum1, surfel_index) = 0;
    // surfels(kSurfelAccum2, surfel_index) = 0;
    // surfels(kSurfelAccum3, surfel_index) = 0;
  }
}

void CallUpdateSurfelPositionAndDescriptorCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  CUDA_AUTO_TUNE_1D(
      UpdateSurfelPositionAndDescriptorCUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */  
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}


__global__ void ResetSurfelAccum0to1CUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
  }
}

void CallResetSurfelAccum0to1CUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    const CUDABuffer_<u8>& active_surfels) {
  CUDA_AUTO_TUNE_1D(
      ResetSurfelAccum0to1CUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}


// This function only considers the depth residual. If the descriptor residual
// is also used, it should be considered jointly.
__global__ void AccumulateSurfelPositionOptimizationCoeffsFromDepthResidualCUDAKernel(
    SurfelProjectionParameters s,
    PixelCenterUnprojector depth_unprojector,
    DepthToColorPixelCorner depth_to_color,
    float color_fx,
    float color_fy,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < s.surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    SurfelProjectionResult6 r;
    if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
      // --- Depth residual ---
      float3 rn = s.frame_T_global.Rotate(r.surfel_normal);
      
      float depth_residual_inv_stddev =
          ComputeDepthResidualInvStddevEstimate(depth_unprojector.nx(r.px), depth_unprojector.ny(r.py), r.pixel_calibrated_depth, rn, s.depth_params.baseline_fx);
      
      const float depth_jacobian = -depth_residual_inv_stddev;
      
      float3 local_unproj;
      float raw_depth_residual;
      ComputeRawDepthResidual(
          depth_unprojector, r.px, r.py, r.pixel_calibrated_depth,
          depth_residual_inv_stddev,
          r.surfel_local_position, rn, &local_unproj, &raw_depth_residual);
      
      // Accumulate:
      // kSurfelAccum0: H
      // kSurfelAccum1: b
      const float depth_weight = ComputeDepthResidualWeight(raw_depth_residual);
      float weighted_jacobian = depth_weight * depth_jacobian;
      
      s.surfels(kSurfelAccum0, surfel_index) += weighted_jacobian * depth_jacobian;
      s.surfels(kSurfelAccum1, surfel_index) += weighted_jacobian * raw_depth_residual;
    }
  }
}

void CallAccumulateSurfelPositionOptimizationCoeffsFromDepthResidualCUDAKernel(
    cudaStream_t stream,
    SurfelProjectionParameters s,
    PixelCenterUnprojector depth_unprojector,
    DepthToColorPixelCorner depth_to_color,
    float color_fx,
    float color_fy,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u8> active_surfels) {
  CUDA_AUTO_TUNE_1D(
      AccumulateSurfelPositionOptimizationCoeffsFromDepthResidualCUDAKernel,
      512,
      s.surfels_size,
      0, stream,
      /* kernel parameters */
      s,
      depth_unprojector,
      depth_to_color,
      color_fx,
      color_fy,
      color_texture,
      active_surfels);
  CUDA_CHECK();
}


__global__ void UpdateSurfelPositionCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    float H = surfels(kSurfelAccum0, surfel_index);
    constexpr float kEpsilon = 1e-6f;
    if (H > kEpsilon) {
      float3 global_position = SurfelGetPosition(surfels, surfel_index);
      float t = -1.f * surfels(kSurfelAccum1, surfel_index) / H;
      float3 surfel_normal = SurfelGetNormal(surfels, surfel_index);
      SurfelSetPosition(&surfels, surfel_index, global_position + t * surfel_normal);
    }
  }
}

void CallUpdateSurfelPositionCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  CUDA_AUTO_TUNE_1D(
      UpdateSurfelPositionCUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */  
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}


__global__ void AccumulateSurfelNormalOptimizationCoeffsCUDAKernel(
    SurfelProjectionParameters s,
    CUDAMatrix3x3 global_R_frame,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < s.surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    SurfelProjectionResultXY r;
    if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
      // Transform the frame's normal to global space.
      float3 local_normal = U16ToImageSpaceNormal(s.normals_buffer(r.py, r.px));
      float3 global_normal = global_R_frame * local_normal;
      
      // Accumulate.
      // kSurfelAccum0: normal.x
      // kSurfelAccum1: normal.y
      // kSurfelAccum2: normal.z
      // kSurfelAccum3: count
      // NOTE: This does a simple averaging of the normals, it does not
      //       optimize according to the cost function.
      s.surfels(kSurfelAccum0, surfel_index) += global_normal.x;
      s.surfels(kSurfelAccum1, surfel_index) += global_normal.y;
      s.surfels(kSurfelAccum2, surfel_index) += global_normal.z;
      s.surfels(kSurfelAccum3, surfel_index) += 1.f;
    }
  }
}

void CallAccumulateSurfelNormalOptimizationCoeffsCUDAKernel(
    cudaStream_t stream,
    SurfelProjectionParameters s,
    CUDAMatrix3x3 global_R_frame,
    CUDABuffer_<u8> active_surfels) {
  CUDA_AUTO_TUNE_1D(
      AccumulateSurfelNormalOptimizationCoeffsCUDAKernel,
      512,
      s.surfels_size,
      0, stream,
      /* kernel parameters */
      s,
      global_R_frame,
      active_surfels);
  CUDA_CHECK();
}


__global__ void UpdateSurfelNormalCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    if (!(active_surfels(0, surfel_index) & kSurfelActiveFlag)) {
      return;
    }
    
    float count = surfels(kSurfelAccum3, surfel_index);
    if (count >= 1) {
      float3 normal_sum =
          make_float3(surfels(kSurfelAccum0, surfel_index),
                      surfels(kSurfelAccum1, surfel_index),
                      surfels(kSurfelAccum2, surfel_index));
      SurfelSetNormal(&surfels, surfel_index, (1.f / count) * normal_sum);
    }
  }
}

void CallUpdateSurfelNormalCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    CUDABuffer_<u8> active_surfels) {
  CUDA_AUTO_TUNE_1D(
      UpdateSurfelNormalCUDAKernel,
      512,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels,
      active_surfels);
  CUDA_CHECK();
}

}
