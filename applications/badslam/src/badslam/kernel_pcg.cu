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

#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

// NOTE: This parameter corresponds to the lambda parameter in Levenberg-Marquardt,
//       and must be positive to make the matrix to solve positive definite. It
//       should be as small as possible in general however (unless actually doing
//       Levenberg-Marquardt) such as not to slow the optimization down.
constexpr PCGScalar kDiagEpsilon = 1e-8;


// TODO: De-duplicate this setting with the alternating optimization; make it configurable?
constexpr float kAPriorWeight = 10;


// Implementation of CUDA's "atomicAdd()" that works for both floats and doubles
// (i.e., can be used with PCGScalar).
template<typename T> __forceinline__ __device__ T atomicAddFloatOrDouble(T* address, T value);

template<> __forceinline__ __device__ float atomicAddFloatOrDouble(float* address, float value) {
  return atomicAdd(address, value);
}

// Implementation of CUDA's "atomicAdd()" for doubles. Directly taken from the
// CUDA C Programming Guide.
template<> __forceinline__ __device__ double atomicAddFloatOrDouble(double* address, double value) {
  unsigned long long int* address_as_ull =
                            (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

// Efficient sum over all threads in a CUDA thread block.
// Every thread participating here must aim to use the same dest (since only the
// value of thread 0 will be used), and all threads in the block must participate.
// For varying dest, use atomicAddFloatOrDouble().
template<int block_width, int block_height>
__forceinline__ __device__ void BlockedAtomicSum(
    PCGScalar* dest,
    PCGScalar value,
    bool visible,
    typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* storage) {
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  const PCGScalar sum = BlockReduceScalar(*storage).Sum(visible ? value : 0.f);
  if (threadIdx.x == 0 && (block_height == 1 || threadIdx.y == 0)) {
    atomicAddFloatOrDouble(dest, sum);
  }
}

// Implements part of the accumulation for PCG.
// Every thread participating here must aim to use the same unknown index (since only the
// value of thread 0 will be used), and all threads in the block must participate.
// For varying unknown indices, use AtomicSumRAndM().
template<int block_width, int block_height>
__forceinline__ __device__ void BlockedAtomicSumRAndM(
    int unknown_index,
    PCGScalar jacobian,
    PCGScalar weight,
    PCGScalar raw_residual,
    bool visible,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* storage) {
  const PCGScalar weighted_jacobian = weight * jacobian;
  
  BlockedAtomicSum<block_width, block_height>(
      &(*pcg_r)(0, unknown_index),
      -1 * weighted_jacobian * raw_residual,
      visible, storage);
  
  __syncthreads();
  
  BlockedAtomicSum<block_width, block_height>(
      &(*pcg_M)(0, unknown_index),
      jacobian * weighted_jacobian,
      visible, storage);
}

// Version of BlockedAtomicSumRAndM() for two residuals relating to the same
// unknown.
template<int block_width, int block_height>
__forceinline__ __device__ void BlockedAtomicSumRAndM2(
    int unknown_index,
    PCGScalar jacobian_1,
    PCGScalar weight_1,
    PCGScalar raw_residual_1,
    PCGScalar jacobian_2,
    PCGScalar weight_2,
    PCGScalar raw_residual_2,
    bool visible,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M,
    typename cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height>::TempStorage* storage) {
  const PCGScalar weighted_jacobian_1 = weight_1 * jacobian_1;
  const PCGScalar weighted_jacobian_2 = weight_2 * jacobian_2;
  
  BlockedAtomicSum<block_width, block_height>(
      &(*pcg_r)(0, unknown_index),
      -1 * weighted_jacobian_1 * raw_residual_1 +
      -1 * weighted_jacobian_2 * raw_residual_2,
      visible, storage);
  
  __syncthreads();
  
  BlockedAtomicSum<block_width, block_height>(
      &(*pcg_M)(0, unknown_index),
      jacobian_1 * weighted_jacobian_1 +
      jacobian_2 * weighted_jacobian_2,
      visible, storage);
}

__forceinline__ __device__ void AtomicSumRAndM(
    int unknown_index,
    PCGScalar jacobian,
    PCGScalar weight,
    PCGScalar raw_residual,
    bool visible,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_M) {
  if (!visible) {
    return;
  }
  
  const PCGScalar weighted_jacobian = weight * jacobian;
  
  atomicAddFloatOrDouble(
      &(*pcg_r)(0, unknown_index),
      -1 * weighted_jacobian * raw_residual);
  
  atomicAddFloatOrDouble(
      &(*pcg_M)(0, unknown_index),
      jacobian * weighted_jacobian);
}

template<int block_width, bool optimize_poses, bool optimize_geometry, bool optimize_depth_intrinsics, bool optimize_color_intrinsics>
__global__ void PCGInitCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterUnprojector depth_center_unprojector,
    PixelCornerProjector color_corner_projector,
    cudaTextureObject_t color_texture,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    u32 kf_pose_unknown_index,
    u32 surfel_unknown_start_index,
    u32 depth_intrinsics_unknown_start_index,
    u32 color_intrinsics_unknown_start_index,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  __shared__ int have_visible;
  __shared__ int have_visible_2;
  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &have_visible, &have_visible_2, &visible, &r)) {
    return;
  }
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
  
  float3 rn = s.frame_T_global.Rotate(r.surfel_normal);
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    // Compute residual
    PCGScalar depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_center_unprojector.nx(r.px), depth_center_unprojector.ny(r.py), r.pixel_calibrated_depth, rn, s.depth_params.baseline_fx);
    float3 local_unproj;
    float raw_residual;
    ComputeRawDepthResidual(
        depth_center_unprojector, r.px, r.py, r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position, rn, &local_unproj, &raw_residual);
    
    // Compute weight
    const PCGScalar weight = ComputeDepthResidualWeight(raw_residual);
    
    // Jacobian wrt. position changes:
    if (visible && optimize_geometry) {
      const PCGScalar jacobian_wrt_position = -depth_residual_inv_stddev;
      pcg_r(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index) -= jacobian_wrt_position * weight * raw_residual;
      pcg_M(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index) += jacobian_wrt_position * weight * jacobian_wrt_position;
    }
    
    // Jacobian wrt. pose changes:
    if (optimize_poses) {
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 0,
          depth_residual_inv_stddev * rn.x,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 1,
          depth_residual_inv_stddev * rn.y,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 2,
          depth_residual_inv_stddev * rn.z,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 3,
          depth_residual_inv_stddev * (-rn.y * local_unproj.z + rn.z * local_unproj.y),  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 4,
          depth_residual_inv_stddev * ( rn.x * local_unproj.z - rn.z * local_unproj.x),  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          kf_pose_unknown_index + 5,
          depth_residual_inv_stddev * (-rn.x * local_unproj.y + rn.y * local_unproj.x),  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
    }
    
    // Jacobian wrt. depth intrinsics changes:
    if (optimize_depth_intrinsics) {
      int sparse_px = r.px / s.depth_params.sparse_surfel_cell_size;
      int sparse_py = r.py / s.depth_params.sparse_surfel_cell_size;
      float cfactor = s.depth_params.cfactor_buffer(sparse_py, sparse_px);
      
      float raw_inv_depth = 1.0f / (s.depth_params.raw_to_float_depth * s.depth_buffer(r.py, r.px));  // TODO: SurfelProjectsToAssociatedPixel() also reads that value, could be gotten from there
      float exp_inv_depth = expf(- s.depth_params.a * raw_inv_depth);
      float corrected_inv_depth = cfactor * exp_inv_depth + raw_inv_depth;
      if (fabs(corrected_inv_depth) < 1e-4f) {  // NOTE: Corresponds to 1000 meters
        visible = false;
      }
      
      float3 local_surfel_normal = s.frame_T_global.Rotate(r.surfel_normal);
      float nx = depth_center_unprojector.nx(r.px);
      float ny = depth_center_unprojector.ny(r.py);
      float dot = Dot(make_float3(nx, ny, 1), local_surfel_normal);
      
      float jac_base = depth_residual_inv_stddev * dot * exp_inv_depth / (corrected_inv_depth * corrected_inv_depth);
      
      // Depth residual derivative wrt. ...
      // cx_inv (Attention: notice the indexing order!)
      __syncthreads();
      float d_residual_d_cx_inv = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row0.x, s.frame_T_global.row0.y, s.frame_T_global.row0.z));
      BlockedAtomicSumRAndM<block_width, block_height>(
          depth_intrinsics_unknown_start_index + 2,
          d_residual_d_cx_inv,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      // cy_inv
      __syncthreads();
      float d_residual_d_cy_inv = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row1.x, s.frame_T_global.row1.y, s.frame_T_global.row1.z));
      BlockedAtomicSumRAndM<block_width, block_height>(
          depth_intrinsics_unknown_start_index + 3,
          d_residual_d_cy_inv,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      // fx_inv
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          depth_intrinsics_unknown_start_index + 0,
          r.px * d_residual_d_cx_inv,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      // fy_inv
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          depth_intrinsics_unknown_start_index + 1,
          r.py * d_residual_d_cy_inv,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      // a
      __syncthreads();
      BlockedAtomicSumRAndM<block_width, block_height>(
          depth_intrinsics_unknown_start_index + 4,
          cfactor * raw_inv_depth * jac_base,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M, &scalar_storage);
      // cfactor
      // NOTE: cannot do a block sum here as different threads may contribute to different pixels!
      __syncthreads();
      AtomicSumRAndM(
          depth_intrinsics_unknown_start_index + 5 + sparse_px + sparse_py * s.depth_params.cfactor_buffer.width(),
          -jac_base,  // Jacobian
          weight, raw_residual, visible, &pcg_r, &pcg_M);
    }
  }
  
  // TODO: It could be worth checking whether splitting the kernel here might
  //       improve performance (since perhaps one of the resulting parts could
  //       be executed with higher parallelism than the combined kernel).
  
  // --- Descriptor residual ---
  float2 color_pxy;
  if (use_descriptor_residuals) {
    visible = visible && TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy);
    
    // Compute residual
    float2 t1_pxy, t2_pxy;
    ComputeTangentProjections(
        r.surfel_global_position,
        r.surfel_normal,
        SurfelGetRadiusSquared(s.surfels, surfel_index),
        s.frame_T_global,
        color_corner_projector,
        &t1_pxy,
        &t2_pxy);
    
    float raw_residual_1;
    float raw_residual_2;
    ComputeRawDescriptorResidual(
        color_texture,
        color_pxy,
        t1_pxy,
        t2_pxy,
        s.surfels(kSurfelDescriptor1, ::min(surfel_index, s.surfels_size - 1)),
        s.surfels(kSurfelDescriptor2, ::min(surfel_index, s.surfels_size - 1)),
        &raw_residual_1,
        &raw_residual_2);
    
    float grad_x_fx_1;
    float grad_y_fy_1;
    float grad_x_fx_2;
    float grad_y_fy_2;
    DescriptorJacobianWrtProjectedPosition(
        color_texture, color_pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
    grad_x_fx_1 *= color_corner_projector.fx;
    grad_x_fx_2 *= color_corner_projector.fx;
    grad_y_fy_1 *= color_corner_projector.fy;
    grad_y_fy_2 *= color_corner_projector.fy;
    
    // Compute weight
    const PCGScalar weight_1 = ComputeDescriptorResidualWeight(raw_residual_1);
    const PCGScalar weight_2 = ComputeDescriptorResidualWeight(raw_residual_2);
    
    // Jacobians wrt. position changes and wrt. descriptor changes:
    if (visible && optimize_geometry) {
      const float term1 = -(rn.x*r.surfel_local_position.z - rn.z*r.surfel_local_position.x);
      const float term2 = -(rn.y*r.surfel_local_position.z - rn.z*r.surfel_local_position.y);
      const float term3 = 1.f / (r.surfel_local_position.z * r.surfel_local_position.z);
      float jacobian_wrt_position_1 = -(grad_x_fx_1 * term1 + grad_y_fy_1 * term2) * term3;
      float jacobian_wrt_position_2 = -(grad_x_fx_2 * term1 + grad_y_fy_2 * term2) * term3;
      
      pcg_r(0, surfel_unknown_start_index + 3 * surfel_index + 0) -=
          jacobian_wrt_position_1 * weight_1 * raw_residual_1 +
          jacobian_wrt_position_2 * weight_2 * raw_residual_2;
      pcg_M(0, surfel_unknown_start_index + 3 * surfel_index + 0) +=
          jacobian_wrt_position_1 * weight_1 * jacobian_wrt_position_1 +
          jacobian_wrt_position_2 * weight_2 * jacobian_wrt_position_2;
      
      constexpr PCGScalar jacobian_wrt_descriptor1_1 = -1;
      constexpr PCGScalar jacobian_wrt_descriptor1_2 = 0;
      pcg_r(0, surfel_unknown_start_index + 3 * surfel_index + 1) -=
          jacobian_wrt_descriptor1_1 * weight_1 * raw_residual_1 +
          jacobian_wrt_descriptor1_2 * weight_2 * raw_residual_2;
      pcg_M(0, surfel_unknown_start_index + 3 * surfel_index + 1) +=
          jacobian_wrt_descriptor1_1 * weight_1 * jacobian_wrt_descriptor1_1 +
          jacobian_wrt_descriptor1_2 * weight_2 * jacobian_wrt_descriptor1_2;
      
      constexpr PCGScalar jacobian_wrt_descriptor2_1 = 0;
      constexpr PCGScalar jacobian_wrt_descriptor2_2 = -1;
      pcg_r(0, surfel_unknown_start_index + 3 * surfel_index + 2) -=
          jacobian_wrt_descriptor2_1 * weight_1 * raw_residual_1 +
          jacobian_wrt_descriptor2_2 * weight_2 * raw_residual_2;
      pcg_M(0, surfel_unknown_start_index + 3 * surfel_index + 2) +=
          jacobian_wrt_descriptor2_1 * weight_1 * jacobian_wrt_descriptor2_1 +
          jacobian_wrt_descriptor2_2 * weight_2 * jacobian_wrt_descriptor2_2;
    }
    
    // Jacobian wrt. pose changes:
    if (optimize_poses) {
      float inv_ls_z = 1.f / r.surfel_local_position.z;
      float ls_z_sq = r.surfel_local_position.z * r.surfel_local_position.z;
      float inv_ls_z_sq = inv_ls_z * inv_ls_z;
      
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 0,
          -grad_x_fx_1 * inv_ls_z,  // Jacobian 1
          weight_1, raw_residual_1,
          -grad_x_fx_2 * inv_ls_z,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 1,
          -grad_y_fy_1 * inv_ls_z,  // Jacobian 1
          weight_1, raw_residual_1,
          -grad_y_fy_2 * inv_ls_z,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 2,
          (r.surfel_local_position.x * grad_x_fx_1 + r.surfel_local_position.y * grad_y_fy_1) * inv_ls_z_sq,  // Jacobian 1
          weight_1, raw_residual_1,
          (r.surfel_local_position.x * grad_x_fx_2 + r.surfel_local_position.y * grad_y_fy_2) * inv_ls_z_sq,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      
      float ls_x_y = r.surfel_local_position.x * r.surfel_local_position.y;
      
      const float term1 = r.surfel_local_position.y * r.surfel_local_position.y + ls_z_sq;
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 3,
          (term1 * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq,  // Jacobian 1
          weight_1, raw_residual_1,
          (term1 * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      const float term2 = r.surfel_local_position.x * r.surfel_local_position.x + ls_z_sq;
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 4,
          -(term2 * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq,  // Jacobian 1
          weight_1, raw_residual_1,
          -(term2 * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          kf_pose_unknown_index + 5,
          -(r.surfel_local_position.x * grad_y_fy_1 - r.surfel_local_position.y * grad_x_fx_1) * inv_ls_z,  // Jacobian 1
          weight_1, raw_residual_1,
          -(r.surfel_local_position.x * grad_y_fy_2 - r.surfel_local_position.y * grad_x_fx_2) * inv_ls_z,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
    }
    
    // Jacobian wrt. color intrinsics changes:
    if (optimize_color_intrinsics) {
      const float grad_x_1 = grad_x_fx_1 / color_corner_projector.fx;
      const float grad_y_1 = grad_y_fy_1 / color_corner_projector.fy;
      const float grad_x_2 = grad_x_fx_2 / color_corner_projector.fx;
      const float grad_y_2 = grad_y_fy_2 / color_corner_projector.fy;
      
      // Descriptor residual derivative wrt. ...
      // fx
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          color_intrinsics_unknown_start_index + 0,
          grad_x_1 * depth_center_unprojector.nx(r.px),  // Jacobian 1
          weight_1, raw_residual_1,
          grad_x_2 * depth_center_unprojector.nx(r.px),  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      // fy
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          color_intrinsics_unknown_start_index + 1,
          grad_y_1 * depth_center_unprojector.ny(r.py),  // Jacobian 1
          weight_1, raw_residual_1,
          grad_y_2 * depth_center_unprojector.ny(r.py),  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      // cx
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          color_intrinsics_unknown_start_index + 2,
          grad_x_1,  // Jacobian 1
          weight_1, raw_residual_1,
          grad_x_2,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
      // cy
      __syncthreads();
      BlockedAtomicSumRAndM2<block_width, block_height>(
          color_intrinsics_unknown_start_index + 3,
          grad_y_1,  // Jacobian 1
          weight_1, raw_residual_1,
          grad_y_2,  // Jacobian 2
          weight_2, raw_residual_2,
          visible, &pcg_r, &pcg_M, &scalar_storage);
    }
  }
}

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
    u32 surfels_size) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  COMPILE_OPTION_4(optimize_poses, optimize_geometry, optimize_depth_intrinsics, optimize_color_intrinsics,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          PCGInitCUDAKernel,
          1024,
          surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _optimize_poses, _optimize_geometry, _optimize_depth_intrinsics, _optimize_color_intrinsics),
          /* kernel parameters */
          s,
          depth_to_color,
          depth_unprojector,
          color_projector,
          color_texture,
          use_depth_residuals,  // TODO: Would be better as template parameter
          use_descriptor_residuals,  // TODO: Would be better as template parameter
          kf_pose_unknown_index,
          surfel_unknown_start_index,
          depth_intrinsics_unknown_start_index,
          color_intrinsics_unknown_start_index,
          *pcg_r,
          *pcg_M));
  CUDA_CHECK();
}

template<int block_width>
__global__ void PCGInit2CUDAKernel(
    u32 unknown_count,
    u32 a_unknown_index,
    float a,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M,
    CUDABuffer_<PCGScalar> pcg_delta,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
  
  PCGScalar alpha_term;
  
  if (unknown_index < unknown_count) {
    pcg_g(0, unknown_index) = 0;
    
    // p_0 = M^-1 r_0
    // Here, we add the prior term on the "a" depth deformation parameter if needed, both for pcg_r and pcg_M, since it hasn't been added before.
    // An alternative (in case this becomes too convoluted here) would be to use a separate kernel to add it beforehand.
    // The addition of kDiagEpsilon is also handled here.
    PCGScalar r_value = pcg_r(0, unknown_index) + ((unknown_index == a_unknown_index) ? (-kAPriorWeight * kAPriorWeight * a) : 0);
    PCGScalar p_value = r_value / (pcg_M(0, unknown_index) + kDiagEpsilon + ((unknown_index == a_unknown_index) ? (kAPriorWeight * kAPriorWeight) : 0));
    pcg_p(0, unknown_index) = p_value;
    
    // delta_0 = 0
    pcg_delta(0, unknown_index) = 0;
    
    // alpha_n_0 = r_0^T p_0
    alpha_term = r_value * p_value;
  }
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_alpha_n(0, 0),
      alpha_term,
      unknown_index < unknown_count,
      &scalar_storage);
}

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
    CUDABuffer_<PCGScalar>* pcg_alpha_n) {
  CUDA_CHECK();
  if (unknown_count == 0) {
    return;
  }
  
  cudaMemsetAsync(pcg_alpha_n->address(), 0, 1 * sizeof(PCGScalar), stream);
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGInit2CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      a_unknown_index,
      a,
      pcg_r,
      pcg_M,
      *pcg_delta,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n);
  CUDA_CHECK();
}

template<int block_width, bool optimize_poses, bool optimize_geometry, bool optimize_depth_intrinsics, bool optimize_color_intrinsics>
__global__ void PCGStep1CUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCenterUnprojector depth_center_unprojector,
    PixelCornerProjector color_corner_projector,
    cudaTextureObject_t color_texture,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    u32 kf_pose_unknown_index,
    u32 surfel_unknown_start_index,
    u32 depth_intrinsics_unknown_start_index,
    u32 color_intrinsics_unknown_start_index,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_alpha_d) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int have_visible;
  __shared__ int have_visible_2;
  bool visible;
  SurfelProjectionResult6 r;
  if (!AnySurfelProjectsToAssociatedPixel(&surfel_index, s, &have_visible, &have_visible_2, &visible, &r)) {
    return;
  }
  
  float3 rn = s.frame_T_global.Rotate(r.surfel_normal);
  
  // --- Depth residual ---
  if (use_depth_residuals) {
    // Compute residual
    PCGScalar depth_residual_inv_stddev =
        ComputeDepthResidualInvStddevEstimate(depth_center_unprojector.nx(r.px), depth_center_unprojector.ny(r.py), r.pixel_calibrated_depth, rn, s.depth_params.baseline_fx);
    float3 local_unproj;
    float raw_residual;
    ComputeRawDepthResidual(
        depth_center_unprojector, r.px, r.py, r.pixel_calibrated_depth,
        depth_residual_inv_stddev,
        r.surfel_local_position, rn, &local_unproj, &raw_residual);
    
    // Compute weight
    const PCGScalar weight = ComputeDepthResidualWeight(raw_residual);
    
    PCGScalar sum = 0;  // holds the result of (J * p) for the row of this residual.
    PCGScalar geometry_jacobian;
    PCGScalar pose_jacobian[6];
    PCGScalar depth_global_intrinsics_jacobian[5];
    PCGScalar cfactor_entry_jacobian;
    // PCGScalar jacobian[
    //     (optimize_geometry ? 1 : 0) +
    //     (optimize_poses ? 6 : 0) +
    //     (optimize_depth_intrinsics ? 7 : 0)];
    
    // Jacobian wrt. position changes:
    if (visible && optimize_geometry) {
      geometry_jacobian = -depth_residual_inv_stddev;
      sum += geometry_jacobian * pcg_p(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index + 0);
    }
    
    // Jacobian wrt. pose changes:
    if (visible && optimize_poses) {
      pose_jacobian[0] = depth_residual_inv_stddev * rn.x;
      sum += pose_jacobian[0] * pcg_p(0, kf_pose_unknown_index + 0);
      
      pose_jacobian[1] = depth_residual_inv_stddev * rn.y;
      sum += pose_jacobian[1] * pcg_p(0, kf_pose_unknown_index + 1);
      
      pose_jacobian[2] = depth_residual_inv_stddev * rn.z;
      sum += pose_jacobian[2] * pcg_p(0, kf_pose_unknown_index + 2);
      
      pose_jacobian[3] = depth_residual_inv_stddev * (-rn.y * local_unproj.z + rn.z * local_unproj.y);
      sum += pose_jacobian[3] * pcg_p(0, kf_pose_unknown_index + 3);
      
      pose_jacobian[4] = depth_residual_inv_stddev * ( rn.x * local_unproj.z - rn.z * local_unproj.x);
      sum += pose_jacobian[4] * pcg_p(0, kf_pose_unknown_index + 4);
      
      pose_jacobian[5] = depth_residual_inv_stddev * (-rn.x * local_unproj.y + rn.y * local_unproj.x);
      sum += pose_jacobian[5] * pcg_p(0, kf_pose_unknown_index + 5);
    }
    
    // Jacobian wrt. depth intrinsics changes:
    bool depth_intrinsics_jac_valid;
    u32 cfactor_entry_index;
    if (optimize_depth_intrinsics) {
      int sparse_px = r.px / s.depth_params.sparse_surfel_cell_size;
      int sparse_py = r.py / s.depth_params.sparse_surfel_cell_size;
      float cfactor = s.depth_params.cfactor_buffer(sparse_py, sparse_px);
      
      float raw_inv_depth = 1.0f / (s.depth_params.raw_to_float_depth * s.depth_buffer(r.py, r.px));  // TODO: SurfelProjectsToAssociatedPixel() also reads that value, could be gotten from there
      float exp_inv_depth = expf(- s.depth_params.a * raw_inv_depth);
      float corrected_inv_depth = cfactor * exp_inv_depth + raw_inv_depth;
      depth_intrinsics_jac_valid = visible && !(fabs(corrected_inv_depth) < 1e-4f);  // NOTE: Corresponds to 1000 meters
      
      if (depth_intrinsics_jac_valid) {
        float3 local_surfel_normal = s.frame_T_global.Rotate(r.surfel_normal);
        float nx = depth_center_unprojector.nx(r.px);
        float ny = depth_center_unprojector.ny(r.py);
        float dot = Dot(make_float3(nx, ny, 1), local_surfel_normal);
        
        float jac_base = depth_residual_inv_stddev * dot * exp_inv_depth / (corrected_inv_depth * corrected_inv_depth);
        
        // Depth residual derivative wrt. ...
        // cx_inv (Attention: notice the indexing order!)
        depth_global_intrinsics_jacobian[2] = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row0.x, s.frame_T_global.row0.y, s.frame_T_global.row0.z));
        sum += depth_global_intrinsics_jacobian[2] * pcg_p(0, depth_intrinsics_unknown_start_index + 2);
        
        // cy_inv
        depth_global_intrinsics_jacobian[3] = depth_residual_inv_stddev * r.pixel_calibrated_depth * Dot(r.surfel_normal, make_float3(s.frame_T_global.row1.x, s.frame_T_global.row1.y, s.frame_T_global.row1.z));
        sum += depth_global_intrinsics_jacobian[3] * pcg_p(0, depth_intrinsics_unknown_start_index + 3);
        
        // fx_inv
        depth_global_intrinsics_jacobian[0] = r.px * depth_global_intrinsics_jacobian[2];
        sum += depth_global_intrinsics_jacobian[0] * pcg_p(0, depth_intrinsics_unknown_start_index + 0);
        
        // fy_inv
        depth_global_intrinsics_jacobian[1] = r.py * depth_global_intrinsics_jacobian[3];
        sum += depth_global_intrinsics_jacobian[1] * pcg_p(0, depth_intrinsics_unknown_start_index + 1);
        
        // a
        depth_global_intrinsics_jacobian[4] = cfactor * raw_inv_depth * jac_base;
        sum += depth_global_intrinsics_jacobian[4] * pcg_p(0, depth_intrinsics_unknown_start_index + 4);
        
        // cfactor
        cfactor_entry_index = depth_intrinsics_unknown_start_index + 5 + sparse_px + sparse_py * s.depth_params.cfactor_buffer.width();
        cfactor_entry_jacobian = -jac_base;
        sum += cfactor_entry_jacobian * pcg_p(0, cfactor_entry_index);
      }
    }
    
    sum *= weight;
    
    constexpr int block_height = 1;
    typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
    __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
    
    PCGScalar alpha_d_term = 0;
    
    if (visible && optimize_geometry) {
      pcg_g(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index + 0) += geometry_jacobian * sum;
      alpha_d_term += geometry_jacobian * pcg_p(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index + 0);  // TODO: Cache value from pcg_p?
    }
    
    if (optimize_poses) {
      #pragma unroll
      for (int i = 0; i < 6; ++ i) {
        if (i > 0) {
          __syncthreads();
        }
        BlockedAtomicSum<block_width, block_height>(
            &pcg_g(0, kf_pose_unknown_index + i),
            pose_jacobian[i] * sum,
            visible, &scalar_storage);
        alpha_d_term += pose_jacobian[i] * pcg_p(0, kf_pose_unknown_index + i);  // TODO: Cache values from pcg_p?
      }
    }
    
    if (optimize_depth_intrinsics) {
      #pragma unroll
      for (int i = 0; i < 5; ++ i) {
        __syncthreads();
        BlockedAtomicSum<block_width, block_height>(
            &pcg_g(0, depth_intrinsics_unknown_start_index + i),
            depth_global_intrinsics_jacobian[i] * sum,
            visible && depth_intrinsics_jac_valid, &scalar_storage);
        alpha_d_term += depth_intrinsics_jac_valid ? (depth_global_intrinsics_jacobian[i] * pcg_p(0, depth_intrinsics_unknown_start_index + i)) : 0;  // TODO: Cache values from pcg_p?
      }
      if (visible && depth_intrinsics_jac_valid) {
        atomicAddFloatOrDouble(
            &pcg_g(0, cfactor_entry_index),
            cfactor_entry_jacobian * sum);
      }
      alpha_d_term += depth_intrinsics_jac_valid ? (cfactor_entry_jacobian * pcg_p(0, cfactor_entry_index)) : 0;  // TODO: Cache values from pcg_p?
    }
    
    __syncthreads();
    BlockedAtomicSum<block_width, block_height>(
        &pcg_alpha_d(0, 0), sum * alpha_d_term, visible, &scalar_storage);
  }
  
  // --- Descriptor residual ---
  float2 color_pxy;
  if (use_descriptor_residuals) {
    visible = visible && TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy);
    
    // Compute residual
    float2 t1_pxy, t2_pxy;
    ComputeTangentProjections(
        r.surfel_global_position,
        r.surfel_normal,
        SurfelGetRadiusSquared(s.surfels, surfel_index),
        s.frame_T_global,
        color_corner_projector,
        &t1_pxy,
        &t2_pxy);
    
    float raw_residual_1;
    float raw_residual_2;
    ComputeRawDescriptorResidual(
        color_texture,
        color_pxy,
        t1_pxy,
        t2_pxy,
        s.surfels(kSurfelDescriptor1, ::min(surfel_index, s.surfels_size - 1)),
        s.surfels(kSurfelDescriptor2, ::min(surfel_index, s.surfels_size - 1)),
        &raw_residual_1,
        &raw_residual_2);
    
    float grad_x_fx_1;
    float grad_y_fy_1;
    float grad_x_fx_2;
    float grad_y_fy_2;
    DescriptorJacobianWrtProjectedPosition(
        color_texture, color_pxy, t1_pxy, t2_pxy, &grad_x_fx_1, &grad_y_fy_1, &grad_x_fx_2, &grad_y_fy_2);
    grad_x_fx_1 *= color_corner_projector.fx;
    grad_x_fx_2 *= color_corner_projector.fx;
    grad_y_fy_1 *= color_corner_projector.fy;
    grad_y_fy_2 *= color_corner_projector.fy;
    
    // Compute weight
    const PCGScalar weight_1 = ComputeDescriptorResidualWeight(raw_residual_1);
    const PCGScalar weight_2 = ComputeDescriptorResidualWeight(raw_residual_2);
    
    PCGScalar sum_1 = 0;  // holds the result of (J * p) for the row of the first residual.
    PCGScalar sum_2 = 0;  // holds the result of (J * p) for the row of the second residual.
    PCGScalar geometry_jacobian_1;
    PCGScalar geometry_jacobian_2;
    constexpr PCGScalar descriptor1_jacobian_1 = -1;
    constexpr PCGScalar descriptor1_jacobian_2 = 0;
    constexpr PCGScalar descriptor2_jacobian_1 = 0;
    constexpr PCGScalar descriptor2_jacobian_2 = -1;
    PCGScalar pose_jacobian_1[6];
    PCGScalar pose_jacobian_2[6];
    PCGScalar color_intrinsics_jacobian_1[4];
    PCGScalar color_intrinsics_jacobian_2[4];
    
    // Jacobians wrt. position changes and wrt. descriptor changes:
    if (visible && optimize_geometry) {
      const float term1 = -(rn.x*r.surfel_local_position.z - rn.z*r.surfel_local_position.x);
      const float term2 = -(rn.y*r.surfel_local_position.z - rn.z*r.surfel_local_position.y);
      const float term3 = 1.f / (r.surfel_local_position.z * r.surfel_local_position.z);
      geometry_jacobian_1 = -(grad_x_fx_1 * term1 + grad_y_fy_1 * term2) * term3;
      geometry_jacobian_2 = -(grad_x_fx_2 * term1 + grad_y_fy_2 * term2) * term3;
      
      PCGScalar p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 0);
      sum_1 += geometry_jacobian_1 * p;
      sum_2 += geometry_jacobian_2 * p;
      
      // constexpr PCGScalar descriptor1_jacobian_1 = -1;  // this is constexpr and thus set above
      // constexpr PCGScalar descriptor1_jacobian_2 = 0;  // this is constexpr and thus set above
      p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 1);
      sum_1 += descriptor1_jacobian_1 * p;
      // descriptor1_jacobian_2 is zero
      
      // constexpr PCGScalar descriptor2_jacobian_1 = 0;  // this is constexpr and thus set above
      // constexpr PCGScalar descriptor2_jacobian_2 = -1;  // this is constexpr and thus set above
      p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 2);
      // descriptor2_jacobian_1 is zero
      sum_2 += descriptor2_jacobian_2 * p;
    }
    
    // Jacobian wrt. pose changes:
    if (visible && optimize_poses) {
      float inv_ls_z = 1.f / r.surfel_local_position.z;
      float ls_z_sq = r.surfel_local_position.z * r.surfel_local_position.z;
      float inv_ls_z_sq = inv_ls_z * inv_ls_z;
      
      PCGScalar p = pcg_p(0, kf_pose_unknown_index + 0);
      pose_jacobian_1[0] = -grad_x_fx_1 * inv_ls_z;
      sum_1 += pose_jacobian_1[0] * p;
      pose_jacobian_2[0] = -grad_x_fx_2 * inv_ls_z;
      sum_2 += pose_jacobian_2[0] * p;
      
      p = pcg_p(0, kf_pose_unknown_index + 1);
      pose_jacobian_1[1] = -grad_y_fy_1 * inv_ls_z;
      sum_1 += pose_jacobian_1[1] * p;
      pose_jacobian_2[1] = -grad_y_fy_2 * inv_ls_z;
      sum_2 += pose_jacobian_2[1] * p;
      
      p = pcg_p(0, kf_pose_unknown_index + 2);
      pose_jacobian_1[2] = (r.surfel_local_position.x * grad_x_fx_1 + r.surfel_local_position.y * grad_y_fy_1) * inv_ls_z_sq;
      sum_1 += pose_jacobian_1[2] * p;
      pose_jacobian_2[2] = (r.surfel_local_position.x * grad_x_fx_2 + r.surfel_local_position.y * grad_y_fy_2) * inv_ls_z_sq;
      sum_2 += pose_jacobian_2[2] * p;
      
      float ls_x_y = r.surfel_local_position.x * r.surfel_local_position.y;
      
      p = pcg_p(0, kf_pose_unknown_index + 3);
      const float term1 = r.surfel_local_position.y * r.surfel_local_position.y + ls_z_sq;
      pose_jacobian_1[3] = (term1 * grad_y_fy_1 + ls_x_y * grad_x_fx_1) * inv_ls_z_sq;
      sum_1 += pose_jacobian_1[3] * p;
      pose_jacobian_2[3] = (term1 * grad_y_fy_2 + ls_x_y * grad_x_fx_2) * inv_ls_z_sq;
      sum_2 += pose_jacobian_2[3] * p;
      
      p = pcg_p(0, kf_pose_unknown_index + 4);
      const float term2 = r.surfel_local_position.x * r.surfel_local_position.x + ls_z_sq;
      pose_jacobian_1[4] = -(term2 * grad_x_fx_1 + ls_x_y * grad_y_fy_1) * inv_ls_z_sq;
      sum_1 += pose_jacobian_1[4] * p;
      pose_jacobian_2[4] = -(term2 * grad_x_fx_2 + ls_x_y * grad_y_fy_2) * inv_ls_z_sq;
      sum_2 += pose_jacobian_2[4] * p;
      
      p = pcg_p(0, kf_pose_unknown_index + 5);
      pose_jacobian_1[5] = -(r.surfel_local_position.x * grad_y_fy_1 - r.surfel_local_position.y * grad_x_fx_1) * inv_ls_z;
      sum_1 += pose_jacobian_1[5] * p;
      pose_jacobian_2[5] = -(r.surfel_local_position.x * grad_y_fy_2 - r.surfel_local_position.y * grad_x_fx_2) * inv_ls_z;
      sum_2 += pose_jacobian_2[5] * p;
    }
    
    // Jacobian wrt. color intrinsics changes:
    if (visible && optimize_color_intrinsics) {
      const float grad_x_1 = grad_x_fx_1 / color_corner_projector.fx;
      const float grad_y_1 = grad_y_fy_1 / color_corner_projector.fy;
      const float grad_x_2 = grad_x_fx_2 / color_corner_projector.fx;
      const float grad_y_2 = grad_y_fy_2 / color_corner_projector.fy;
      
      // Descriptor residual derivative wrt. ...
      // fx
      PCGScalar p = pcg_p(0, color_intrinsics_unknown_start_index + 0);
      color_intrinsics_jacobian_1[0] = grad_x_1 * depth_center_unprojector.nx(r.px);
      sum_1 += color_intrinsics_jacobian_1[0] * p;
      color_intrinsics_jacobian_2[0] = grad_x_2 * depth_center_unprojector.nx(r.px);
      sum_2 += color_intrinsics_jacobian_2[0] * p;
      
      // fy
      p = pcg_p(0, color_intrinsics_unknown_start_index + 1);
      color_intrinsics_jacobian_1[1] = grad_y_1 * depth_center_unprojector.ny(r.py);
      sum_1 += color_intrinsics_jacobian_1[1] * p;
      color_intrinsics_jacobian_2[1] = grad_y_2 * depth_center_unprojector.ny(r.py);
      sum_2 += color_intrinsics_jacobian_2[1] * p;
      
      // cx
      p = pcg_p(0, color_intrinsics_unknown_start_index + 2);
      color_intrinsics_jacobian_1[2] = grad_x_1;
      sum_1 += color_intrinsics_jacobian_1[2] * p;
      color_intrinsics_jacobian_2[2] = grad_x_2;
      sum_2 += color_intrinsics_jacobian_2[2] * p;
      
      // cy
      p = pcg_p(0, color_intrinsics_unknown_start_index + 3);
      color_intrinsics_jacobian_1[3] = grad_y_1;
      sum_1 += color_intrinsics_jacobian_1[3] * p;
      color_intrinsics_jacobian_2[3] = grad_y_2;
      sum_2 += color_intrinsics_jacobian_2[3] * p;
    }
    
    sum_1 *= weight_1;
    sum_2 *= weight_2;
    
    constexpr int block_height = 1;
    typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
    __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
    
    PCGScalar alpha_d_term_1 = 0;
    PCGScalar alpha_d_term_2 = 0;
    
    // Jacobians wrt. position changes and wrt. descriptor changes:
    if (visible && optimize_geometry) {
      pcg_g(0, surfel_unknown_start_index + 3 * surfel_index + 0) +=
          geometry_jacobian_1 * sum_1 +
          geometry_jacobian_2 * sum_2;
      PCGScalar p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 0);  // TODO: Cache value from pcg_p?
      alpha_d_term_1 += geometry_jacobian_1 * p;
      alpha_d_term_2 += geometry_jacobian_2 * p;
      
      pcg_g(0, surfel_unknown_start_index + 3 * surfel_index + 1) +=
          descriptor1_jacobian_1 * sum_1 +
          descriptor1_jacobian_2 * sum_2;
      p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 1);  // TODO: Cache value from pcg_p?
      alpha_d_term_1 += descriptor1_jacobian_1 * p;
      // descriptor1_jacobian_2 is zero
      
      pcg_g(0, surfel_unknown_start_index + 3 * surfel_index + 2) +=
          descriptor2_jacobian_1 * sum_1 +
          descriptor2_jacobian_2 * sum_2;
      p = pcg_p(0, surfel_unknown_start_index + 3 * surfel_index + 2);  // TODO: Cache value from pcg_p?
      // descriptor2_jacobian_1 is zero
      alpha_d_term_2 += descriptor2_jacobian_2 * p;
    }
    
    if (optimize_poses) {
      #pragma unroll
      for (int i = 0; i < 6; ++ i) {
        if (i > 0) {
          __syncthreads();
        }
        BlockedAtomicSum<block_width, block_height>(
            &pcg_g(0, kf_pose_unknown_index + i),
            pose_jacobian_1[i] * sum_1 +
            pose_jacobian_2[i] * sum_2,
            visible, &scalar_storage);
        PCGScalar p = pcg_p(0, kf_pose_unknown_index + i);  // TODO: Cache values from pcg_p?
        alpha_d_term_1 += pose_jacobian_1[i] * p;
        alpha_d_term_2 += pose_jacobian_2[i] * p;
      }
    }
    
    if (optimize_color_intrinsics) {
      #pragma unroll
      for (int i = 0; i < 4; ++ i) {
        __syncthreads();
        BlockedAtomicSum<block_width, block_height>(
            &pcg_g(0, color_intrinsics_unknown_start_index + i),
            color_intrinsics_jacobian_1[i] * sum_1 +
            color_intrinsics_jacobian_2[i] * sum_2,
            visible, &scalar_storage);
        PCGScalar p = pcg_p(0, color_intrinsics_unknown_start_index + i);  // TODO: Cache values from pcg_p?
        alpha_d_term_1 += color_intrinsics_jacobian_1[i] * p;
        alpha_d_term_2 += color_intrinsics_jacobian_2[i] * p;
      }
    }
    
    __syncthreads();
    BlockedAtomicSum<block_width, block_height>(
        &pcg_alpha_d(0, 0), sum_1 * alpha_d_term_1 + sum_2 * alpha_d_term_2, visible, &scalar_storage);
  }
}

template<int block_width>
__global__ void AddAlphaDEpsilonTermsCUDAKernel(
    u32 unknown_count,
    u32 a_unknown_index,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_d) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = unknown_index < unknown_count;
  if (!valid) {
    unknown_index = unknown_count - 1;
  }
  
  PCGScalar p_value = pcg_p(0, unknown_index);
  PCGScalar term = (kDiagEpsilon + ((unknown_index == a_unknown_index) ? (kAPriorWeight * kAPriorWeight) : 0)) * p_value * p_value;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_alpha_d(0, 0), term, valid, &scalar_storage);
}

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
    u32 surfels_size) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  COMPILE_OPTION_4(optimize_poses, optimize_geometry, optimize_depth_intrinsics, optimize_color_intrinsics,
      CUDA_AUTO_TUNE_1D_TEMPLATED(
          PCGStep1CUDAKernel,
          512,
          surfels_size,
          0, stream,
          TEMPLATE_ARGUMENTS(block_width, _optimize_poses, _optimize_geometry, _optimize_depth_intrinsics, _optimize_color_intrinsics),
          /* kernel parameters */
          s,
          depth_to_color,
          depth_unprojector,
          color_projector,
          color_texture,
          use_depth_residuals,  // TODO: Would be better as template parameter
          use_descriptor_residuals,  // TODO: Would be better as template parameter
          kf_pose_unknown_index,
          surfel_unknown_start_index,
          depth_intrinsics_unknown_start_index,
          color_intrinsics_unknown_start_index,
          *pcg_p,
          *pcg_g,
          *pcg_alpha_d));
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      AddAlphaDEpsilonTermsCUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      a_unknown_index,
      *pcg_p,
      *pcg_alpha_d);
  CUDA_CHECK();
}

template<int block_width>
__global__ void PCGStep2CUDAKernel(
    u32 unknown_count,
    u32 a_unknown_index,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_M,
    CUDABuffer_<PCGScalar> pcg_delta,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n,
    CUDABuffer_<PCGScalar> pcg_alpha_d,
    CUDABuffer_<PCGScalar> pcg_beta_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  PCGScalar beta_term;
  
  constexpr int block_height = 1;
  typedef cub::BlockReduce<PCGScalar, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceScalar;
  __shared__ typename BlockReduceScalar::TempStorage scalar_storage;
  
  if (unknown_index < unknown_count) {
    // TODO: Default to 1 or to 0 if denominator is near-zero? Stop optimization if that happens?
    PCGScalar alpha =
        (pcg_alpha_d(0, 0) >= 1e-35f) ? (pcg_alpha_n(0, 0) / pcg_alpha_d(0, 0)) : 0;
    
//     if (unknown_index == 0) {
//       printf("alpha: %f  =  %f / %f\n", alpha, pcg_alpha_n(0, 0), pcg_alpha_d(0, 0));
//     }
    
    PCGScalar p_value = pcg_p(0, unknown_index);
    pcg_delta(0, unknown_index) += alpha * p_value;
    
    PCGScalar r_value = pcg_r(0, unknown_index);
    r_value -= alpha * (pcg_g(0, unknown_index) + (kDiagEpsilon + ((unknown_index == a_unknown_index) ? (kAPriorWeight * kAPriorWeight) : 0)) * p_value);
    pcg_r(0, unknown_index) = r_value;
    
    // This is called z in the Opt paper, but stored in g here to save memory.
    PCGScalar z_value = r_value / (pcg_M(0, unknown_index) + kDiagEpsilon + ((unknown_index == a_unknown_index) ? (kAPriorWeight * kAPriorWeight) : 0));
    pcg_g(0, unknown_index) = z_value;
    
    beta_term = z_value * r_value;
    
    //if (unknown_index < 20) {
      //printf("residual[%i]: %f\n", unknown_index, r_value);
      // printf("z_value: %f\n", z_value);
      // printf("beta_term: %f\n", beta_term);
    //}
  }
  
  BlockedAtomicSum<block_width, block_height>(
      &pcg_beta_n(0, 0),
      beta_term,
      unknown_index < unknown_count,
      &scalar_storage);
}

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
    CUDABuffer_<PCGScalar>* pcg_beta_n) {
  CUDA_CHECK();
  if (unknown_count == 0) {
    return;
  }
  
  cudaMemsetAsync(pcg_beta_n->address(), 0, 1 * sizeof(PCGScalar), stream);
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGStep2CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      a_unknown_index,
      pcg_r,
      pcg_M,
      *pcg_delta,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n,
      *pcg_alpha_d,
      *pcg_beta_n);
  CUDA_CHECK();
}

template<int block_width>
__global__ void PCGStep3CUDAKernel(
    u32 unknown_count,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_p,
    CUDABuffer_<PCGScalar> pcg_alpha_n,
    CUDABuffer_<PCGScalar> pcg_beta_n) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (unknown_index < unknown_count) {
    // TODO: Default to 1 or to 0 if denominator is near-zero? Stop optimization if that happens?
    PCGScalar beta =
        (pcg_alpha_n(0, 0) >= 1e-35f) ? (pcg_beta_n(0, 0) / pcg_alpha_n(0, 0)) : 0;
    
//     if (unknown_index == 0) {
//       printf("beta: %f  =  %f / %f\n", beta, pcg_beta_n(0, 0), pcg_alpha_n(0, 0));
//     }
    
    pcg_p(0, unknown_index) = pcg_g/*z*/(0, unknown_index) + beta * pcg_p(0, unknown_index);
  }
}

void PCGStep3CUDA(
    cudaStream_t stream,
    u32 unknown_count,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_p,
    CUDABuffer_<PCGScalar>* pcg_alpha_n,
    CUDABuffer_<PCGScalar>* pcg_beta_n) {
  CUDA_CHECK();
  if (unknown_count == 0) {
    return;
  }
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGStep3CUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      *pcg_g,
      *pcg_p,
      *pcg_alpha_n,
      *pcg_beta_n);
  CUDA_CHECK();
}

template<int block_width>
__global__ void PCGDebugVerifyResultCUDAKernel(
    u32 unknown_count,
    u32 a_unknown_index,
    CUDABuffer_<PCGScalar> pcg_r,
    CUDABuffer_<PCGScalar> pcg_g,
    CUDABuffer_<PCGScalar> pcg_delta) {
  unsigned int unknown_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (unknown_index < unknown_count) {
    PCGScalar r_value = pcg_r(0, unknown_index);
    r_value -= pcg_g(0, unknown_index) + (kDiagEpsilon + ((unknown_index == a_unknown_index) ? (kAPriorWeight * kAPriorWeight) : 0)) * pcg_delta(0, unknown_index);
    
    // Output the PCG matrix solving residual if bad (should be zero if solved perfectly).
    if (fabs(r_value) > (PCGScalar)0.0001) {
      printf("r[%i]: %f\n", (int)unknown_index, r_value);
    }
  }
}

void PCGDebugVerifyResultCUDA(
    cudaStream_t stream,
    u32 unknown_count,
    u32 a_unknown_index,
    CUDABuffer_<PCGScalar>* pcg_r,
    CUDABuffer_<PCGScalar>* pcg_g,
    CUDABuffer_<PCGScalar>* pcg_delta) {
  CUDA_CHECK();
  if (unknown_count == 0) {
    return;
  }
  
  CUDA_AUTO_TUNE_1D_TEMPLATED(
      PCGDebugVerifyResultCUDAKernel,
      1024,
      unknown_count,
      0, stream,
      TEMPLATE_ARGUMENTS(block_width),
      /* kernel parameters */
      unknown_count,
      a_unknown_index,
      *pcg_r,
      *pcg_g,
      *pcg_delta);
  CUDA_CHECK();
}

template <bool use_descriptor_residuals>
__global__ void UpdateSurfelsFromPCGDeltaCUDAKernel(
    CUDABuffer_<float> surfels,
    u32 surfels_size,
    u32 surfel_unknown_start_index,
    CUDABuffer_<PCGScalar> pcg_delta) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    PCGScalar t = pcg_delta(0, surfel_unknown_start_index + (use_descriptor_residuals ? 3 : 1) * surfel_index);
    
    if (t != 0) {
      // Update surfel position
      float3 global_position = SurfelGetPosition(surfels, surfel_index);
      float3 surfel_normal = SurfelGetNormal(surfels, surfel_index);
      SurfelSetPosition(&surfels, surfel_index, global_position + t * surfel_normal);
    }
    
    if (use_descriptor_residuals) {
      // Update surfel descriptor
      float surfel_descriptor_1 = surfels(kSurfelDescriptor1, surfel_index);
      surfel_descriptor_1 += pcg_delta(0, surfel_unknown_start_index + 3 * surfel_index + 1);
      surfels(kSurfelDescriptor1, surfel_index) = ::max(-180.f, ::min(180.f, surfel_descriptor_1));
      
      float surfel_descriptor_2 = surfels(kSurfelDescriptor2, surfel_index);
      surfel_descriptor_2 += pcg_delta(0, surfel_unknown_start_index + 3 * surfel_index + 2);
      surfels(kSurfelDescriptor2, surfel_index) = ::max(-180.f, ::min(180.f, surfel_descriptor_2));
    }
  }
}

void UpdateSurfelsFromPCGDeltaCUDA(
    cudaStream_t stream,
    u32 surfels_size,
    CUDABuffer_<float>* surfels,
    bool use_descriptor_residuals,
    u32 surfel_unknown_start_index,
    const CUDABuffer_<PCGScalar>& pcg_delta) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  COMPILE_OPTION(use_descriptor_residuals, CUDA_AUTO_TUNE_1D_TEMPLATED(
      UpdateSurfelsFromPCGDeltaCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      TEMPLATE_ARGUMENTS(_use_descriptor_residuals),
      /* kernel parameters */
      *surfels,
      surfels_size,
      surfel_unknown_start_index,
      pcg_delta));
  CUDA_CHECK();
}

__global__ void UpdateCFactorsFromPCGDeltaCUDAKernel(
    u32 cfactor_unknown_start_index,
    CUDABuffer_<float> cfactor_buffer,
    CUDABuffer_<PCGScalar> pcg_delta) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < cfactor_buffer.width() * cfactor_buffer.height()) {
    u32 y = index / cfactor_buffer.width();
    u32 x = index - y * cfactor_buffer.width();
    
    cfactor_buffer(y, x) += pcg_delta(0, cfactor_unknown_start_index + index);
  }
}

void UpdateCFactorsFromPCGDeltaCUDA(
    cudaStream_t stream,
    CUDABuffer_<float>* cfactor_buffer,
    u32 cfactor_unknown_start_index,
    const CUDABuffer_<PCGScalar>& pcg_delta) {
  CUDA_AUTO_TUNE_1D(
      UpdateCFactorsFromPCGDeltaCUDAKernel,
      1024,
      cfactor_buffer->height() * cfactor_buffer->width(),
      0, stream,
      /* kernel parameters */
      cfactor_unknown_start_index,
      *cfactor_buffer,
      pcg_delta);
  CUDA_CHECK();
}

}
