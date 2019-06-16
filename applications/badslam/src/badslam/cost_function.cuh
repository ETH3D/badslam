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
#include <libvis/libvis.h>

#include "badslam/cuda_util.cuh"
#include "badslam/util.cuh"
#include "badslam/robust_weighting.cuh"

namespace vis {

// --- Depth (geometric) residual ---

// Weight factor on the depth residual in the cost term.
constexpr float kDepthResidualWeight = 1.f;

// Default Tukey parameter (= factor on standard deviation at which the
// residuals have zero weight). This gets scaled for multi-res pose estimation.
constexpr float kDepthResidualDefaultTukeyParam = 10.f;

// Expected stereo matching uncertainty in pixels in the depth estimation
// process. Determines the final propagated depth uncertainty.
constexpr float kDepthUncertaintyEmpiricalFactor = 0.1f;


// Computes the "raw" depth (geometric) residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawDepthResidual(
    const PixelCenterUnprojector& unprojector,
    int px,
    int py,
    float pixel_calibrated_depth,
    float raw_residual_inv_stddev_estimate,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    float3* local_unproj,
    float* raw_residual) {
  *local_unproj = unprojector.UnprojectPoint(px, py, pixel_calibrated_depth);
  *raw_residual = raw_residual_inv_stddev_estimate * Dot(surfel_local_normal, *local_unproj - surfel_local_position);
}

// Computes the "raw" depth (geometric) residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawDepthResidual(
    float raw_residual_inv_stddev_estimate,
    const float3& surfel_local_position,
    const float3& surfel_local_normal,
    const float3& local_unproj,
    float* raw_residual) {
  *raw_residual = raw_residual_inv_stddev_estimate * Dot(surfel_local_normal, local_unproj - surfel_local_position);
}

// Computes the propagated standard deviation estimate for the depth residual.
__forceinline__ __device__ float ComputeDepthResidualStddevEstimate(float nx, float ny, float depth, const float3& surfel_local_normal, float baseline_fx) {
  return (kDepthUncertaintyEmpiricalFactor * fabs(surfel_local_normal.x * nx + surfel_local_normal.y * ny + surfel_local_normal.z) * (depth * depth)) / baseline_fx;
}

// Computes the propagated inverse standard deviation estimate for the depth residual.
__forceinline__ __device__ float ComputeDepthResidualInvStddevEstimate(float nx, float ny, float depth, const float3& surfel_local_normal, float baseline_fx) {
  return baseline_fx / (kDepthUncertaintyEmpiricalFactor * fabs(surfel_local_normal.x * nx + surfel_local_normal.y * ny + surfel_local_normal.z) * (depth * depth));
}

// Computes the weight of the depth residual in the optimization.
__forceinline__ __device__ float ComputeDepthResidualWeight(float raw_residual, float scaling = 1.f) {
  return kDepthResidualWeight * TukeyWeight(raw_residual, scaling * kDepthResidualDefaultTukeyParam);
}

// Computes the weighted depth residual for summing up the optimization cost.
__forceinline__ __device__ float ComputeWeightedDepthResidual(float raw_residual, float scaling = 1.f) {
  return kDepthResidualWeight * TukeyResidual(raw_residual, scaling * kDepthResidualDefaultTukeyParam);
}


// --- Descriptor (photometric) residual ---

// Weight factor from the cost term.
// TODO: Tune further. Make parameter?
constexpr float kDescriptorResidualWeight = 1e-2f;

// Parameter for the Huber robust loss function for photometric residuals.
// TODO: Make parameter?
constexpr float kDescriptorResidualHuberParameter = 10.f;


// Computes the projections in an image of two (mostly) fixed points on the
// border of a surfel, whose direction to the surfel center differs by 90
// degrees. These points are used to compute the descriptor residual.
__forceinline__ __device__ void ComputeTangentProjections(
    const float3& surfel_global_position,
    const float3& surfel_global_normal,
    const float surfel_radius_squared,
    const CUDAMatrix3x4& frame_T_global,
    const PixelCornerProjector& color_corner_projector,
    float2* t1_pxy,
    float2* t2_pxy) {
  // With scaling 1, the tangent sample points are ca. 0.5 pixels away from the
  // center point when looking at the surfel from directly above.
  // TODO: Tune this! I think this has received very little tuning, if any at all.
  constexpr float kTangentScaling = 2.0f;
  
  float3 t1;
  CrossProduct(surfel_global_normal, (fabs(surfel_global_normal.x) > 0.9f) ? make_float3(0, 1, 0) : make_float3(1, 0, 0), &t1);
  t1 = t1 * kTangentScaling * sqrtf(surfel_radius_squared / max(1e-12f, SquaredLength(t1)));
  *t1_pxy = color_corner_projector.Project(frame_T_global * (surfel_global_position + t1));
  float3 t2;
  CrossProduct(surfel_global_normal, t1, &t2);
  t2 = t2 * kTangentScaling * sqrtf(surfel_radius_squared / max(1e-12f, SquaredLength(t2)));
  *t2_pxy = color_corner_projector.Project(frame_T_global * (surfel_global_position + t2));
}

// Computes the "raw" descriptor (photometric) residual, i.e., without any
// weighting.
__forceinline__ __device__ void ComputeRawDescriptorResidual(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2) {
  float intensity = tex2D<float4>(color_texture, pxy.x, pxy.y).w;
  
  float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w;
  float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  
  *raw_residual_1 = (180.f * (t1_intensity - intensity)) - surfel_descriptor_1;
  *raw_residual_2 = (180.f * (t2_intensity - intensity)) - surfel_descriptor_2;
}

__forceinline__ __device__ void ComputeRawDescriptorResidualWithFloatTexture(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float surfel_descriptor_1,
    float surfel_descriptor_2,
    float* raw_residual_1,
    float* raw_residual_2) {
  float intensity = tex2D<float>(color_texture, pxy.x, pxy.y);
  
  float t1_intensity = tex2D<float>(color_texture, t1_pxy.x, t1_pxy.y);
  float t2_intensity = tex2D<float>(color_texture, t2_pxy.x, t2_pxy.y);
  
  *raw_residual_1 = (180.f * (t1_intensity - intensity)) - surfel_descriptor_1;
  *raw_residual_2 = (180.f * (t2_intensity - intensity)) - surfel_descriptor_2;
}

// Computes the weight of the descriptor residual in the optimization.
__forceinline__ __device__ float ComputeDescriptorResidualWeight(float raw_residual, float scaling = 1.f) {
  return scaling * kDescriptorResidualWeight * HuberWeight(raw_residual, kDescriptorResidualHuberParameter);
}

// Computes the weighted descriptor residual for summing up the optimization 
// cost.
__forceinline__ __device__ float ComputeWeightedDescriptorResidual(float raw_residual, float scaling = 1.f) {
  return scaling * kDescriptorResidualWeight * HuberResidual(raw_residual, kDescriptorResidualHuberParameter);
}

// Computes the Jacobian of a surfel descriptor with regard to changes in the
// projected pixel position of the surfel. This function makes the approximation that
// the projected positions of all points on the surfel move equally. This should
// be valid since those points should all be very close together.
__forceinline__ __device__ void DescriptorJacobianWrtProjectedPosition(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_1,
    float* grad_y_1,
    float* grad_x_2,
    float* grad_y_2) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  float top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  float bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  float bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 0.5f).w;
  top_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 0.5f).w;
  bottom_left = tex2D<float4>(color_texture, ix + 0.5f, iy + 1.5f).w;
  bottom_right = tex2D<float4>(color_texture, ix + 1.5f, iy + 1.5f).w;
  
  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  float intensity = tex2D<float4>(color_texture, color_pxy.x, color_pxy.y).w;
  float t1_intensity = tex2D<float4>(color_texture, t1_pxy.x, t1_pxy.y).w;
  float t2_intensity = tex2D<float4>(color_texture, t2_pxy.x, t2_pxy.y).w;
  
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_1 = 180.f * (t1_dx - center_dx);
  *grad_y_1 = 180.f * (t1_dy - center_dy);
  *grad_x_2 = 180.f * (t2_dx - center_dx);
  *grad_y_2 = 180.f * (t2_dy - center_dy);
}

__forceinline__ __device__ void DescriptorJacobianWrtProjectedPositionWithFloatTexture(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    const float2& t1_pxy,
    const float2& t2_pxy,
    float* grad_x_fx_1,
    float* grad_y_fy_1,
    float* grad_x_fx_2,
    float* grad_y_fy_2) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  float top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  float bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  float bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float center_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float center_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  

  ix = static_cast<int>(::max(0.f, t1_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t1_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t1_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t1_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float t1_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t1_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  
  ix = static_cast<int>(::max(0.f, t2_pxy.x - 0.5f));
  iy = static_cast<int>(::max(0.f, t2_pxy.y - 0.5f));
  tx = ::max(0.f, ::min(1.f, t2_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  ty = ::max(0.f, ::min(1.f, t2_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  top_left = tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  top_right = tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  bottom_left = tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  bottom_right = tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  float t2_dx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  float t2_dy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
  
  float intensity = tex2D<float>(color_texture, color_pxy.x, color_pxy.y);
  float t1_intensity = tex2D<float>(color_texture, t1_pxy.x, t1_pxy.y);
  float t2_intensity = tex2D<float>(color_texture, t2_pxy.x, t2_pxy.y);
  
  // NOTE: It is approximate to mix all the center, t1, t2 derivatives
  //       directly since the points would move slightly differently on most
  //       pose changes. However, the approximation is possibly pretty good since
  //       the points are all close to each other.
  
  *grad_x_fx_1 = 180.f * (t1_dx - center_dx);
  *grad_y_fy_1 = 180.f * (t1_dy - center_dy);
  *grad_x_fx_2 = 180.f * (t2_dx - center_dx);
  *grad_y_fy_2 = 180.f * (t2_dy - center_dy);
}


// --- Color (photometric) residual for frame-to-frame tracking on precomputed gradient magnitudes ---

// Computes the "raw" color residual, i.e., without any weighting.
__forceinline__ __device__ void ComputeRawColorResidual(
    cudaTextureObject_t color_texture,
    const float2& pxy,
    float surfel_gradmag,
    float* raw_residual) {
  *raw_residual = 255.f * tex2D<float>(color_texture, pxy.x, pxy.y) - surfel_gradmag;
}

// Computes the Jacobian of the color residual with regard to changes in the
// projected position of a 3D point.
__forceinline__ __device__ void ColorJacobianWrtProjectedPosition(
    cudaTextureObject_t color_texture,
    const float2& color_pxy,
    float* grad_x_fx,
    float* grad_y_fy) {
  int ix = static_cast<int>(::max(0.f, color_pxy.x - 0.5f));
  int iy = static_cast<int>(::max(0.f, color_pxy.y - 0.5f));
  float tx = ::max(0.f, ::min(1.f, color_pxy.x - 0.5f - ix));  // truncated x = trunc(cx + fx*ls.x/ls.z)
  float ty = ::max(0.f, ::min(1.f, color_pxy.y - 0.5f - iy));  // truncated y = trunc(cy + fy*ls.y/ls.z)
  
  float top_left = 255.f * tex2D<float>(color_texture, ix + 0.5f, iy + 0.5f);
  float top_right = 255.f * tex2D<float>(color_texture, ix + 1.5f, iy + 0.5f);
  float bottom_left = 255.f * tex2D<float>(color_texture, ix + 0.5f, iy + 1.5f);
  float bottom_right = 255.f * tex2D<float>(color_texture, ix + 1.5f, iy + 1.5f);
  
  *grad_x_fx = (bottom_right - bottom_left) * ty + (top_right - top_left) * (1 - ty);
  *grad_y_fy = (bottom_right - top_right) * tx + (bottom_left - top_left) * (1 - tx);
}

}
