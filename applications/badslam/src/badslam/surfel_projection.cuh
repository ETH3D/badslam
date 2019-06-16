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
#include <libvis/cuda/cuda_buffer.cuh>

#include "badslam/cuda_matrix.cuh"

namespace vis {

// Helper for point projection using the "pixel corner" origin convention, in CUDA code.
struct PixelCornerProjector {
  __host__ PixelCornerProjector(float fx, float fy, float cx, float cy)
      : fx(fx), fy(fy), cx(cx), cy(cy) {}
  
  // Host-only copy (should not run on device since it would be inefficient)
  __host__ PixelCornerProjector(const PixelCornerProjector& other)
      : fx(other.fx),
        fy(other.fy),
        cx(other.cx),
        cy(other.cy) {}
  
  // Assumes that position.z > 0.
  __forceinline__ __device__ float2 Project(float3 position) const {
    return make_float2(fx * (position.x / position.z) + cx,
                       fy * (position.y / position.z) + cy);
  }
  
  float fx;
  float fy;
  float cx;
  float cy;
};

// Helper for point projection using the "pixel center" origin convention, in CUDA code.
struct PixelCenterProjector {
  __host__ PixelCenterProjector(float fx, float fy, float cx, float cy)
      : fx(fx), fy(fy), cx(cx), cy(cy) {}
  
  // Host-only copy (should not run on device since it would be inefficient)
  __host__ PixelCenterProjector(const PixelCenterProjector& other)
      : fx(other.fx),
        fy(other.fy),
        cx(other.cx),
        cy(other.cy) {}
  
  // Assumes that position.z > 0.
  __forceinline__ __device__ float2 Project(float3 position) const {
    return make_float2(fx * (position.x / position.z) + cx,
                       fy * (position.y / position.z) + cy);
  }
  
  float fx;
  float fy;
  float cx;
  float cy;
};

// Helper for point unprojection using the "pixel center" origin convention, in CUDA code.
struct PixelCenterUnprojector {
  __host__ PixelCenterUnprojector(float fx_inv, float fy_inv, float cx_inv, float cy_inv)
      : fx_inv(fx_inv), fy_inv(fy_inv), cx_inv(cx_inv), cy_inv(cy_inv) {}
  
  __host__ PixelCenterUnprojector(const PixelCornerProjector& corner_projector) {
    fx_inv = 1.0f / corner_projector.fx;
    fy_inv = 1.0f / corner_projector.fy;
    const float cx_pixel_center = corner_projector.cx - 0.5f;
    const float cy_pixel_center = corner_projector.cy - 0.5f;
    cx_inv = -cx_pixel_center * fx_inv;
    cy_inv = -cy_pixel_center * fy_inv;
  }
  
  // Host-only copy (should not run on device since it would be inefficient)
  __host__ PixelCenterUnprojector(const PixelCenterUnprojector& other)
      : fx_inv(other.fx_inv),
        fy_inv(other.fy_inv),
        cx_inv(other.cx_inv),
        cy_inv(other.cy_inv) {}
  
  __forceinline__ __device__ float3 UnprojectPoint(int x, int y, float depth) const {
    return make_float3(depth * (fx_inv * x + cx_inv),
                       depth * (fy_inv * y + cy_inv),
                       depth);
  }
  
  __forceinline__ __device__ float nx(float px) const {
    return fx_inv * px + cx_inv;
  }
  
  __forceinline__ __device__ float ny(float py) const {
    return fy_inv * py + cy_inv;
  }
  
  float fx_inv;
  float fy_inv;
  float cx_inv;
  float cy_inv;
};

// Common parameters for depth which are passed to many functions.
struct DepthParameters {
  // Parameter image of "c" factors (corresponding to "c" in depth distortion
  // compensation, which is actually called D_\delta in the BAD SLAM paper).
  CUDABuffer_<float> cfactor_buffer;
  
  // Factor \alpha_1 in depth distortion compensation.
  float a;
  
  // Factor which is applied to the raw depth values (of type unsigned short) to
  // obtain metric depth values in meters.
  float raw_to_float_depth;
  
  // The baseline (in meters) times the focal length (in pixels) of the
  // stereo system which was used to estimate the input depth images. Used
  // to estimate the depth uncertainty.
  float baseline_fx;
  
  // Surfel sparsification grid cell size. A cell size of 1 leads to fully dense
  // surfel creation, 2 creates surfels for one quarter of the pixels only, etc.
  int sparse_surfel_cell_size;
};

// Groups common parameters that are used for projecting surfels to images.
struct SurfelProjectionParameters {
  __host__ SurfelProjectionParameters(
      CUDABuffer_<float> surfels,
      CUDABuffer_<u16> depth_buffer,
      CUDABuffer_<u16> normals_buffer,
      DepthParameters depth_params,
      PixelCornerProjector projector,
      PixelCenterUnprojector center_unprojector,
      CUDAMatrix3x4 frame_T_global,
      u32 surfels_size)
      : surfels(surfels),
        depth_buffer(depth_buffer),
        normals_buffer(normals_buffer),
        depth_params(depth_params),
        projector(projector),
        center_unprojector(center_unprojector),
        frame_T_global(frame_T_global),
        surfels_size(surfels_size) {}
  
  CUDABuffer_<float> surfels;
  CUDABuffer_<u16> depth_buffer;
  CUDABuffer_<u16> normals_buffer;
  DepthParameters depth_params;
  PixelCornerProjector projector;
  PixelCenterUnprojector center_unprojector;
  CUDAMatrix3x4 frame_T_global;
  u32 surfels_size;
};

// Helper struct which stores parameters for transforming depth image pixel
// coordinates to color image pixel coordinates, using the "pixel corner" origin
// convention.
struct DepthToColorPixelCorner {
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;
};

// Transforms depth image pixel coordinates to color image pixel coordinates,
// using the "pixel corner" origin convention.
// TODO: Make member function?
__forceinline__ __device__ bool TransformDepthToColorPixelCorner(
    const float2& pxy,
    const DepthToColorPixelCorner& depth_to_color,
    float2* color_pxy) {
  color_pxy->x = depth_to_color.fx * pxy.x + depth_to_color.cx;
  color_pxy->y = depth_to_color.fy * pxy.y + depth_to_color.cy;
  
  return color_pxy->x >= 0 &&
         color_pxy->y >= 0 &&
         static_cast<int>(color_pxy->x) < depth_to_color.width &&
         static_cast<int>(color_pxy->y) < depth_to_color.height;
}

}
