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

#include "libvis/cuda/cuda_buffer.cuh"

namespace vis {

// Helper for point projection using the "pixel corner" origin convention, in CUDA code.
struct PixelCornerProjector_ {
  PixelCornerProjector_() = default;
  
  // Host-only copy (should not run on device since it would be inefficient)
  __host__ PixelCornerProjector_(const PixelCornerProjector_& other)
      : resolution_x(other.resolution_x),
        resolution_y(other.resolution_y),
        min_nx(other.min_nx),
        min_ny(other.min_ny),
        max_nx(other.max_nx),
        max_ny(other.max_ny),
        grid(other.grid),
        type(other.type) {}
//       : fx(other.fx), fy(other.fy), cx(other.cx), cy(other.cy),
//         k1(other.k1), k2(other.k2), k3(other.k3), k4(other.k4),
//         p1(other.p1), p2(other.p2), sx1(other.sx1), sy1(other.sy1), type(other.type) {}
  
  __forceinline__ __device__ float2 CubicHermiteSpline(
      const float2& p0,
      const float2& p1,
      const float2& p2,
      const float2& p3,
      const float x) const {
    const float2 a = make_float2(
        static_cast<float>(0.5) * (-p0.x + static_cast<float>(3.0) * p1.x - static_cast<float>(3.0) * p2.x + p3.x),
        static_cast<float>(0.5) * (-p0.y + static_cast<float>(3.0) * p1.y - static_cast<float>(3.0) * p2.y + p3.y));
    const float2 b = make_float2(
        static_cast<float>(0.5) * (static_cast<float>(2.0) * p0.x - static_cast<float>(5.0) * p1.x + static_cast<float>(4.0) * p2.x - p3.x),
        static_cast<float>(0.5) * (static_cast<float>(2.0) * p0.y - static_cast<float>(5.0) * p1.y + static_cast<float>(4.0) * p2.y - p3.y));
    const float2 c = make_float2(
        static_cast<float>(0.5) * (-p0.x + p2.x),
        static_cast<float>(0.5) * (-p0.y + p2.y));
    const float2 d = p1;
    
    // Use Horner's rule to evaluate the function value and its
    // derivative.
    
    // f = ax^3 + bx^2 + cx + d
    return make_float2(
        d.x + x * (c.x + x * (b.x + x * a.x)),
        d.y + x * (c.y + x * (b.y + x * a.y)));
  }
  
  // Assumes that position.z > 0.
  __forceinline__ __device__ float2 Project(float3 position) const {
    // NOTE: As above, commented out for shorter compile times.
//     if (type == Camera::Type::kPinholeCamera4f) {
//       return make_float2(fx * (position.x / position.z) + cx,
//                          fy * (position.y / position.z) + cy);
//     } else { // if (type == Camera::Type::kRadtanCamera8d) {
//       float2 undistorted_point = make_float2(position.x / position.z,
//                                              position.y / position.z);
//       const float mx2_u = undistorted_point.x * undistorted_point.x;
//       const float my2_u = undistorted_point.y * undistorted_point.y;
//       const float mxy_u = undistorted_point.x * undistorted_point.y;
//       const float rho2_u = mx2_u + my2_u;
//       const float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
//       float2 distorted_point = make_float2(undistorted_point.x + undistorted_point.x * rad_dist_u + 2.0f * p1 * mxy_u + p2 * (rho2_u + 2.0f * mx2_u),
//                                            undistorted_point.y + undistorted_point.y * rad_dist_u + 2.0f * p2 * mxy_u + p1 * (rho2_u + 2.0f * my2_u));
//       return make_float2(fx * distorted_point.x + cx,
//                          fy * distorted_point.y + cy);
//     }
    
// //     if (type == Camera::Type::kThinPrismFisheyeCamera12d) {
//       float2 undistorted_nxy = make_float2(position.x / position.z,
//                                            position.y / position.z);
//       
//       float r = sqrtf(undistorted_nxy.x * undistorted_nxy.x + undistorted_nxy.y * undistorted_nxy.y);
//       
// //       if (r > radius_cutoff_) {
// //         return Eigen::Vector2f((undistorted_nxy.x < 0) ? -100 : 100,
// //                           (undistorted_nxy.y < 0) ? -100 : 100);
// //       }
//       
//       float fisheye_x, fisheye_y;
//       const float kEpsilon = static_cast<float>(1e-6);
//       if (r > kEpsilon) {
//         float theta_by_r = atanf(r) / r;
//         fisheye_x = theta_by_r * undistorted_nxy.x;
//         fisheye_y = theta_by_r * undistorted_nxy.y;
//       } else {
//         fisheye_x = undistorted_nxy.x;
//         fisheye_y = undistorted_nxy.y;
//       }
//       
//       const float x2 = fisheye_x * fisheye_x;
//       const float xy = fisheye_x * fisheye_y;
//       const float y2 = fisheye_y * fisheye_y;
//       const float r2 = x2 + y2;
//       const float r4 = r2 * r2;
//       const float r6 = r4 * r2;
//       const float r8 = r6 * r2;
//       
//       const float radial =
//           k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
//       const float dx = static_cast<float>(2) * p1 * xy + p2 * (r2 + static_cast<float>(2) * x2) + sx1 * r2;
//       const float dy = static_cast<float>(2) * p2 * xy + p1 * (r2 + static_cast<float>(2) * y2) + sy1 * r2;
//       
//       float nx = fisheye_x + radial * fisheye_x + dx;
//       float ny = fisheye_y + radial * fisheye_y + dy;
//       
//       return make_float2(fx * nx + cx,
//                          fy * ny + cy);
// //     }
    
    // For nonparametric bicubic projection camera:
    float2 undistorted_nxy = make_float2(position.x / position.z,
                                         position.y / position.z);
    
    float fc = (undistorted_nxy.x - min_nx) * ((resolution_x - 1) / (max_nx - min_nx));
    float fr = (undistorted_nxy.y - min_ny) * ((resolution_y - 1) / (max_ny - min_ny));
    const int row = ::floor(fr);
    const int col = ::floor(fc);
    float r_frac = fr - row;
    float c_frac = fc - col;
    
    int c[4];
    int r[4];
    for (int i = 0; i < 4; ++ i) {
      c[i] = min(max(0, col - 1 + i), resolution_x - 1);
      r[i] = min(max(0, row - 1 + i), resolution_y - 1);
    }
    
    float2 f[4];
    for (int wrow = 0; wrow < 4; ++ wrow) {
      float2 p0 = grid(r[wrow], c[0]);
      float2 p1 = grid(r[wrow], c[1]);
      float2 p2 = grid(r[wrow], c[2]);
      float2 p3 = grid(r[wrow], c[3]);
      
      f[wrow] = CubicHermiteSpline(p0, p1, p2, p3, c_frac);
    }
    
    return CubicHermiteSpline(f[0], f[1], f[2], f[3], r_frac);
  }
  
  int resolution_x;
  int resolution_y;
  float min_nx;
  float min_ny;
  float max_nx;
  float max_ny;
  
  CUDABuffer_<float2> grid;
  
//   float fx, fy, cx, cy;
//   float k1, k2, k3, k4, p1, p2;
//   float sx1, sy1;
  
  int type;  // from Camera::Type enum
  int width;
  int height;
};

}
