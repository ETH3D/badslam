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

#include "badslam/surfel_projection.cuh"

#include <cuda_runtime.h>
#include <libvis/camera.h>
#include <libvis/libvis.h>
#include <libvis/cuda/cuda_buffer.h>

#include "badslam/keyframe.h"

namespace vis {

inline PixelCornerProjector CreatePixelCornerProjector(const PinholeCamera4f& camera) {
  return PixelCornerProjector(
      camera.parameters()[0],
      camera.parameters()[1],
      camera.parameters()[2],
      camera.parameters()[3]);
}

inline PixelCenterProjector CreatePixelCenterProjector(const PinholeCamera4f& camera) {
  return PixelCenterProjector(
      camera.parameters()[0],
      camera.parameters()[1],
      camera.parameters()[2] - 0.5f,
      camera.parameters()[3] - 0.5f);
}

inline PixelCenterUnprojector CreatePixelCenterUnprojector(const PinholeCamera4f& camera) {
  float fx_inv = 1.0f / camera.parameters()[0];
  float fy_inv = 1.0f / camera.parameters()[1];
  const float cx_pixel_center = camera.parameters()[2] - 0.5f;
  const float cy_pixel_center = camera.parameters()[3] - 0.5f;
  return PixelCenterUnprojector(
      fx_inv, fy_inv,
      -cx_pixel_center * fx_inv,
      -cy_pixel_center * fy_inv);
}

inline SurfelProjectionParameters CreateSurfelProjectionParameters(
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    const Keyframe* keyframe) {
  return SurfelProjectionParameters(
      surfels.ToCUDA(),
      keyframe->depth_buffer().ToCUDA(),
      keyframe->normals_buffer().ToCUDA(),
      depth_params,
      CreatePixelCornerProjector(camera),
      CreatePixelCenterUnprojector(camera),
      keyframe->frame_T_global_cuda(),
      surfels_size);
}

inline SurfelProjectionParameters CreateSurfelProjectionParameters(
    const PinholeCamera4f& camera,
    const DepthParameters& depth_params,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    const CUDAMatrix3x4& frame_T_global) {
  return SurfelProjectionParameters(
      surfels.ToCUDA(),
      depth_buffer.ToCUDA(),
      normals_buffer.ToCUDA(),
      depth_params,
      CreatePixelCornerProjector(camera),
      CreatePixelCenterUnprojector(camera),
      frame_T_global,
      surfels_size);
}

inline DepthToColorPixelCorner CreateDepthToColorPixelCorner(
    const PinholeCamera4f& depth_camera,
    const PinholeCamera4f& color_camera) {
  DepthToColorPixelCorner result;
  
  result.width = color_camera.width();
  result.height = color_camera.height();
  
  // Dfx * nx + Dcx = Dpx
  // nx = 1 / Dfx * Dpx - Dcx / Dfx
  // Cpx = Cfx / Dfx * Dpx - Cfx * Dcx / Dfx + Ccx
  
  result.fx = color_camera.parameters()[0] / depth_camera.parameters()[0];
  result.cx = -1 * color_camera.parameters()[0] * depth_camera.parameters()[2] / depth_camera.parameters()[0] + color_camera.parameters()[2];
  
  result.fy = color_camera.parameters()[1] / depth_camera.parameters()[1];
  result.cy = -1 * color_camera.parameters()[1] * depth_camera.parameters()[3] / depth_camera.parameters()[1] + color_camera.parameters()[3];
  
  return result;
}

}
