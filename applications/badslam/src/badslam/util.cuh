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

#include <libvis/cuda/cuda_buffer.cuh>
#include <libvis/libvis.h>
#include <libvis/opengl.h>

#include "badslam/cuda_matrix.cuh"
#include "badslam/cuda_util.cuh"
#include "badslam/surfel_projection.cuh"

namespace vis {

/// Type conversion operator for CUB.
///
/// Used to convert input types such as u8 to for example u32 for device scans
/// in order to force CUB's internal accumulator variable(s) to have this type, preventing overflows.
///
/// Use within a TransformInputIterator like this:
///   cub::TransformInputIterator<TargetT, TypeConversionOp<TargetT>, const InputT*> inIt(src, TypeConversionOp<TargetT>());
template <typename TargetT>
struct TypeConversionOp {
  template <typename InputT>
  __forceinline__ __host__ __device__
  TargetT operator()(const InputT& v) const {
    return static_cast<TargetT>(v);
  }
};

// Converts raw (measured) depth values to calibrated depth values, given the
// depth deformation parameters and the raw-to-metric depth factor.
__forceinline__ __host__ __device__ float RawToCalibratedDepth(
    float a,
    float cfactor,
    float raw_to_float_depth,
    u16 measured_depth) {
  const float inv_depth = 1.0f / (raw_to_float_depth * measured_depth);
  return 1.f / (inv_depth + cfactor * expf(- a * inv_depth));
}

// Version of RawToCalibratedDepth() that takes metric depth (in meters) as
// input.
__forceinline__ __host__ __device__ float RawToCalibratedDepth(
    float a,
    float cfactor,
    float metric_depth) {
  const float inv_depth = 1.0f / metric_depth;
  return 1.f / (inv_depth + cfactor * expf(- a * inv_depth));
}

// Projects a point into an image. Returns true if it projects into the image
// bounds, false otherwise. Assumes that surfel_local_position.z > 0.
__forceinline__ __device__ bool ProjectSurfelToImage(
    int width, int height,
    const PixelCornerProjector& projector,
    float3 surfel_local_position,
    int* __restrict__ px, int* __restrict__ py) {
  float2 pixel_pos = projector.Project(surfel_local_position);
  *px = static_cast<int>(pixel_pos.x);
  *py = static_cast<int>(pixel_pos.y);
  if (pixel_pos.x < 0 || pixel_pos.y < 0 ||
      // *px < 0 || *py < 0 ||
      *px >= width || *py >= height) {
    return false;
  }
  
  return true;
}

// Projects a point into an image. Returns true if it projects into the image
// bounds, false otherwise. Assumes that surfel_local_position.z > 0.
__forceinline__ __device__ bool ProjectSurfelToImage(
    int width, int height,
    const PixelCornerProjector& projector,
    float3 surfel_local_position,
    int* __restrict__ px, int* __restrict__ py,
    float2* __restrict__ pixel_pos) {
  *pixel_pos = projector.Project(surfel_local_position);
  *px = static_cast<int>(pixel_pos->x);
  *py = static_cast<int>(pixel_pos->y);
  if (pixel_pos->x < 0 || pixel_pos->y < 0 ||
      // *px < 0 || *py < 0 ||
      *px >= width || *py >= height) {
    return false;
  }
  
  return true;
}

// Converts a float value in [-1, 1] to an 8-bit integer to save memory.
__forceinline__ __device__ i8 SmallFloatToEightBitSigned(float value) {
  return static_cast<i8>(value * ((1 << 7) - 1) + ((value > 0) ? 0.5f : -0.5f));
}

// Inverse of SmallFloatToEightBitSigned().
__forceinline__ __device__ float EightBitSignedToSmallFloat(i8 value) {
  return value * (1.0f / ((1 << 7) - 1));
}

// Converts an image-space normal (assumed to point towards the image, such that
// specifying its x and y components is sufficient to define it) to a 16-bit
// value to save memory.
__forceinline__ __device__ u16 ImageSpaceNormalToU16(float x, float y) {
  return (static_cast<u16>(static_cast<u8>(SmallFloatToEightBitSigned(x))) << 0) |
         (static_cast<u16>(static_cast<u8>(SmallFloatToEightBitSigned(y))) << 8);
}

// Inverse of ImageSpaceNormalToU16().
__forceinline__ __device__ float3 U16ToImageSpaceNormal(u16 value) {
  float3 result;
  result.x = EightBitSignedToSmallFloat(value & 0x00ff);
  result.y = EightBitSignedToSmallFloat((value & 0xff00) >> 8);
  result.z = 1 - result.x * result.x - result.y * result.y;
  result.z = -sqrtf((result.z > 0.f) ? result.z : 0.f);
  return result;
}

// Converts a 10-bit value to a float (allows unpacking three floats from a
// 32-bit value).
__forceinline__ float TenBitSignedToFloat(u32 value) {
  u16 temp = ((0x0200 & value) ? 0xfc00 : 0) | (0x03ff & value);
  return reinterpret_cast<i16&>(temp) * (1.0f / ((1 << 9) - 1));
}

}
