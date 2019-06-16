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

#include "badslam/cost_function.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/cuda_util.cuh"
#include "badslam/kernels.cuh"
#include "badslam/util.cuh"


// This file contains functions which can be compiled by the CUDA compiler (nvcc) only.
// It must not be included by .cc or .h files.

namespace vis {

// Sets the position of a surfel.
__forceinline__ __device__ void SurfelSetPosition(CUDABuffer_<float>* surfels, u32 surfel_index, float3 position) {
  (*surfels)(kSurfelX, surfel_index) = position.x;
  (*surfels)(kSurfelY, surfel_index) = position.y;
  (*surfels)(kSurfelZ, surfel_index) = position.z;
}

// Retrieves the position of a surfel.
__forceinline__ __device__ float3 SurfelGetPosition(const CUDABuffer_<float>& surfels, u32 surfel_index) {
  return make_float3(
      surfels(kSurfelX, surfel_index),
      surfels(kSurfelY, surfel_index),
      surfels(kSurfelZ, surfel_index));
}

// Converts a float (that must be in [-1, 1]) to a 10-bit value. Allows packing
// three floats into a 32-bit value. Also see TenBitSignedToSmallFloat().
__forceinline__ __device__ u32 SmallFloatToTenBitSigned(float value) {
  return 0x03ff & static_cast<u16>(static_cast<i16>(value * ((1 << 9) - 1) + ((value > 0) ? 0.5f : -0.5f)));
}

// Converts a 10-bit value to a float. Allows unpacking three floats from a
// 32-bit value. Also see SmallFloatToTenBitSigned().
__forceinline__ __device__ float TenBitSignedToSmallFloat(u32 value) {
  u16 temp = ((0x0200 & value) ? 0xfc00 : 0) | (0x03ff & value);
  return reinterpret_cast<i16&>(temp) * (1.0f / ((1 << 9) - 1));
}

// Sets the normal vector of a surfel.
__forceinline__ __device__ void SurfelSetNormal(CUDABuffer_<float>* surfels, u32 surfel_index, float3 normal) {
  *reinterpret_cast<u32*>(&((*surfels)(kSurfelNormal, surfel_index))) =
      (SmallFloatToTenBitSigned(normal.x) << 0) |
      (SmallFloatToTenBitSigned(normal.y) << 10) |
      (SmallFloatToTenBitSigned(normal.z) << 20);
}

// Retrieves the normal vector of a surfel.
__forceinline__ __device__ float3 SurfelGetNormal(const CUDABuffer_<float>& surfels, u32 surfel_index) {
  u32 value = *reinterpret_cast<const u32*>(&(surfels(kSurfelNormal, surfel_index)));
  float3 normal = make_float3(
      TenBitSignedToSmallFloat(value >> 0),
      TenBitSignedToSmallFloat(value >> 10),
      TenBitSignedToSmallFloat(value >> 20));
  float factor = 1.0f / Norm(normal);
  return factor * normal;
}

// Sets the squared radius of a surfel.
__forceinline__ __device__ void SurfelSetRadiusSquared(CUDABuffer_<float>* surfels, u32 surfel_index, float radius_squared) {
  (*surfels)(kSurfelRadiusSquared, surfel_index) = radius_squared;
}

// Retrieves the squared radius of a surfel.
__forceinline__ __device__ float SurfelGetRadiusSquared(const CUDABuffer_<float>& surfels, u32 surfel_index) {
  return surfels(kSurfelRadiusSquared, surfel_index);
}

// Sets the color of a surfel.
__forceinline__ __device__ void SurfelSetColor(CUDABuffer_<float>* surfels, u32 surfel_index, const uchar4& color) {
  *reinterpret_cast<uchar4*>(&((*surfels)(kSurfelColor, surfel_index))) = color;
}

// Retrieves the color of a surfel.
__forceinline__ __device__ uchar4 SurfelGetColor(const CUDABuffer_<float>& surfels, u32 surfel_index) {
  return *reinterpret_cast<const uchar4*>(&(surfels(kSurfelColor, surfel_index)));
}

}
