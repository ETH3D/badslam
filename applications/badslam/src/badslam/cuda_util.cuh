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

namespace vis {

// Uncomment this to perform synchronous CUDA kernel execution and checking
// #define CUDA_SEQUENTIAL_CHECKS

#ifdef CUDA_SEQUENTIAL_CHECKS
  #define CUDA_CHECK() do {cudaDeviceSynchronize(); CHECK_CUDA_NO_ERROR();} while (false)
#else
  #define CUDA_CHECK() CHECK_CUDA_NO_ERROR()
#endif


// Computes the squared length of a 3-vector.
__forceinline__ __device__ float SquaredLength(const float3& vec) {
  return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
}

// Computes the dot product of two 3-vectors.
__forceinline__ __device__ float Dot(const float3& a, const float3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Computes the squared distance between two 3D points.
__forceinline__ __device__ float SquaredDistance(const float3& a, const float3& b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

// Replacement for std::swap in CUDA device code.
template<typename T>
__forceinline__ __device__ void Swap(T& a, T& b) {
  T temp = a;
  a = b;
  b = temp;
}

// Computes the cross produce between two 3-vectors.
__forceinline__ __device__ void CrossProduct(const float3& a, const float3& b, float3* result) {
  *result = make_float3(a.y * b.z - b.y * a.z,
                        b.x * a.z - a.x * b.z,
                        a.x * b.y - b.x * a.y);
}

// Computes the L2-norm of a 3-vector.
__forceinline__ __device__ float Norm(const float3& vec) {
#ifdef __CUDA_ARCH__
  return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
#else
  (void) vec;
  return -1;  // We can neither use std::sqrtf nor ::sqrtf here. Either nvcc or the host compiler will complain.
#endif
}

// Define some of the missing operators for float3 vectors.

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator*(float m, const float3& b) {
  return make_float3(m * b.x, m * b.y, m * b.z);
}

__forceinline__ __device__ float3 operator*(const float3& b, float m) {
  return make_float3(m * b.x, m * b.y, m * b.z);
}

}
