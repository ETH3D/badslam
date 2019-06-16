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

namespace vis {

// Stores a 3x3 matrix. Supports multiplication with float3 vectors.
struct CUDAMatrix3x3 {
  // Default constructor, leaves the matrix uninitialized.
  __forceinline__ __host__ __device__ CUDAMatrix3x3() {}
  
  // Copy constructor.
  __forceinline__ __host__ CUDAMatrix3x3(const CUDAMatrix3x3& other)
      : row0(other.row0),
        row1(other.row1),
        row2(other.row2) {}
  
  // Constructs the matrix from an array-like matrix object (works with Eigen matrices).
  template <typename T> __host__ explicit
  CUDAMatrix3x3(const T& matrix) {
    row0.x = matrix(0, 0);
    row0.y = matrix(0, 1);
    row0.z = matrix(0, 2);
    row1.x = matrix(1, 0);
    row1.y = matrix(1, 1);
    row1.z = matrix(1, 2);
    row2.x = matrix(2, 0);
    row2.y = matrix(2, 1);
    row2.z = matrix(2, 2);
  }
  
  // Matrix-vector multiplication.
  __forceinline__ __host__ __device__
  float3 operator* (const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z,
        row1.x * point.x + row1.y * point.y + row1.z * point.z,
        row2.x * point.x + row2.y * point.y + row2.z * point.z);
  }
  
  // Row-wise storage.
  float3 row0;
  float3 row1;
  float3 row2;
};

// Stores a 3x4 matrix. Supports homogeneous multiplication with float3 vectors.
struct CUDAMatrix3x4 {
  // Default constructor, leaves the matrix uninitialized.
  __forceinline__ __host__ __device__ CUDAMatrix3x4() {}
  
  // Copy constructor.
  __forceinline__ __host__ CUDAMatrix3x4(const CUDAMatrix3x4& other)
      : row0(other.row0),
        row1(other.row1),
        row2(other.row2) {}
  
  // Constructs the matrix from an array-like matrix object (works with Eigen matrices).
  template <typename T> __host__ explicit
  CUDAMatrix3x4(const T& matrix) {
    row0.x = matrix(0, 0);
    row0.y = matrix(0, 1);
    row0.z = matrix(0, 2);
    row0.w = matrix(0, 3);
    row1.x = matrix(1, 0);
    row1.y = matrix(1, 1);
    row1.z = matrix(1, 2);
    row1.w = matrix(1, 3);
    row2.x = matrix(2, 0);
    row2.y = matrix(2, 1);
    row2.z = matrix(2, 2);
    row2.w = matrix(2, 3);
  }
  
  // Homogeneous matrix-vector multiplication.
  __forceinline__ __host__ __device__
  float3 operator* (const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z + row0.w,
        row1.x * point.x + row1.y * point.y + row1.z * point.z + row1.w,
        row2.x * point.x + row2.y * point.y + row2.z * point.z + row2.w);
  }
  
  // Homogeneous matrix-vector multiplication, aborts early if the z-component
  // of the result is not positive.
  __forceinline__ __host__ __device__
  bool MultiplyIfResultZIsPositive(const float3& point, float3* result) const {
    result->z = row2.x * point.x + row2.y * point.y + row2.z * point.z + row2.w;
    if (result->z <= 0.f) {
      return false;
    }
    result->x = row0.x * point.x + row0.y * point.y + row0.z * point.z + row0.w;
    result->y = row1.x * point.x + row1.y * point.y + row1.z * point.z + row1.w;
    return true;
  }
  
  // Matrix-vector multiplication of the left 3x3 submatrix with the given
  // vector. If the 3x4 matrix is an element of SE(3), which it always is in the
  // context of this project, then this is a rotation, hence the naming.
  __forceinline__ __host__ __device__
  float3 Rotate(const float3& point) const {
    return make_float3(
        row0.x * point.x + row0.y * point.y + row0.z * point.z,
        row1.x * point.x + row1.y * point.y + row1.z * point.z,
        row2.x * point.x + row2.y * point.y + row2.z * point.z);
  }
  
  // Row-wise storage.
  float4 row0;
  float4 row1;
  float4 row2;
};

}
