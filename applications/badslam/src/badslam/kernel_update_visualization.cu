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
#include <cub/device/device_scan.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>

#include "badslam/cuda_util.cuh"
#include "badslam/util_nvcc_only.cuh"


namespace vis {

template <bool visualize_normals, bool visualize_descriptors, bool visualize_radii>
__global__ void UpdateSurfelVertexBufferCUDAKernel(
    u32 point_size_in_floats,
    u32 surfels_size,
    CUDABuffer_<float> surfels,
    float* vertex_buffer_ptr) {
  unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    // Vertex layout (Point3fC3u8):
    // float x, float y, float z, u8 r, u8 g, u8 b, u8 unused;
    
    float3 global_point = SurfelGetPosition(surfels, surfel_index);
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 0] = global_point.x;
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 1] = global_point.y;
    vertex_buffer_ptr[surfel_index * point_size_in_floats + 2] = global_point.z;
    
    if (visualize_descriptors) {
      // Get descriptors in [-1, 1].
      float descriptor1 = surfels(kSurfelDescriptor1, surfel_index) / 180.f;
      float descriptor2 = surfels(kSurfelDescriptor2, surfel_index) / 180.f;
      
      // Stretch contrast
      int sign1 = (descriptor1 > 0) ? 1 : -1;
      int sign2 = (descriptor2 > 0) ? 1 : -1;
      descriptor1 = sign1 * powf(fabs(descriptor1), 0.35f);
      descriptor2 = sign2 * powf(fabs(descriptor2), 0.35f);
      
      uchar4 color;
      color.x = 255.99f * (0.5f * descriptor1 + 0.5f);
      color.y = 255.99f * (0.5f * descriptor2 + 0.5f);
      color.z = 127;
      color.w = 0;
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
    } else if (visualize_normals) {
      float3 normal = SurfelGetNormal(surfels, surfel_index);
      uchar4 color = make_uchar4(255.99f / 2.0f * (normal.x + 1.0f),
                                 255.99f / 2.0f * (normal.y + 1.0f),
                                 255.99f / 2.0f * (normal.z + 1.0f),
                                 0);
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
    } else if (visualize_radii) {
      float radius_squared = surfels(kSurfelRadiusSquared, surfel_index);
      float radius = sqrtf(radius_squared);
      
      // NOTE: This defines the bounds for one interval, however we allow
      //       wrap-around such that we can also visualize other values without
      //       clamping.
      constexpr float kMinRadius = 1e-6f;
      constexpr float kMaxRadius = 2e-4f;
      
      float factor = (radius - kMinRadius) / (kMaxRadius - kMinRadius);
      uchar4 color = make_uchar4(255.99f * factor,
                                 fmodf(255.99f * (9999 - factor), 255.99f),
                                 0, 0);
      vertex_buffer_ptr[surfel_index * point_size_in_floats + 3] = *reinterpret_cast<float*>(&color);
    } else {
      *reinterpret_cast<uchar4*>(&vertex_buffer_ptr[surfel_index * point_size_in_floats + 3]) = SurfelGetColor(surfels, surfel_index);
    }
  }
}

void UpdateVisualizationBuffersCUDA(
    cudaStream_t stream,
    cudaGraphicsResource_t vertex_buffer_resource,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels,
    bool visualize_normals,
    bool visualize_descriptors,
    bool visualize_radii) {
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  // Map OpenGL buffer object for writing from CUDA.
  cudaGraphicsMapResources(1, &vertex_buffer_resource, stream);
  CUDA_CHECK();
  
  usize num_bytes;
  float* vertex_buffer_ptr;
  cudaGraphicsResourceGetMappedPointer((void**)&vertex_buffer_ptr, &num_bytes, vertex_buffer_resource);
  CUDA_CHECK();
  
//   CHECK(sizeof(Point3fC3u8) % sizeof(float) == 0);
//   u32 point_size_in_floats = sizeof(Point3fC3u8) / sizeof(float);
  constexpr u32 point_size_in_floats = 4;
  
  if ((visualize_descriptors && visualize_normals) ||
      (visualize_descriptors && visualize_radii) ||
      (visualize_normals && visualize_radii)) {
    LOG(FATAL) << "Only one of visualize_descriptors, visualize_normals, and visualize_radii may be true.";
  } else if (visualize_descriptors) {
    CUDA_AUTO_TUNE_1D(
        UpdateSurfelVertexBufferCUDAKernel<TEMPLATE_ARGUMENTS(false, true, false)>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        point_size_in_floats,
        surfels_size,
        surfels,
        vertex_buffer_ptr);
  } else if (visualize_normals) {
    CUDA_AUTO_TUNE_1D(
        UpdateSurfelVertexBufferCUDAKernel<TEMPLATE_ARGUMENTS(true, false, false)>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        point_size_in_floats,
        surfels_size,
        surfels,
        vertex_buffer_ptr);
  } else if (visualize_radii) {
    CUDA_AUTO_TUNE_1D(
        UpdateSurfelVertexBufferCUDAKernel<TEMPLATE_ARGUMENTS(false, false, true)>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        point_size_in_floats,
        surfels_size,
        surfels,
        vertex_buffer_ptr);
  } else {
    CUDA_AUTO_TUNE_1D(
        UpdateSurfelVertexBufferCUDAKernel<TEMPLATE_ARGUMENTS(false, false, false)>,
        1024,
        surfels_size,
        0, stream,
        /* kernel parameters */
        point_size_in_floats,
        surfels_size,
        surfels,
        vertex_buffer_ptr);
  }
  CUDA_CHECK();
  
  cudaGraphicsUnmapResources(1, &vertex_buffer_resource, stream);
}

}
