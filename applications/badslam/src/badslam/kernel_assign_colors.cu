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
#include "badslam/kernels.cuh"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

__global__ void ResetSurfelForColorAssignmentKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
    surfels(kSurfelAccum2, surfel_index) = 0;
    surfels(kSurfelAccum3, surfel_index) = 0;
    surfels(kSurfelAccum4, surfel_index) = 0;
  }
}

void CallResetSurfelForColorAssignmentKernel(
    cudaStream_t stream,
    int surfels_size,
    const CUDABuffer_<float>& surfels) {
  CUDA_AUTO_TUNE_1D(
      ResetSurfelForColorAssignmentKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels);
  CUDA_CHECK();
}


__global__ void AccumulateColorObservationsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    cudaTextureObject_t color_texture) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  SurfelProjectionResultXYFloat r;
  if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
    float2 color_pxy;
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      s.surfels(kSurfelAccum0, surfel_index) += 1.f;
      float4 color = tex2D<float4>(color_texture, color_pxy.x, color_pxy.y);
      s.surfels(kSurfelAccum1, surfel_index) += color.x;
      s.surfels(kSurfelAccum2, surfel_index) += color.y;
      s.surfels(kSurfelAccum3, surfel_index) += color.z;
      s.surfels(kSurfelAccum4, surfel_index) += color.w;
    }
  }
}

void CallAccumulateColorObservationsCUDAKernel(
    cudaStream_t stream,
    int surfels_size,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    cudaTextureObject_t color_texture) {
  CUDA_AUTO_TUNE_1D(
      AccumulateColorObservationsCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      s,
      depth_to_color,
      color_texture);
  CUDA_CHECK();
}


__global__ void AssignColorsCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    float observation_count = surfels(kSurfelAccum0, surfel_index);
    if (observation_count > 0) {
      u8 r = 255.f * surfels(kSurfelAccum1, surfel_index) / observation_count + 0.5f;
      u8 g = 255.f * surfels(kSurfelAccum2, surfel_index) / observation_count + 0.5f;
      u8 b = 255.f * surfels(kSurfelAccum3, surfel_index) / observation_count + 0.5f;
      u8 gradmag = 255.f * surfels(kSurfelAccum4, surfel_index) / observation_count + 0.5f;
      SurfelSetColor(&surfels, surfel_index, make_uchar4(r, g, b, gradmag));
    } else {
      SurfelSetColor(&surfels, surfel_index, make_uchar4(0, 0, 0, 0));
    }
  }
}

void CallAssignColorsCUDAKernel(
    cudaStream_t stream,
    int surfels_size,
    const CUDABuffer_<float>& surfels) {
  CUDA_AUTO_TUNE_1D(
      AssignColorsCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels);
  CUDA_CHECK();
}


__global__ void ResetSurfelForDescriptorColorAssignmentKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (surfel_index < surfels_size) {
    surfels(kSurfelAccum0, surfel_index) = 0;
    surfels(kSurfelAccum1, surfel_index) = 0;
    surfels(kSurfelAccum2, surfel_index) = 0;
  }
}

void CallResetSurfelForDescriptorColorAssignmentKernel(
    cudaStream_t stream,
    int surfels_size,
    const CUDABuffer_<float>& surfels) {
  CUDA_AUTO_TUNE_1D(
      ResetSurfelForDescriptorColorAssignmentKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels);
  CUDA_CHECK();
}


__global__ void AccumulateDescriptorColorObservationsCUDAKernel(
    SurfelProjectionParameters s,
    DepthToColorPixelCorner depth_to_color,
    PixelCornerProjector color_corner_projector,
    cudaTextureObject_t color_texture) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  SurfelProjectionResult6 r;
  if (SurfelProjectsToAssociatedPixel(surfel_index, s, &r)) {
    float2 color_pxy;
    if (TransformDepthToColorPixelCorner(r.pxy, depth_to_color, &color_pxy)) {
      s.surfels(kSurfelAccum0, surfel_index) += 1.f;
      
      float2 t1_pxy, t2_pxy;
      ComputeTangentProjections(
          r.surfel_global_position,
          r.surfel_normal,
          SurfelGetRadiusSquared(s.surfels, surfel_index),
          s.frame_T_global,
          color_corner_projector,
          &t1_pxy,
          &t2_pxy);
      
      float descriptor_1;
      float descriptor_2;
      ComputeRawDescriptorResidual(
          color_texture,
          color_pxy,
          t1_pxy,
          t2_pxy,
          /*surfel_descriptor_1*/ 0,
          /*surfel_descriptor_2*/ 0,
          &descriptor_1,
          &descriptor_2);
      
      s.surfels(kSurfelAccum1, surfel_index) += descriptor_1;
      s.surfels(kSurfelAccum2, surfel_index) += descriptor_2;
    }
  }
}

void CallAccumulateDescriptorColorObservationsCUDAKernel(
    cudaStream_t stream,
    int surfels_size,
    const SurfelProjectionParameters& s,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    cudaTextureObject_t color_texture) {
  CUDA_AUTO_TUNE_1D(
      AccumulateDescriptorColorObservationsCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      s,
      depth_to_color,
      color_corner_projector,
      color_texture);
  CUDA_CHECK();
}


__global__ void AssignDescriptorColorsCUDAKernel(
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  const unsigned int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfel_index < surfels_size) {
    float observation_count = surfels(kSurfelAccum0, surfel_index);
    if (observation_count > 0) {
      // Get average descriptors in [-1, 1].
      float descriptor1 = surfels(kSurfelAccum1, surfel_index) / (observation_count * 180.f);
      float descriptor2 = surfels(kSurfelAccum2, surfel_index) / (observation_count * 180.f);
      
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
      SurfelSetColor(&surfels, surfel_index, color);
    } else {
      SurfelSetColor(&surfels, surfel_index, make_uchar4(0, 0, 0, 0));
    }
  }
}

void CallAssignDescriptorColorsCUDAKernel(
    cudaStream_t stream,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels) {
  CUDA_AUTO_TUNE_1D(
      AssignDescriptorColorsCUDAKernel,
      1024,
      surfels_size,
      0, stream,
      /* kernel parameters */
      surfels_size,
      surfels);
  CUDA_CHECK();
}

}
