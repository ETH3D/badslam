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

#include "badslam/cuda_image_processing.cuh"

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <math_constants.h>

#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/util.cuh"

namespace vis {

template <int block_width, int block_height>
__global__ void ComputeSobelGradientMagnitudeKernel(
    CUDABuffer_<uchar3> rgb_buffer,
    CUDABuffer_<uchar4> color_buffer) {
  int x = -1 + static_cast<int>(blockIdx.x * (block_width - 2) + threadIdx.x);
  int y = -1 + static_cast<int>(blockIdx.y * (block_height - 2) + threadIdx.y);
  
  __shared__ float intensity[block_height][block_width];
  
  uchar3 color = rgb_buffer(::max(0, ::min(color_buffer.height() - 1, y)),
                            ::max(0, ::min(color_buffer.width() - 1, x)));
  intensity[threadIdx.y][threadIdx.x] = 0.299f * color.x + 0.587f * color.y + 0.114f * color.z;
  
  __syncthreads();
  
  if (threadIdx.x >= 1 &&
      threadIdx.y >= 1 &&
      threadIdx.x < block_width - 1 &&
      threadIdx.y < block_height - 1 &&
      x < color_buffer.width() &&
      y < color_buffer.height()) {
    float gx =
        1 * intensity[threadIdx.y - 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x - 1] +
        2 * intensity[threadIdx.y + 0][threadIdx.x + 1] -
        2 * intensity[threadIdx.y + 0][threadIdx.x - 1] +
        1 * intensity[threadIdx.y + 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y + 1][threadIdx.x - 1];
    float gy =
        1 * intensity[threadIdx.y + 1][threadIdx.x - 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x - 1] +
        2 * intensity[threadIdx.y + 1][threadIdx.x + 0] -
        2 * intensity[threadIdx.y - 1][threadIdx.x + 0] +
        1 * intensity[threadIdx.y + 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x + 1];
    
    // gx and gy are in [-4 * 255, 4 * 255].
    constexpr float kNormalizer = 255.99f / (CUDART_SQRT_TWO_F * 4 * 255.f);
    u8 gradient_magnitude = kNormalizer * sqrtf(gx * gx + gy * gy);
    color_buffer(y, x) = make_uchar4(color.x, color.y, color.z, gradient_magnitude);
  }
}

void ComputeSobelGradientMagnitudeCUDA(
    cudaStream_t stream,
    const CUDABuffer_<uchar3>& rgb_buffer,
    CUDABuffer_<uchar4>* color_buffer) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D_BORDER_TEMPLATED(
      ComputeSobelGradientMagnitudeKernel,
      32, 32,
      2, 2,
      color_buffer->width(), color_buffer->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(block_width, block_height),
      /* kernel parameters */
      rgb_buffer,
      *color_buffer);
  CUDA_CHECK();
}


template <int block_width, int block_height>
__global__ void ComputeSobelGradientMagnitudeKernel(
    cudaTextureObject_t rgbi_texture,
    CUDABuffer_<u8> gradmag_buffer) {
  int x = -1 + static_cast<int>(blockIdx.x * (block_width - 2) + threadIdx.x);
  int y = -1 + static_cast<int>(blockIdx.y * (block_height - 2) + threadIdx.y);
  
  __shared__ float intensity[block_height][block_width];
  
  intensity[threadIdx.y][threadIdx.x] = 255.f * tex2D<float4>(rgbi_texture, x + 0.5f, y + 0.5f).w;
  
  __syncthreads();
  
  if (threadIdx.x >= 1 &&
      threadIdx.y >= 1 &&
      threadIdx.x < block_width - 1 &&
      threadIdx.y < block_height - 1 &&
      x < gradmag_buffer.width() &&
      y < gradmag_buffer.height()) {
    float gx =
        1 * intensity[threadIdx.y - 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x - 1] +
        2 * intensity[threadIdx.y + 0][threadIdx.x + 1] -
        2 * intensity[threadIdx.y + 0][threadIdx.x - 1] +
        1 * intensity[threadIdx.y + 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y + 1][threadIdx.x - 1];
    float gy =
        1 * intensity[threadIdx.y + 1][threadIdx.x - 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x - 1] +
        2 * intensity[threadIdx.y + 1][threadIdx.x + 0] -
        2 * intensity[threadIdx.y - 1][threadIdx.x + 0] +
        1 * intensity[threadIdx.y + 1][threadIdx.x + 1] -
        1 * intensity[threadIdx.y - 1][threadIdx.x + 1];
    
    // gx and gy are in [-4 * 255, 4 * 255].
    constexpr float kNormalizer = 255.99f / (CUDART_SQRT_TWO_F * 4 * 255.f);
    u8 gradient_magnitude = kNormalizer * sqrtf(gx * gx + gy * gy);
    gradmag_buffer(y, x) = gradient_magnitude;
  }
}

void ComputeSobelGradientMagnitudeCUDA(
    cudaStream_t stream,
    cudaTextureObject_t rgbi_texture,
    CUDABuffer_<u8>* gradmag_buffer) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D_BORDER_TEMPLATED(
      ComputeSobelGradientMagnitudeKernel,
      32, 32,
      2, 2,
      gradmag_buffer->width(), gradmag_buffer->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(block_width, block_height),
      /* kernel parameters */
      rgbi_texture,
      *gradmag_buffer);
  CUDA_CHECK();
}


__global__ void ComputeBrightnessKernel(
    CUDABuffer_<uchar3> rgb_buffer,
    CUDABuffer_<uchar4> color_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < color_buffer.width() && y < color_buffer.height()) {
    uchar3 color = rgb_buffer(y, x);
    u8 intensity = (0.299f * color.x + 0.587f * color.y + 0.114f * color.z) + 0.5f;
    color_buffer(y, x) = make_uchar4(color.x, color.y, color.z, intensity);
  }
}

void ComputeBrightnessCUDA(
    cudaStream_t stream,
    const CUDABuffer_<uchar3>& rgb_buffer,
    CUDABuffer_<uchar4>* color_buffer) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D(
      ComputeBrightnessKernel,
      32, 32,
      color_buffer->width(), color_buffer->height(),
      0, stream,
      /* kernel parameters */
      rgb_buffer,
      *color_buffer);
  CUDA_CHECK();
}


__global__ void ComputeBrightnessKernel(
    cudaTextureObject_t rgbi_texture,
    CUDABuffer_<u8> intensity_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < intensity_buffer.width() && y < intensity_buffer.height()) {
    u8 intensity = 255.f * tex2D<float4>(rgbi_texture, x + 0.5f, y + 0.5f).w;
    intensity_buffer(y, x) = intensity;
  }
}

void ComputeBrightnessCUDA(
    cudaStream_t stream,
    cudaTextureObject_t rgbi_texture,
    CUDABuffer_<u8>* intensity_buffer) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D(
      ComputeBrightnessKernel,
      32, 32,
      intensity_buffer->width(), intensity_buffer->height(),
      0, stream,
      /* kernel parameters */
      rgbi_texture,
      *intensity_buffer);
  CUDA_CHECK();
}


template <typename T>
__global__ void UpscaleBufferBilinearlyCUDAKernel(
    cudaTextureObject_t texture,
    CUDABuffer_<T> dest) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < dest.width() && y < dest.height()) {
    dest(y, x) = tex2D<float>(texture, (x + 0.5f) / dest.width(), (y + 0.5f) / dest.height());
  }
}

template <typename T>
void UpscaleBufferBilinearlyCUDA(
    cudaStream_t stream,
    cudaTextureObject_t src,
    CUDABuffer_<T>* dest) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D_TEMPLATED(
      UpscaleBufferBilinearlyCUDAKernel,
      32, 32,
      dest->width(), dest->height(),
      0, stream,
      TEMPLATE_ARGUMENTS(T),
      /* kernel parameters */
      src,
      *dest);
  
  CUDA_CHECK();
}

template void UpscaleBufferBilinearlyCUDA<float>(cudaStream_t stream, cudaTextureObject_t src, CUDABuffer_<float>* dest);

}
