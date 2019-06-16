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
#include <math_constants.h>

#include "badslam/cuda_util.cuh"
#include "badslam/surfel_projection.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

template <bool downsample_color>
__global__ void CalibrateAndDownsampleImagesCUDAKernel(
    DepthParameters depth_params,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u16> normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float> downsampled_depth,
    CUDABuffer_<u16> downsampled_normals,
    CUDABuffer_<u8> downsampled_color) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < downsampled_depth.width() && y < downsampled_depth.height()) {
    constexpr int kOffsets[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float depths[4];
    
    float depth_sum = 0;
    int depth_count = 0;
    
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
      u16 raw_depth = depth_buffer(2 * y + kOffsets[i][0], 2 * x + kOffsets[i][1]);
      if (!(raw_depth & kInvalidDepthBit)) {
        depths[i] = RawToCalibratedDepth(
            depth_params.a,
            depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                        x / depth_params.sparse_surfel_cell_size),
            depth_params.raw_to_float_depth, raw_depth);
        depth_sum += depths[i];
        depth_count += 1;
      } else {
        depths[i] = CUDART_INF_F;
      }
    }
    
    if (depth_count == 0) {
      // Normal does not need to be set here, as the pixel is invalid by setting its depth to 0.
      // However, the color must be set, as it might become relevant again for further downsampling.
      downsampled_depth(y, x) = 0;
    } else {
      float average_depth = depth_sum / depth_count;
      int closest_index;
      float closest_distance = CUDART_INF_F;
      #pragma unroll
      for (int i = 0; i < 4; ++ i) {
        float distance = fabs(depths[i] - average_depth);
        if (distance < closest_distance) {
          closest_index = i;
          closest_distance = distance;
        }
      }
      
      downsampled_depth(y, x) = depths[closest_index];
      downsampled_normals(y, x) = normals_buffer(2 * y + kOffsets[closest_index][0], 2 * x + kOffsets[closest_index][1]);
    }
    
    if (downsample_color) {
      // Bilinearly interpolate in the middle of the original 4 pixels to get their average.
      float color = tex2D<float>(color_texture, 2 * x + 1.0f, 2 * y + 1.0f);
      downsampled_color(y, x) = 255.f * color + 0.5f;
    } else {
      float color = tex2D<float>(color_texture, x + 0.5f, y + 0.5f);
      downsampled_color(y, x) = 255.f * color + 0.5f;
    }
  }
}

__global__ void DownsampleImagesCUDAKernel(
    CUDABuffer_<float> depth_buffer,
    CUDABuffer_<u16> normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float> downsampled_depth,
    CUDABuffer_<u16> downsampled_normals,
    CUDABuffer_<u8> downsampled_color) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < downsampled_depth.width() && y < downsampled_depth.height()) {
    constexpr int kOffsets[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    float depths[4];
    
    float depth_sum = 0;
    int depth_count = 0;
    
    #pragma unroll
    for (int i = 0; i < 4; ++ i) {
      depths[i] = depth_buffer(2 * y + kOffsets[i][0], 2 * x + kOffsets[i][1]);
      if (depths[i] > 0) {
        depth_sum += depths[i];
        depth_count += 1;
      } else {
        depths[i] = CUDART_INF_F;
      }
    }
    
    if (depth_count == 0) {
      // Normal does not need to be set here, as the pixel is invalid by setting its depth to 0.
      // However, the color must be set, as it might become relevant again for further downsampling.
      downsampled_depth(y, x) = 0;
    } else {
      float average_depth = depth_sum / depth_count;
      int closest_index;
      float closest_distance = CUDART_INF_F;
      #pragma unroll
      for (int i = 0; i < 4; ++ i) {
        float distance = fabs(depths[i] - average_depth);
        if (distance < closest_distance) {
          closest_index = i;
          closest_distance = distance;
        }
      }
      
      downsampled_depth(y, x) = depths[closest_index];
      downsampled_normals(y, x) = normals_buffer(2 * y + kOffsets[closest_index][0], 2 * x + kOffsets[closest_index][1]);
    }
    
    // Bilinearly interpolate in the middle of the original 4 pixels to get their average.
    float color = tex2D<float>(color_texture, 2 * x + 1.0f, 2 * y + 1.0f);
    downsampled_color(y, x) = 255.f * color + 0.5f;
  }
}

// __global__ void DownsampleImagesConsistentlyCUDAKernel(
//     CUDABuffer_<float> comparison_depth_buffer,
//     CUDABuffer_<float> depth_buffer,
//     CUDABuffer_<u16> normals_buffer,
//     cudaTextureObject_t color_texture,
//     CUDABuffer_<float> downsampled_depth,
//     CUDABuffer_<u16> downsampled_normals,
//     CUDABuffer_<uchar> downsampled_color) {
//   unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//   unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
//   
//   if (x < downsampled_depth.width() && y < downsampled_depth.height()) {
//     constexpr int kOffsets[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
//     float depths[4];
//     
//     float depth_sum = 0;
//     int depth_count = 0;
//     
//     #pragma unroll
//     for (int i = 0; i < 4; ++ i) {
//       depths[i] = depth_buffer(2 * y + kOffsets[i][0], 2 * x + kOffsets[i][1]);
//       if (depths[i] > 0) {
//         depth_sum += depths[i];
//         depth_count += 1;
//       } else {
//         depths[i] = CUDART_INF_F;
//       }
//     }
//     
//     if (depth_count == 0) {
//       downsampled_depth(y, x) = 0;
//     } else {
//       float comparison_depth = comparison_depth_buffer(y, x);
//       if (comparison_depth == 0) {
//         // Use average if no comparison depth exists for this pixel
//         comparison_depth = depth_sum / depth_count;
//       }
//       int closest_index;
//       float closest_distance = CUDART_INF_F;
//       
//       #pragma unroll
//       for (int i = 0; i < 4; ++ i) {
//         float distance = fabs(depths[i] - comparison_depth);
//         if (distance < closest_distance) {
//           closest_index = i;
//           closest_distance = distance;
//         }
//       }
//       
//       downsampled_depth(y, x) = depths[closest_index];
//       downsampled_normals(y, x) = normals_buffer(2 * y + kOffsets[closest_index][0], 2 * x + kOffsets[closest_index][1]);
//       
//       // For color averaging, use only pixels with valid and similar depth to the chosen depth.
//       // This is to avoid using occluded surfels for averaging which "shine through" in the surfel rendering.
//       // Notice that this will not properly simulate pixels with mixed colors at occlusion boundaries!
//       constexpr float kColorAveragingDistanceThreshold = 0.15f;  // TODO: tune this threshold
//       
//       depth_sum = 0;  // misused for color averaging
//       depth_count = 0;
// 
//       #pragma unroll
//       for (int i = 0; i < 4; ++ i) {
//         float distance = fabs(depths[i] - depths[closest_index]);
//         if (distance < kColorAveragingDistanceThreshold) {
//           depth_sum += 255.f * tex2D<float>(color_texture, 2 * x + kOffsets[i][1] + 0.5f, 2 * y + kOffsets[i][0] + 0.5f);
//           depth_count += 1;
//         }
//       }
//       
//       downsampled_color(y, x) = depth_sum / depth_count + 0.5f;
//     }
//   }
// }

void CalibrateAndDownsampleImagesCUDA(
    cudaStream_t stream,
    bool downsample_color,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* downsampled_depth,
    CUDABuffer_<u16>* downsampled_normals,
    CUDABuffer_<u8>* downsampled_color,
    bool debug) {
  CUDA_CHECK();
  
  if (debug) {
    downsampled_depth->Clear(0, stream);
    downsampled_normals->Clear(0, stream);
    downsampled_color->Clear(0, stream);
  }
  
  COMPILE_OPTION(downsample_color,
      CUDA_AUTO_TUNE_2D(
          CalibrateAndDownsampleImagesCUDAKernel<_downsample_color>,
          32, 32,
          downsampled_depth->width(), downsampled_depth->height(),
          0, stream,
          /* kernel parameters */
          depth_params,
          depth_buffer,
          normals_buffer,
          color_texture,
          *downsampled_depth,
          *downsampled_normals,
          *downsampled_color));
  CUDA_CHECK();
}

void DownsampleImagesCUDA(
    cudaStream_t stream,
    const CUDABuffer_<float>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* downsampled_depth,
    CUDABuffer_<u16>* downsampled_normals,
    CUDABuffer_<u8>* downsampled_color,
    bool debug) {
  CUDA_CHECK();
  
  if (debug) {
    downsampled_depth->Clear(0, stream);
    downsampled_normals->Clear(0, stream);
    downsampled_color->Clear(0, stream);
  }
  
  CUDA_AUTO_TUNE_2D(
      DownsampleImagesCUDAKernel,
      32, 32,
      downsampled_depth->width(), downsampled_depth->height(),
      0, stream,
      /* kernel parameters */
      depth_buffer,
      normals_buffer,
      color_texture,
      *downsampled_depth,
      *downsampled_normals,
      *downsampled_color);
  CUDA_CHECK();
}

// void DownsampleImagesConsistentlyCUDA(
//     cudaStream_t stream,
//     const CUDABuffer_<float>& comparison_depth_buffer,
//     const CUDABuffer_<u16>& /*comparison_normals_buffer*/,
//     const CUDABuffer_<float>& depth_buffer,
//     const CUDABuffer_<u16>& normals_buffer,
//     cudaTextureObject_t color_texture,
//     CUDABuffer_<float>* downsampled_depth,
//     CUDABuffer_<u16>* downsampled_normals,
//     CUDABuffer_<uchar>* downsampled_color,
//     bool debug) {
//   // TODO: comparison_normals_buffer is not used currently, could remove it
//   
//   CUDA_CHECK();
//   
//   if (debug) {
//     downsampled_depth->Clear(0, stream);
//     downsampled_normals->Clear(0, stream);
//     downsampled_color->Clear(0, stream);
//   }
//   
//   CUDA_AUTO_TUNE_2D(
//       DownsampleImagesConsistentlyCUDAKernel,
//       32, 32,
//       downsampled_depth->width(), downsampled_depth->height(),
//       0, stream,
//       /* kernel parameters */
//       comparison_depth_buffer,
//       depth_buffer,
//       normals_buffer,
//       color_texture,
//       *downsampled_depth,
//       *downsampled_normals,
//       *downsampled_color);
//   CUDA_CHECK();
// }


// -----------------------------------------------------------------------------


__global__ void CalibrateDepthAndTransformColorToDepthCUDAKernel(
    DepthToColorPixelCorner depth_to_color,
    DepthParameters depth_params,
    CUDABuffer_<u16> depth_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float> out_depth,
    CUDABuffer_<u8> out_color) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < out_depth.width() && y < out_depth.height()) {
    u16 raw_depth = depth_buffer(y, x);
    float depth;
    if (!(raw_depth & kInvalidDepthBit)) {
      depth = RawToCalibratedDepth(
          depth_params.a,
          depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                      x / depth_params.sparse_surfel_cell_size),
          depth_params.raw_to_float_depth, raw_depth);
    } else {
      depth = 0;
    }
    
    float2 color_pxy;
    bool color_in_bounds = TransformDepthToColorPixelCorner(make_float2(x + 0.5f, y + 0.5f), depth_to_color, &color_pxy);
    
    out_depth(y, x) = color_in_bounds ? depth : 0;
    
    float color = tex2D<float>(color_texture, color_pxy.x, color_pxy.y);
    out_color(y, x) = 255.f * color + 0.5f;
  }
}

void CalibrateDepthAndTransformColorToDepthCUDA(
    cudaStream_t stream,
    const DepthToColorPixelCorner& depth_to_color,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>* out_depth,
    CUDABuffer_<u8>* out_color) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D(
      CalibrateDepthAndTransformColorToDepthCUDAKernel,
      32, 32,
      out_depth->width(), out_depth->height(),
      0, stream,
      /* kernel parameters */
      depth_to_color,
      depth_params,
      depth_buffer,
      color_texture,
      *out_depth,
      *out_color);
  CUDA_CHECK();
}


__global__ void CalibrateDepthCUDAKernel(
    DepthParameters depth_params,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float> out_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < out_depth.width() && y < out_depth.height()) {
    u16 raw_depth = depth_buffer(y, x);
    float depth;
    if (!(raw_depth & kInvalidDepthBit)) {
      depth = RawToCalibratedDepth(
          depth_params.a,
          depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                      x / depth_params.sparse_surfel_cell_size),
          depth_params.raw_to_float_depth, raw_depth);
    } else {
      depth = 0;
    }
    
    out_depth(y, x) = depth;
  }
}

void CalibrateDepthCUDA(
    cudaStream_t stream,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<float>* out_depth) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D(
      CalibrateDepthCUDAKernel,
      32, 32,
      out_depth->width(), out_depth->height(),
      0, stream,
      /* kernel parameters */
      depth_params,
      depth_buffer,
      *out_depth);
  CUDA_CHECK();
}

}
