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

#include "badslam/cuda_depth_processing.cuh"

#include <cub/cub.cuh>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <math_constants.h>

#include "badslam/cuda_util.cuh"
#include "badslam/cuda_matrix.cuh"
#include "badslam/kernels.cuh"
#include "badslam/util.cuh"

namespace vis {

__global__ void BilateralFilteringAndDepthCutoffCUDAKernel(
    float denom_xy,
    float denom_value,
    int radius,
    int radius_squared,
    u16 max_depth,
    float raw_to_float_depth,
    CUDABuffer_<u16> input_depth,
    CUDABuffer_<u16> output_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < output_depth.width() && y < output_depth.height()) {
    // Depth cutoff.
    const u16 center_value = input_depth(y, x);
    if (center_value == 0 || center_value > max_depth) {
      output_depth(y, x) = kUnknownDepth;
      return;
    }
    const float inv_center_value = 1.0f / (raw_to_float_depth * center_value);
    
    // Bilateral filtering.
    float sum = 0;
    float weight = 0;
    
    const int min_y = max(static_cast<int>(0), static_cast<int>(y - radius));
    const int max_y = min(static_cast<int>(output_depth.height() - 1), static_cast<int>(y + radius));
    for (int sample_y = min_y; sample_y <= max_y; ++ sample_y) {
      const int dy = sample_y - y;
      
      const int min_x = max(static_cast<int>(0), static_cast<int>(x - radius));
      const int max_x = min(static_cast<int>(output_depth.width() - 1), static_cast<int>(x + radius));
      for (int sample_x = min_x; sample_x <= max_x; ++ sample_x) {
        const int dx = sample_x - x;
        
        const int grid_distance_squared = dx * dx + dy * dy;
        if (grid_distance_squared > radius_squared) {
          continue;
        }
        
        const u16 sample = input_depth(sample_y, sample_x);
        if (sample == 0) {
          continue;
        }
        const float inv_sample = 1.0f / (raw_to_float_depth * sample);
        
        float value_distance_squared = inv_center_value - inv_sample;
        value_distance_squared *= value_distance_squared;
        float w = exp(-grid_distance_squared / denom_xy + -value_distance_squared / denom_value);
        sum += w * inv_sample;
        weight += w;
      }
    }
    
    output_depth(y, x) = (weight == 0) ? kUnknownDepth : (1.0f / (raw_to_float_depth * sum / weight));
  }
}

void BilateralFilteringAndDepthCutoffCUDA(
    cudaStream_t stream,
    float sigma_xy,
    float sigma_value,
    float radius_factor,
    u16 max_depth,
    float raw_to_float_depth,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth) {
  CUDA_CHECK();
  
  int radius = radius_factor * sigma_xy + 0.5f;
  
  CUDA_AUTO_TUNE_2D(
      BilateralFilteringAndDepthCutoffCUDAKernel,
      32, 32,
      output_depth->width(), output_depth->height(),
      0, stream,
      /* kernel parameters */
      2.0f * sigma_xy * sigma_xy,
      2.0f * sigma_value * sigma_value,
      radius,
      radius * radius,
      max_depth,
      raw_to_float_depth,
      input_depth,
      *output_depth);
  CUDA_CHECK();
}


// -----------------------------------------------------------------------------


__global__ void ComputeNormalsCUDAKernel(
    PixelCenterUnprojector unprojector,
    DepthParameters depth_params,
    CUDABuffer_<u16> in_depth,
    CUDABuffer_<u16> out_depth,
    CUDABuffer_<u16> out_normals) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < in_depth.width() && y < in_depth.height()) {
    constexpr int kBorder = 1;
    if (x < kBorder ||
        y < kBorder ||
        x >= in_depth.width() - kBorder ||
        y >= in_depth.height() - kBorder) {
      out_depth(y, x) = kUnknownDepth;
      out_normals(y, x) = ImageSpaceNormalToU16(0, 0);
      return;
    }
    u16 center_raw_depth = in_depth(y, x);
    if (center_raw_depth & kInvalidDepthBit) {
      out_depth(y, x) = kUnknownDepth;
      out_normals(y, x) = ImageSpaceNormalToU16(0, 0);
      return;
    }
    
    u16 right_raw_depth = in_depth(y, x + 1);
    u16 left_raw_depth = in_depth(y, x - 1);
    u16 bottom_raw_depth = in_depth(y + 1, x);
    u16 top_raw_depth = in_depth(y - 1, x);
    if (right_raw_depth & kInvalidDepthBit ||
        left_raw_depth & kInvalidDepthBit ||
        bottom_raw_depth & kInvalidDepthBit ||
        top_raw_depth & kInvalidDepthBit) {
      // TODO: Still use this pixel by only using the valid neighbors (if there are enough).
      //       Should test the effect on accuracy though since pixels at borders might be likely to be uncertain!
      out_depth(y, x) = kUnknownDepth;
      out_normals(y, x) = ImageSpaceNormalToU16(0, 0);
      return;
    }
    
    float center_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                    x / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth, center_raw_depth);
    float left_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                    (x - 1) / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth, left_raw_depth);
    float top_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer((y - 1) / depth_params.sparse_surfel_cell_size,
                                    x / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth, top_raw_depth);
    float right_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                    (x + 1) / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth, right_raw_depth);
    float bottom_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer((y + 1) / depth_params.sparse_surfel_cell_size,
                                    x / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth, bottom_raw_depth);
    
    float3 left_point = unprojector.UnprojectPoint(x - 1, y, left_depth);
    float3 top_point = unprojector.UnprojectPoint(x, y - 1, top_depth);
    float3 right_point = unprojector.UnprojectPoint(x + 1, y, right_depth);
    float3 bottom_point = unprojector.UnprojectPoint(x, y + 1, bottom_depth);
    float3 center_point = unprojector.UnprojectPoint(x, y, center_depth);
    
    constexpr float kRatioThreshold = 2.f;
    constexpr float kRatioThresholdSquared = kRatioThreshold * kRatioThreshold;
    
    float left_dist_squared = SquaredLength(left_point - center_point);
    float right_dist_squared = SquaredLength(right_point - center_point);
    float left_right_ratio = left_dist_squared / right_dist_squared;
    float3 left_to_right;
    if (left_right_ratio < kRatioThresholdSquared &&
        left_right_ratio > 1.f / kRatioThresholdSquared) {
      left_to_right = right_point - left_point;
    } else if (left_dist_squared < right_dist_squared) {
      left_to_right = center_point - left_point;
    } else {  // left_dist_squared >= right_dist_squared
      left_to_right = right_point - center_point;
    }
    
    float bottom_dist_squared = SquaredLength(bottom_point - center_point);
    float top_dist_squared = SquaredLength(top_point - center_point);
    float bottom_top_ratio = bottom_dist_squared / top_dist_squared;
    float3 bottom_to_top;
    if (bottom_top_ratio < kRatioThresholdSquared &&
        bottom_top_ratio > 1.f / kRatioThresholdSquared) {
      bottom_to_top = top_point - bottom_point;
    } else if (bottom_dist_squared < top_dist_squared) {
      bottom_to_top = center_point - bottom_point;
    } else {  // bottom_dist_squared >= top_dist_squared
      bottom_to_top = top_point - center_point;
    }
    
    float3 normal;
    CrossProduct(left_to_right, bottom_to_top, &normal);
    
    float length = Norm(normal);
    if (!(length > 1e-6f)) {
      normal = make_float3(0, 0, -1);  // avoid NaNs
    } else {
      // This accounts for negative fy in ICL-NUIM data. Though such weird
      // things should best be avoided in dataset creation ...
      float inv_length = ((unprojector.fy_inv < 0) ? -1.0f : 1.0f) / length;
      
      normal.x *= inv_length;
      normal.y *= inv_length;
      // normal.z *= inv_length;  // not used later, thus not assigned.
    }
    
    out_normals(y, x) = ImageSpaceNormalToU16(normal.x, normal.y);
    out_depth(y, x) = in_depth(y, x);
  }
}

void ComputeNormalsCUDA(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& input_depth,
    CUDABuffer_<u16>* output_depth,
    CUDABuffer_<u16>* normals_buffer) {
  CUDA_CHECK();
  
  CUDA_AUTO_TUNE_2D(
      ComputeNormalsCUDAKernel,
      32, 32,
      output_depth->width(), output_depth->height(),
      0, stream,
      /* kernel parameters */
      unprojector,
      depth_params,
      input_depth,
      *output_depth,
      *normals_buffer);
  CUDA_CHECK();
}


// -----------------------------------------------------------------------------


// Computes the minimum squared distance of the point to one of its neighbor
// points, which is used as the point radius.
__forceinline__ __device__ float ComputePointRadius(
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    float raw_to_float_depth,
    const CUDABuffer_<u16>& depth_buffer,
    unsigned int x,
    unsigned int y,
    u16 depth_u16,
    int* neighbor_count) {
  float depth = raw_to_float_depth * depth_u16;
  float3 local_position =
      make_float3(depth * (fx_inv * x + cx_inv),
                  depth * (fy_inv * y + cy_inv),
                  depth);
  
  // Determine the radius of the pixel's 3D point as the minimum distance to
  // a point from its 4-neighborhood.
  *neighbor_count = 0;
  float min_neighbor_distance_squared = CUDART_INF_F;
  for (int dy = y - 1, end_dy = y + 2; dy < end_dy; ++ dy) {
    for (int dx = x - 1, end_dx = x + 2; dx < end_dx; ++ dx) {
      u16 d_depth = depth_buffer(dy, dx);
      if ((dx != x && dy != y) ||  // no diagonal directions
          (dx == x && dy == y) ||
            d_depth & kInvalidDepthBit) {
        continue;
      }
      ++ (*neighbor_count);
      
      float ddepth = raw_to_float_depth * d_depth;
      float3 other_point =
          make_float3(ddepth * (fx_inv * dx + cx_inv),
                      ddepth * (fy_inv * dy + cy_inv),
                      ddepth);
      float3 local_to_other = other_point - local_position;
      float distance_squared = SquaredLength(local_to_other);
      if (distance_squared < min_neighbor_distance_squared) {
        min_neighbor_distance_squared = distance_squared;
      }
    }
  }
  
  return min_neighbor_distance_squared;
}

template <int min_neighbors_for_radius_computation>
__global__ void ComputePointRadiiAndRemoveIsolatedPixelsCUDAKernel(
    float fx_inv, float fy_inv, float cx_inv, float cy_inv,
    float raw_to_float_depth,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u16> radius_buffer,
    CUDABuffer_<u16> out_depth) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    const u16 depth_u16 = depth_buffer(y, x);
    if (depth_u16 & kInvalidDepthBit) {
      out_depth(y, x) = kUnknownDepth;
      return;
    }
    
    int neighbor_count;
    float point_radius = ComputePointRadius(
        fx_inv, fy_inv, cx_inv, cy_inv, raw_to_float_depth, depth_buffer,
        x, y, depth_u16, &neighbor_count);
    
    // Require all neighbors to have depth values.
    bool valid = neighbor_count >= min_neighbors_for_radius_computation;
    
    radius_buffer(y, x) = __half_as_ushort(__float2half_rn(valid ? point_radius : 0));
    out_depth(y, x) = valid ? depth_u16 : kUnknownDepth;
  }
}

void ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    float raw_to_float_depth,
    const CUDABuffer_<u16>& depth_buffer,
    CUDABuffer_<u16>* radius_buffer,
    CUDABuffer_<u16>* out_depth) {
  CUDA_CHECK();
  
  constexpr int kMinNeighborsForRadiusComputation = 4;
  
  CUDA_AUTO_TUNE_2D_TEMPLATED(
      ComputePointRadiiAndRemoveIsolatedPixelsCUDAKernel,
      32, 32,
      depth_buffer.width(), depth_buffer.height(),
      0, stream,
      TEMPLATE_ARGUMENTS(kMinNeighborsForRadiusComputation),
      /* kernel parameters */
      unprojector.fx_inv, unprojector.fy_inv, unprojector.cx_inv, unprojector.cy_inv,
      raw_to_float_depth,
      depth_buffer,
      *radius_buffer,
      *out_depth);
  CUDA_CHECK();
}


// -----------------------------------------------------------------------------


template <int block_width, int block_height>
__global__ void ComputeMinMaxDepthCUDAKernel(
    float raw_to_float_depth,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<float> result_buffer) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  float depth = CUDART_NAN_F;
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    const u16 depth_u16 = depth_buffer(y, x);
    if (!(depth_u16 & kInvalidDepthBit)) {
      depth = raw_to_float_depth * depth_u16;
    }
  }
  
  typedef cub::BlockReduce<float, block_width, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_height> BlockReduceFloat;
  __shared__ typename BlockReduceFloat::TempStorage float_storage;
  
  const float min_depth =
      BlockReduceFloat(float_storage).Reduce(::isnan(depth) ?
                                             CUDART_INF_F :
                                             depth, cub::Min());
  // TODO: Would it be faster to use different shared memory buffers for the reduce operations to avoid the __syncthreads() call?
  __syncthreads();  // Required before re-use of shared memory.
  const float max_depth =
      BlockReduceFloat(float_storage).Reduce(::isnan(depth) ?
                                             0 :
                                             depth, cub::Max());
  
  if (threadIdx.x == 0) {
    // Should behave properly as long as all the floats are positive.
    atomicMin(reinterpret_cast<int*>(&result_buffer(0, 0)), __float_as_int(min_depth));
    atomicMax(reinterpret_cast<int*>(&result_buffer(0, 1)), __float_as_int(max_depth));
  }
}

void ComputeMinMaxDepthCUDA(
    cudaStream_t stream,
    const CUDABuffer_<u16>& depth_buffer,
    float raw_to_float_depth,
    const CUDABuffer_<float>& init_buffer,
    CUDABuffer_<float>* result_buffer,
    float* keyframe_min_depth,
    float* keyframe_max_depth) {
  CUDA_CHECK();
  
  cudaMemcpyAsync(result_buffer->address(),
                  init_buffer.address(),
                  2 * sizeof(float),
                  cudaMemcpyDeviceToDevice,
                  stream);
  
  CUDA_AUTO_TUNE_2D_TEMPLATED(
      ComputeMinMaxDepthCUDAKernel,
      32, 32,
      depth_buffer.width(), depth_buffer.height(),
      0, stream,
      TEMPLATE_ARGUMENTS(block_width, block_height),
      /* kernel parameters */
      raw_to_float_depth,
      depth_buffer,
      *result_buffer);
  CUDA_CHECK();
  
  float results_cpu[2];
  cudaMemcpyAsync(results_cpu,
                  result_buffer->address(),
                  2 * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);
  
  *keyframe_min_depth = results_cpu[0];
  *keyframe_max_depth = results_cpu[1];
}

}
