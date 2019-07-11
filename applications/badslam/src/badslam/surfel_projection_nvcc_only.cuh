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
#include "badslam/surfel_projection.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

// Tests whether a pixel corresponds to a surfel (i.e., probably represents the
// same surface).
template <bool return_free_space_violations, bool return_surfel_normal>
__forceinline__ __device__ bool IsAssociatedWithPixel(
    const CUDABuffer_<float>& surfels,
    u32 surfel_index,
    const float3& surfel_local_position,
    const CUDAMatrix3x4& frame_T_global,
    const CUDABuffer_<u16>& normals_buffer,
    int px,
    int py,
    const DepthParameters& depth_params,
    u16 measured_depth,
    float depth_tukey_parameter,
    const PixelCenterUnprojector& center_unprojector,
    bool* is_free_space_violation,
    float3* surfel_normal,
    float* out_calibrated_depth) {
  if (measured_depth & kInvalidDepthBit) {
    return false;
  }
  
  float calibrated_depth = RawToCalibratedDepth(
      depth_params.a,
      depth_params.cfactor_buffer(py / depth_params.sparse_surfel_cell_size,
                                  px / depth_params.sparse_surfel_cell_size),
      depth_params.raw_to_float_depth,
      measured_depth);
  if (out_calibrated_depth) {
    *out_calibrated_depth = calibrated_depth;
  }
  
  float3 surfel_local_normal;
  if (return_surfel_normal) {
    *surfel_normal = SurfelGetNormal(surfels, surfel_index);
    surfel_local_normal = frame_T_global.Rotate(*surfel_normal);
  } else {
    surfel_local_normal = frame_T_global.Rotate(SurfelGetNormal(surfels, surfel_index));
  }
  
  // Compute association depth difference threshold
  float depth_residual_stddev_estimate = ComputeDepthResidualStddevEstimate(
      center_unprojector.nx(px), center_unprojector.ny(py), calibrated_depth, surfel_local_normal, depth_params.baseline_fx);  // TODO: return to caller if useful
  const float depth_difference_threshold = depth_tukey_parameter * depth_residual_stddev_estimate;
  
  // Check whether the depth is similar enough to consider the measurement to belong to the surfel
  if (return_free_space_violations) {
    float depth_difference = calibrated_depth - surfel_local_position.z;
    if (depth_difference > depth_difference_threshold) {
      *is_free_space_violation = true;
      return false;
    } else if (depth_difference < -depth_difference_threshold) {
      return false;
    }
  } else {
    if (fabs(surfel_local_position.z - calibrated_depth) > depth_difference_threshold) {
      return false;
    }
  }
  
  // Check whether the surfel normal looks towards the camera (instead of away from it).
  float surfel_distance = Norm(surfel_local_position);
  float surfel_vs_camera_dir_dot_angle = (1.0f / surfel_distance) * Dot(surfel_local_position, surfel_local_normal);
  if (surfel_vs_camera_dir_dot_angle > 0) {
    return false;
  }
  
  // Check whether the surfel normal is compatible with the measurement normal.
  float3 local_normal = U16ToImageSpaceNormal(normals_buffer(py, px));  // TODO: export this to calling functions if they need it to improve performance
  
  float surfel_vs_measurement_dot_angle = Dot(surfel_local_normal, local_normal);
  if (surfel_vs_measurement_dot_angle < cos_normal_compatibility_threshold) {
//     if (return_free_space_violations && surfel_local_position.z < calibrated_depth) {
//       // NOTE: Careful here, this might lead to a bias since we only remove
//       //       surfels on one side of the surface.
//       *is_free_space_violation = true;
//     }
    return false;
  }
  
  return true;
}

// Version of IsAssociatedWithPixel() for using a surfel that is implicitly
// defined by a pixel.
template <bool return_free_space_violations>
__forceinline__ __device__ bool IsAssociatedWithPixel(
    const float3& surfel_local_position,
    const CUDABuffer_<u16>& surfel_normals_buffer,
    int x,
    int y,
    const CUDAMatrix3x4& test_frame_T_surfel_frame,
    const CUDABuffer_<u16>& test_normals_buffer,
    int px,
    int py,
    const DepthParameters& depth_params,
    u16 pixel_measured_depth,
    float depth_tukey_parameter,
    const PixelCenterUnprojector& center_unprojector,
    bool* is_free_space_violation) {
  if (pixel_measured_depth & kInvalidDepthBit) {
    return false;
  }
  
  float pixel_calibrated_depth = RawToCalibratedDepth(
      depth_params.a,
      depth_params.cfactor_buffer(py / depth_params.sparse_surfel_cell_size,
                                  px / depth_params.sparse_surfel_cell_size),
      depth_params.raw_to_float_depth,
      pixel_measured_depth);
  
  float3 surfel_local_normal;
  return IsAssociatedWithPixel<return_free_space_violations>(
      surfel_local_position,
      surfel_normals_buffer,
      x,
      y,
      test_frame_T_surfel_frame,
      test_normals_buffer,
      px,
      py,
      pixel_calibrated_depth,
      depth_tukey_parameter,
      depth_params.baseline_fx,
      center_unprojector,
      is_free_space_violation,
      &surfel_local_normal);
}

// Version of IsAssociatedWithPixel() for using a surfel that is implicitly
// defined by a pixel.
template <bool return_free_space_violations>
__forceinline__ __device__ bool IsAssociatedWithPixel(
    const float3& surfel_local_position,
    const CUDABuffer_<u16>& surfel_normals_buffer,
    int x,
    int y,
    const CUDAMatrix3x4& test_frame_T_surfel_frame,
    const CUDABuffer_<u16>& test_normals_buffer,
    int px,
    int py,
    float pixel_calibrated_depth,
    float depth_tukey_parameter,
    float baseline_fx,
    const PixelCenterUnprojector& center_unprojector,
    bool* is_free_space_violation,
    float3* surfel_local_normal) {
  *surfel_local_normal = test_frame_T_surfel_frame.Rotate(U16ToImageSpaceNormal(surfel_normals_buffer(y, x)));
  
  // Compute association depth difference threshold
  float depth_residual_stddev_estimate = ComputeDepthResidualStddevEstimate(
      center_unprojector.nx(px), center_unprojector.ny(py), pixel_calibrated_depth, *surfel_local_normal, baseline_fx);  // TODO: return to caller if useful
  const float depth_difference_threshold = depth_tukey_parameter * depth_residual_stddev_estimate;
  
  // Check whether the depth is similar enough to consider the measurement to belong to the surfel
  if (return_free_space_violations) {
    float depth_difference = pixel_calibrated_depth - surfel_local_position.z;
    if (depth_difference > depth_difference_threshold) {
      *is_free_space_violation = true;
      return false;
    } else if (depth_difference < -depth_difference_threshold) {
      return false;
    }
  } else {
    if (fabs(surfel_local_position.z - pixel_calibrated_depth) > depth_difference_threshold) {
      return false;
    }
  }
  
  // Check whether the surfel normal looks towards the camera (instead of away from it).
  float surfel_distance = Norm(surfel_local_position);
  float surfel_vs_camera_dir_dot_angle = (1.0f / surfel_distance) * Dot(surfel_local_position, *surfel_local_normal);
  if (surfel_vs_camera_dir_dot_angle > 0) {
    return false;
  }
  
  // Check whether the surfel normal is compatible with the measurement normal.
  float3 local_normal = U16ToImageSpaceNormal(test_normals_buffer(py, px));
  
  float surfel_vs_measurement_dot_angle = Dot(*surfel_local_normal, local_normal);
  if (surfel_vs_measurement_dot_angle < cos_normal_compatibility_threshold) {
//     if (return_free_space_violations && surfel_local_position.z < pixel_calibrated_depth) {
//       // NOTE: Careful here, this might lead to a bias since we only remove
//       //       surfels on one side of the surface.
//       *is_free_space_violation = true;
//     }
    return false;
  }
  
  return true;
}


// -----------------------------------------------------------------------------


// Groups result variables of surfel projection.
struct SurfelProjectionResult6 {
  float3 surfel_global_position;
  
  // Local position of the surfel in the keyframe coordinate system.
  float3 surfel_local_position;
  
  // Global normal vector of the surfel.
  float3 surfel_normal;
  
  // Calibrated depth value of the pixel the surfel projects to.
  float pixel_calibrated_depth;
  
  // Integer coordinates of the pixel the surfel projects to.
  int px;
  int py;
  
  // Float coordinates of the pixel the surfel projects to ("pixel corner" convention).
  float2 pxy;
};

// Groups result variables of surfel projection.
struct SurfelProjectionResult5 {
  // Local position of the surfel in the keyframe coordinate system.
  float3 surfel_local_position;
  
  // Global normal vector of the surfel.
  float3 surfel_normal;
  
  // Calibrated depth value of the pixel the surfel projects to.
  float pixel_calibrated_depth;
  
  // Integer coordinates of the pixel the surfel projects to.
  int px;
  int py;
};

// Groups result variables of surfel projection.
struct SurfelProjectionResultXY {
  // Integer coordinates of the pixel the surfel projects to.
  int px;
  int py;
};

// Groups result variables of surfel projection.
struct SurfelProjectionResultXYFloat {
  float2 pxy;
};

// Groups result variables of surfel projection.
struct SurfelProjectionResultXYFreeSpace {
  // Integer coordinates of the pixel the surfel projects to.
  int px;
  int py;
  
  bool is_free_space_violation;
};

// Projects a surfel to a pixel. Returns true if it projects to a corresponding
// pixel, and outputs the projection result.
__forceinline__ __device__ bool SurfelProjectsToAssociatedPixel(
    unsigned int surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    SurfelProjectionResult5* result) {
  if (surfel_index < surfel_projection.surfels_size) {
    // Project the surfel onto depth_buffer to find the corresponding pixel
    float3 global_position = SurfelGetPosition(surfel_projection.surfels, surfel_index);
    if (surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(global_position, &result->surfel_local_position)) {
      if (ProjectSurfelToImage(
          surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
          surfel_projection.projector,
          result->surfel_local_position,
          &result->px, &result->py)) {
        // Check whether the surfel gets associated with the pixel.
        if (IsAssociatedWithPixel<false, true>(
            surfel_projection.surfels, surfel_index, result->surfel_local_position, surfel_projection.frame_T_global,
            surfel_projection.normals_buffer, result->px, result->py, surfel_projection.depth_params,
            surfel_projection.depth_buffer(result->py, result->px),
            kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
            nullptr, &result->surfel_normal, &result->pixel_calibrated_depth)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Projects a surfel to a pixel. Returns true if it projects to a corresponding
// pixel, and outputs the projection result.
__forceinline__ __device__ bool SurfelProjectsToAssociatedPixel(
    unsigned int surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    SurfelProjectionResult6* result) {
  if (surfel_index < surfel_projection.surfels_size) {
    // Project the surfel onto depth_buffer to find the corresponding pixel
    result->surfel_global_position = SurfelGetPosition(surfel_projection.surfels, surfel_index);
    if (surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(result->surfel_global_position, &result->surfel_local_position)) {
      if (ProjectSurfelToImage(
          surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
          surfel_projection.projector,
          result->surfel_local_position,
          &result->px, &result->py,
          &result->pxy)) {
        // Check whether the surfel gets associated with the pixel.
        if (IsAssociatedWithPixel<false, true>(
            surfel_projection.surfels, surfel_index, result->surfel_local_position, surfel_projection.frame_T_global,
            surfel_projection.normals_buffer, result->px, result->py, surfel_projection.depth_params,
            surfel_projection.depth_buffer(result->py, result->px),
            kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
            nullptr, &result->surfel_normal, &result->pixel_calibrated_depth)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Projects a surfel to a pixel. Returns true if any surfel from the thread's
// CUDA block projects to a corresponding pixel, and false otherwise. Outputs
// the projection result.
__forceinline__ __device__ bool AnySurfelProjectsToAssociatedPixel(
    unsigned int* surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    bool* visible,
    SurfelProjectionResult6* result) {
  *visible = *surfel_index < surfel_projection.surfels_size;
  if (!*visible) {
    *surfel_index = 0;
  }
  
  // Project the surfel onto depth_buffer to find the corresponding pixel
  result->surfel_global_position = SurfelGetPosition(surfel_projection.surfels, *surfel_index);
  if (!surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(result->surfel_global_position, &result->surfel_local_position)) {
    *visible = false;
  }
  if (!*visible ||
      !ProjectSurfelToImage(
      surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
      surfel_projection.projector,
      result->surfel_local_position,
      &result->px, &result->py,
      &result->pxy)) {
    result->px = 0;
    result->py = 0;
    *visible = false;
  }
  
  // Early exit if all threads within the block (!) have invisible surfels.
  // Checking for invisible threads within the warp (using __all(), for example)
  // is not sufficient since we do block-wide collective operations later.
  if (__syncthreads_or(*visible) == 0) {
    return false;
  }
  
  // Check for depth compatibility.
  if (!IsAssociatedWithPixel<false, true>(
      surfel_projection.surfels, *surfel_index, result->surfel_local_position, surfel_projection.frame_T_global,
      surfel_projection.normals_buffer, result->px, result->py, surfel_projection.depth_params,
      surfel_projection.depth_buffer(result->py, result->px),
      kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
      nullptr, &result->surfel_normal, &result->pixel_calibrated_depth)) {
    *visible = false;
  }
  
  // Second early exit test (see above)
  if (__syncthreads_or(*visible) == 0) {
    return false;
  }
  
  return true;
}

// Projects a surfel to a pixel. Returns true if it projects to a corresponding
// pixel, and outputs the projection result.
__forceinline__ __device__ bool SurfelProjectsToAssociatedPixel(
    unsigned int surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    SurfelProjectionResultXY* result) {
  if (surfel_index < surfel_projection.surfels_size) {
    // Project the surfel onto depth_buffer to find the corresponding pixel
    float3 global_position = SurfelGetPosition(surfel_projection.surfels, surfel_index);
    float3 surfel_local_position;
    if (surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(global_position, &surfel_local_position)) {
      if (ProjectSurfelToImage(
          surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
          surfel_projection.projector,
          surfel_local_position,
          &result->px, &result->py)) {
        // Check whether the surfel gets associated with the pixel.
        if (IsAssociatedWithPixel<false, false>(
            surfel_projection.surfels, surfel_index, surfel_local_position, surfel_projection.frame_T_global,
            surfel_projection.normals_buffer, result->px, result->py, surfel_projection.depth_params,
            surfel_projection.depth_buffer(result->py, result->px),
            kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
            nullptr, nullptr, nullptr)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Projects a surfel to a pixel. Returns true if it projects to a corresponding
// pixel, and outputs the projection result.
__forceinline__ __device__ bool SurfelProjectsToAssociatedPixel(
    unsigned int surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    SurfelProjectionResultXYFloat* result) {
  if (surfel_index < surfel_projection.surfels_size) {
    // Project the surfel onto depth_buffer to find the corresponding pixel
    float3 global_position = SurfelGetPosition(surfel_projection.surfels, surfel_index);
    float3 surfel_local_position;
    if (surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(global_position, &surfel_local_position)) {
      int px, py;
      if (ProjectSurfelToImage(
          surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
          surfel_projection.projector,
          surfel_local_position,
          &px, &py,
          &result->pxy)) {
        // Check whether the surfel gets associated with the pixel.
        if (IsAssociatedWithPixel<false, false>(
            surfel_projection.surfels, surfel_index, surfel_local_position, surfel_projection.frame_T_global,
            surfel_projection.normals_buffer, px, py, surfel_projection.depth_params,
            surfel_projection.depth_buffer(py, px),
            kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
            nullptr, nullptr, nullptr)) {
          return true;
        }
      }
    }
  }
  return false;
}

// Projects a surfel to a pixel. Returns true if it projects to a corresponding
// pixel, and outputs the projection result.
__forceinline__ __device__ bool SurfelProjectsToAssociatedPixel(
    unsigned int surfel_index,
    const SurfelProjectionParameters& surfel_projection,
    SurfelProjectionResultXYFreeSpace* result) {
  result->is_free_space_violation = false;
  
  if (surfel_index < surfel_projection.surfels_size) {
    // Project the surfel onto depth_buffer to find the corresponding pixel
    float3 global_position = SurfelGetPosition(surfel_projection.surfels, surfel_index);
    float3 surfel_local_position;
    if (surfel_projection.frame_T_global.MultiplyIfResultZIsPositive(global_position, &surfel_local_position)) {
      if (ProjectSurfelToImage(
          surfel_projection.depth_buffer.width(), surfel_projection.depth_buffer.height(),
          surfel_projection.projector,
          surfel_local_position,
          &result->px, &result->py)) {
        // Check whether the surfel gets associated with the pixel.
        if (IsAssociatedWithPixel<true, false>(
            surfel_projection.surfels, surfel_index, surfel_local_position, surfel_projection.frame_T_global,
            surfel_projection.normals_buffer, result->px, result->py, surfel_projection.depth_params,
            surfel_projection.depth_buffer(result->py, result->px),
            kDepthResidualDefaultTukeyParam, surfel_projection.center_unprojector,
            &result->is_free_space_violation, nullptr, nullptr)) {
          return true;
        }
      }
    }
  }
  return false;
}

}
