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
#include "badslam/cuda_matrix.cuh"
#include "badslam/surfel_projection_nvcc_only.cuh"
#include "badslam/util.cuh"
#include "badslam/util_nvcc_only.cuh"

namespace vis {

__global__ void CreateSurfelsForKeyframeCUDASerializingKernel(
    int sparse_surfel_cell_size,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<uchar4> color_buffer,
    CUDABuffer_<u32> supporting_surfels,
    CUDABuffer_<u8> new_surfel_flag_vector) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  constexpr bool kSemiDense = false;
  constexpr int kSemiDenseThreshold = 5;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    // TODO: Is this border necessary here?
    constexpr int kBorder = 1;
    // TODO: This atomicCAS() will lead to the selection of a (relatively) random high-res
    //       pixel within the sparsified cell. Can we instead select a good
    //       pixel according to some criteria (e.g, having high gradient magnitude)?
    bool new_surfel = x >= kBorder &&
                      y >= kBorder &&
                      x < depth_buffer.width() - kBorder &&
                      y < depth_buffer.height() - kBorder &&
                      !(depth_buffer(y, x) & kInvalidDepthBit) &&
                      (!kSemiDense || (::abs(color_buffer(y, x).w - color_buffer(y, x + 1).w) >= kSemiDenseThreshold ||
                                       ::abs(color_buffer(y, x).w - color_buffer(y + 1, x).w) >= kSemiDenseThreshold ||
                                       ::abs(color_buffer(y, x).w - color_buffer(y - 1, x).w) >= kSemiDenseThreshold ||
                                       ::abs(color_buffer(y, x).w - color_buffer(y, x - 1).w) >= kSemiDenseThreshold)) &&
                      atomicCAS(&supporting_surfels(y / sparse_surfel_cell_size, x / sparse_surfel_cell_size), kInvalidIndex, 0) == kInvalidIndex;
    u32 seq_index = x + y * depth_buffer.width();
    new_surfel_flag_vector(0, seq_index) = new_surfel ? 1 : 0;
  }
}

void CallCreateSurfelsForKeyframeCUDASerializingKernel(
    cudaStream_t stream,
    int sparse_surfel_cell_size,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<uchar4>& color_buffer,
    const CUDABuffer_<u32>& supporting_surfels,
    const CUDABuffer_<u8>& new_surfel_flag_vector) {
  CUDA_AUTO_TUNE_2D(
      CreateSurfelsForKeyframeCUDASerializingKernel,
      32, 32,
      depth_buffer.width(), depth_buffer.height(),
      0, stream,
      /* kernel parameters */
      sparse_surfel_cell_size,
      depth_buffer,
      color_buffer,
      supporting_surfels,
      new_surfel_flag_vector);
  CUDA_CHECK();
}


__device__ __forceinline__ void CreateNewSurfel(
    u32 x,
    u32 y,
    u32 surfel_index,
    const PixelCenterUnprojector& unprojector,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    const CUDAMatrix3x4& global_T_frame,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    const CUDABuffer_<u16>& radius_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<float>& surfels) {
  float calibrated_depth = RawToCalibratedDepth(
      depth_params.a,
      depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                  x / depth_params.sparse_surfel_cell_size),
      depth_params.raw_to_float_depth,
      depth_buffer(y, x));
  
  float3 surfel_global_position = global_T_frame * unprojector.UnprojectPoint(x, y, calibrated_depth);
  SurfelSetPosition(&surfels, surfel_index, surfel_global_position);
  float3 surfel_local_normal = U16ToImageSpaceNormal(normals_buffer(y, x));
  float3 surfel_global_normal = global_T_frame.Rotate(surfel_local_normal);
  SurfelSetNormal(&surfels, surfel_index, surfel_global_normal);
  float surfel_radius_squared = __half2float(__ushort_as_half(radius_buffer(y, x)));
  SurfelSetRadiusSquared(&surfels, surfel_index, surfel_radius_squared);
  
  float2 color_pxy;
  TransformDepthToColorPixelCorner(make_float2(x + 0.5f, y + 0.5f), depth_to_color, &color_pxy);
  float4 color = tex2D<float4>(color_texture, color_pxy.x, color_pxy.y);
  
  
  float2 t1_pxy, t2_pxy;
  ComputeTangentProjections(
      surfel_global_position,
      surfel_global_normal,
      surfel_radius_squared,
      frame_T_global,
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
  
  SurfelSetColor(&surfels, surfel_index, make_uchar4(
      255.f * color.x,
      255.f * color.y,
      255.f * color.z,
      0));
  
  surfels(kSurfelDescriptor1, surfel_index) = descriptor_1;
  surfels(kSurfelDescriptor2, surfel_index) = descriptor_2;
}


__global__ void WriteNewSurfelIndexAndInitializeObservationsCUDAKernel(
    u32 pixel_count,
    CUDABuffer_<u8> new_surfel_flag_vector,
    CUDABuffer_<u32> new_surfel_indices,
    u16* observation_vector,
    u16* free_space_violation_vector,
    u32* new_surfel_index_list) {
  unsigned int seq_index = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (seq_index < pixel_count) {
    if (new_surfel_flag_vector(0, seq_index) != 1) {
      return;
    }
    
    // One needs to be subtracted because of using an inclusive prefix sum for
    // computing new_surfel_indices, i.e., the first new surfel has
    // new_surfel_indices(...) == 1.
    u32 surfel_index = new_surfel_indices(0, seq_index) - 1;
    
    new_surfel_index_list[surfel_index] = seq_index;
    observation_vector[surfel_index] = 1;
    free_space_violation_vector[surfel_index] = 0;
  }
}

void CallWriteNewSurfelIndexAndInitializeObservationsCUDAKernel(
    cudaStream_t stream,
    u32 pixel_count,
    const CUDABuffer_<u8>& new_surfel_flag_vector,
    const CUDABuffer_<u32>& new_surfel_indices,
    u16* observation_vector,
    u16* free_space_violation_vector,
    u32* new_surfel_index_list) {
  CUDA_AUTO_TUNE_1D(
      WriteNewSurfelIndexAndInitializeObservationsCUDAKernel,
      1024,
      pixel_count,
      0, stream,
      /* kernel parameters */
      pixel_count,
      new_surfel_flag_vector,
      new_surfel_indices,
      observation_vector,
      free_space_violation_vector,
      new_surfel_index_list);
  CUDA_CHECK();
}


__global__ void CountObservationsForNewSurfelsCUDAKernel(
    int new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    DepthParameters depth_params,
    PixelCenterUnprojector unprojector,
    CUDABuffer_<u16> new_surfel_depth_buffer,
    CUDABuffer_<u16> new_surfel_normals_buffer,
    CUDAMatrix3x4 covis_T_frame,
    PixelCornerProjector projector,
    CUDABuffer_<u16> covis_depth_buffer,
    CUDABuffer_<u16> covis_normals_buffer) {
  const unsigned int new_surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (new_surfel_index < new_surfel_count) {
    u32 new_surfel_pixel_index = new_surfel_index_list[new_surfel_index];
    u32 y = new_surfel_pixel_index / new_surfel_depth_buffer.width();
    u32 x = new_surfel_pixel_index - y * new_surfel_depth_buffer.width();
    
    float calibrated_depth = RawToCalibratedDepth(
        depth_params.a,
        depth_params.cfactor_buffer(y / depth_params.sparse_surfel_cell_size,
                                    x / depth_params.sparse_surfel_cell_size),
        depth_params.raw_to_float_depth,
        new_surfel_depth_buffer(y, x));
    float3 surfel_input_position = unprojector.UnprojectPoint(x, y, calibrated_depth);
    
    float3 surfel_local_position;
    if (!covis_T_frame.MultiplyIfResultZIsPositive(surfel_input_position, &surfel_local_position)) {
      return;
    }
    int px, py;
    if (!ProjectSurfelToImage(
        covis_depth_buffer.width(), covis_depth_buffer.height(),
        projector,
        surfel_local_position,
        &px, &py)) {
      return;
    }
    
    // Check for depth compatibility.
    bool is_free_space_violation = false;
    if (!IsAssociatedWithPixel<true>(
        surfel_local_position,
        new_surfel_normals_buffer,
        x, y,
        covis_T_frame,
        covis_normals_buffer,
        px, py,
        depth_params,
        covis_depth_buffer(py, px),
        kDepthResidualDefaultTukeyParam,
        unprojector,
        &is_free_space_violation)) {
      if (is_free_space_violation) {
        free_space_violation_vector[new_surfel_index] += 1;
      }
      return;
    }
    
    // Accumulate.
    observation_vector[new_surfel_index] += 1;
  }
}

void CallCountObservationsForNewSurfelsCUDAKernel(
    cudaStream_t stream,
    int new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    const DepthParameters& depth_params,
    const PixelCenterUnprojector& unprojector,
    const CUDABuffer_<u16>& new_surfel_depth_buffer,
    const CUDABuffer_<u16>& new_surfel_normals_buffer,
    const CUDAMatrix3x4& covis_T_frame,
    const PixelCornerProjector& projector,
    const CUDABuffer_<u16>& covis_depth_buffer,
    const CUDABuffer_<u16>& covis_normals_buffer) {
  CUDA_AUTO_TUNE_1D(
      CountObservationsForNewSurfelsCUDAKernel,
      1024,
      new_surfel_count,
      0, stream,
      /* kernel parameters */
      new_surfel_count,
      new_surfel_index_list,
      observation_vector,
      free_space_violation_vector,
      depth_params,
      unprojector,
      new_surfel_depth_buffer,
      new_surfel_normals_buffer,
      covis_T_frame,
      projector,
      covis_depth_buffer,
      covis_normals_buffer);
  CUDA_CHECK();
}


__global__ void FilterNewSurfelsCUDAKernel(
    u16 min_observation_count,
    u32 new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    CUDABuffer_<u8> new_surfel_flag_vector) {
  const unsigned int new_surfel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (new_surfel_index < new_surfel_count) {
    u16 observation_count = observation_vector[new_surfel_index];
    
    if (observation_count < min_observation_count ||
        free_space_violation_vector[new_surfel_index] > observation_count) {
      // Do not create this surfel.
      u32 new_surfel_pixel_index = new_surfel_index_list[new_surfel_index];
      new_surfel_flag_vector(0, new_surfel_pixel_index) = 0;
    }
  }
}

void CallFilterNewSurfelsCUDAKernel(
    cudaStream_t stream,
    u16 min_observation_count,
    u32 new_surfel_count,
    u32* new_surfel_index_list,
    u16* observation_vector,
    u16* free_space_violation_vector,
    const CUDABuffer_<u8>& new_surfel_flag_vector) {
  CUDA_AUTO_TUNE_1D(
      FilterNewSurfelsCUDAKernel,
      1024,
      new_surfel_count,
      0, stream,
      /* kernel parameters */
      min_observation_count,
      new_surfel_count,
      new_surfel_index_list,
      observation_vector,
      free_space_violation_vector,
      new_surfel_flag_vector);
}


__global__ void CreateSurfelsForKeyframeCUDACreationAppendKernel(
    PixelCenterUnprojector unprojector,
    DepthToColorPixelCorner depth_to_color,
    PixelCornerProjector color_corner_projector,
    CUDAMatrix3x4 global_T_frame,
    CUDAMatrix3x4 frame_T_global,
    DepthParameters depth_params,
    CUDABuffer_<u16> depth_buffer,
    CUDABuffer_<u16> normals_buffer,
    CUDABuffer_<u16> radius_buffer,
    cudaTextureObject_t color_texture,
    CUDABuffer_<u8> new_surfel_flag_vector,
    CUDABuffer_<u32> new_surfel_indices,
    u32 surfels_size,
    CUDABuffer_<float> surfels) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x < depth_buffer.width() && y < depth_buffer.height()) {
    u32 seq_index = x + y * depth_buffer.width();
    if (new_surfel_flag_vector(0, seq_index) != 1) {
      return;
    }
    
    // Compute the index at which to output the new surfel (appending after the
    // old surfels_size). One needs to be subtracted because of using an
    // inclusive prefix sum for computing new_surfel_indices, i.e., the first
    // new surfel has new_surfel_indices(...) == 1.
    u32 surfel_index = surfels_size + new_surfel_indices(0, seq_index) - 1;
    
    CreateNewSurfel(x, y, surfel_index, unprojector, depth_to_color, color_corner_projector, global_T_frame, frame_T_global,
                    depth_params, depth_buffer, normals_buffer, radius_buffer,
                    color_texture, surfels);
  }
}

void CallCreateSurfelsForKeyframeCUDACreationAppendKernel(
    cudaStream_t stream,
    const PixelCenterUnprojector& unprojector,
    const DepthToColorPixelCorner& depth_to_color,
    const PixelCornerProjector& color_corner_projector,
    const CUDAMatrix3x4& global_T_frame,
    const CUDAMatrix3x4& frame_T_global,
    const DepthParameters& depth_params,
    const CUDABuffer_<u16>& depth_buffer,
    const CUDABuffer_<u16>& normals_buffer,
    const CUDABuffer_<u16>& radius_buffer,
    cudaTextureObject_t color_texture,
    const CUDABuffer_<u8>& new_surfel_flag_vector,
    const CUDABuffer_<u32>& new_surfel_indices,
    u32 surfels_size,
    const CUDABuffer_<float>& surfels) {
  CUDA_AUTO_TUNE_2D(
      CreateSurfelsForKeyframeCUDACreationAppendKernel,
      32, 32,
      depth_buffer.width(), depth_buffer.height(),
      0, stream,
      /* kernel parameters */
      unprojector,
      depth_to_color,
      color_corner_projector,
      global_T_frame,
      frame_T_global,
      depth_params,
      depth_buffer,
      normals_buffer,
      radius_buffer,
      color_texture,
      new_surfel_flag_vector,
      new_surfel_indices,
      surfels_size,
      surfels);
  CUDA_CHECK();
}


u32 CreateSurfelsForKeyframeCUDA_CountNewSurfels(
    cudaStream_t stream,
    u32 pixel_count,
    void** new_surfels_temp_storage,
    usize* new_surfels_temp_storage_bytes,
    CUDABuffer_<u8>* new_surfel_flag_vector,
    CUDABuffer_<u32>* new_surfel_indices) {
  // Indices for the new surfels are computed with a parallel inclusive prefix sum from CUB.
  if (*new_surfels_temp_storage_bytes == 0) {
    cub::DeviceScan::InclusiveSum(
        *new_surfels_temp_storage,
        *new_surfels_temp_storage_bytes,
        new_surfel_flag_vector->address(),
        new_surfel_indices->address(),
        pixel_count,
        stream);
    
    cudaMalloc(new_surfels_temp_storage, *new_surfels_temp_storage_bytes);
  }
  
  cub::DeviceScan::InclusiveSum(
      *new_surfels_temp_storage,
      *new_surfels_temp_storage_bytes,
      new_surfel_flag_vector->address(),
      new_surfel_indices->address(),
      pixel_count,
      stream);
  CUDA_CHECK();
  
  // Read back the number of new surfels to the CPU by reading the last element
  // in new_surfel_indices.
  u32 new_surfel_count;
  cudaMemcpyAsync(
      &new_surfel_count,
      reinterpret_cast<u8*>(new_surfel_indices->address()) + ((pixel_count - 1) * sizeof(u32)),
      sizeof(u32),
      cudaMemcpyDeviceToHost,
      stream);
  cudaStreamSynchronize(stream);
  return new_surfel_count;
}

}
