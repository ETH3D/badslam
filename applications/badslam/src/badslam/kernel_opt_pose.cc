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

#include "badslam/kernel_opt_pose.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void AccumulatePoseEstimationCoeffsCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    cudaTextureObject_t color_texture,
    const CUDAMatrix3x4& frame_T_global_estimate,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    bool debug,
    u32* residual_count,
    float* residual_sum,
    float* H,
    float* b,
    PoseEstimationHelperBuffers* helper_buffers) {
  CHECK(use_depth_residuals || use_descriptor_residuals);
  CUDA_CHECK();
  
  CHECK_GT(surfels_size, 0);  // This function is only intended for surfels_size > 0.
  
  // TODO: Clear in a single kernel call to avoid having the kernel call overhead
  //       multiple times; alternatively, try whether using multiple GPU memsets
  //       is better. Another alternative would be to merge the (non-debug) buffers to a
  //       large one. This would also allow to call DownloadAsync() only once later.
  if (debug) {
    helper_buffers->residual_count_buffer.Clear(0, stream);
    helper_buffers->residual_buffer.Clear(0, stream);
  }
  helper_buffers->H_buffer.Clear(0, stream);
  helper_buffers->b_buffer.Clear(0, stream);
  
  CallAccumulatePoseEstimationCoeffsCUDAKernel(
      stream,
      debug,
      use_depth_residuals,
      use_descriptor_residuals,
      CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, depth_buffer, normals_buffer, frame_T_global_estimate),
      CreateDepthToColorPixelCorner(depth_camera, color_camera),
      CreatePixelCenterProjector(color_camera),
      CreatePixelCornerProjector(color_camera),
      CreatePixelCenterUnprojector(depth_camera),
      color_texture,
      helper_buffers->residual_count_buffer.ToCUDA(),
      helper_buffers->residual_buffer.ToCUDA(),
      helper_buffers->H_buffer.ToCUDA(),
      helper_buffers->b_buffer.ToCUDA());
  
  if (debug) {
    helper_buffers->residual_count_buffer.DownloadAsync(stream, residual_count);
    helper_buffers->residual_buffer.DownloadAsync(stream, residual_sum);
  }
  helper_buffers->H_buffer.DownloadAsync(stream, H);
  helper_buffers->b_buffer.DownloadAsync(stream, b);
  cudaStreamSynchronize(stream);
}

void AccumulatePoseEstimationCoeffsFromImagesCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    float baseline_fx,
    float threshold_factor,
    const CUDABuffer<float>& downsampled_depth,
    const CUDABuffer<u16>& downsampled_normals,
    cudaTextureObject_t downsampled_color,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer<float>& surfel_depth,
    const CUDABuffer<u16>& surfel_normals,
    const CUDABuffer<uchar>& surfel_color,
    u32* residual_count,
    float* residual_sum,
    float* H,
    float* b,
    bool debug,
    CUDABuffer<float>* debug_residual_image,
    PoseEstimationHelperBuffers* helper_buffers,
    bool use_gradmag) {
  CUDA_CHECK();
  
  // TODO: Clear in a single kernel call?
  if (debug) {
    debug_residual_image->Clear(numeric_limits<float>::quiet_NaN(), stream);
    
    helper_buffers->residual_count_buffer.Clear(0, stream);
    helper_buffers->residual_buffer.Clear(0, stream);
  }
  helper_buffers->H_buffer.Clear(0, stream);
  helper_buffers->b_buffer.Clear(0, stream);
  
  if (use_gradmag) {
    CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradMag(
        stream,
        debug,
        use_depth_residuals,
        use_descriptor_residuals,
        CreatePixelCornerProjector(depth_camera),
        CreatePixelCenterProjector(color_camera),
        CreatePixelCenterUnprojector(depth_camera),
        baseline_fx,
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth.ToCUDA(),
        surfel_normals.ToCUDA(),
        surfel_color.ToCUDA(),
        downsampled_depth.ToCUDA(),
        downsampled_normals.ToCUDA(),
        downsampled_color,
        helper_buffers->residual_count_buffer.ToCUDA(),
        helper_buffers->residual_buffer.ToCUDA(),
        helper_buffers->H_buffer.ToCUDA(),
        helper_buffers->b_buffer.ToCUDA(),
        debug_residual_image ? &debug_residual_image->ToCUDA() : nullptr);
  } else {
    CallAccumulatePoseEstimationCoeffsFromImagesCUDAKernel_GradientXY(
        stream,
        debug,
        use_depth_residuals,
        use_descriptor_residuals,
        CreatePixelCornerProjector(depth_camera),
        CreatePixelCenterProjector(color_camera),
        CreatePixelCenterUnprojector(depth_camera),
        baseline_fx,
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth.ToCUDA(),
        surfel_normals.ToCUDA(),
        surfel_color.ToCUDA(),
        downsampled_depth.ToCUDA(),
        downsampled_normals.ToCUDA(),
        downsampled_color,
        helper_buffers->residual_count_buffer.ToCUDA(),
        helper_buffers->residual_buffer.ToCUDA(),
        helper_buffers->H_buffer.ToCUDA(),
        helper_buffers->b_buffer.ToCUDA(),
        debug_residual_image ? &debug_residual_image->ToCUDA() : nullptr);
  }
  CUDA_CHECK();
  
  if (debug) {
    helper_buffers->residual_count_buffer.DownloadAsync(stream, residual_count);
    helper_buffers->residual_buffer.DownloadAsync(stream, residual_sum);
  }
  helper_buffers->H_buffer.DownloadAsync(stream, H);
  helper_buffers->b_buffer.DownloadAsync(stream, b);
  cudaStreamSynchronize(stream);
}

void ComputeCostAndResidualCountFromImagesCUDA(
    cudaStream_t stream,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    float baseline_fx,
    float threshold_factor,
    const CUDABuffer<float>& downsampled_depth,
    const CUDABuffer<u16>& downsampled_normals,
    cudaTextureObject_t downsampled_color,
    const CUDAMatrix3x4& estimate_frame_T_surfel_frame,
    const CUDABuffer<float>& surfel_depth,
    const CUDABuffer<u16>& surfel_normals,
    const CUDABuffer<uchar>& surfel_color,
    u32* residual_count,
    float* residual_sum,
    PoseEstimationHelperBuffers* helper_buffers,
    bool use_gradmag) {
  CUDA_CHECK();
  
  // TODO: Clear in a single kernel call?
  helper_buffers->residual_count_buffer.Clear(0, stream);
  helper_buffers->residual_buffer.Clear(0, stream);
  
  if (use_gradmag) {
    CallComputeCostAndResidualCountFromImagesCUDAKernel_GradMag(
        stream,
        use_depth_residuals,
        use_descriptor_residuals,
        CreatePixelCornerProjector(depth_camera),
        CreatePixelCenterUnprojector(depth_camera),
        baseline_fx,
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth.ToCUDA(),
        surfel_normals.ToCUDA(),
        surfel_color.ToCUDA(),
        downsampled_depth.ToCUDA(),
        downsampled_normals.ToCUDA(),
        downsampled_color,
        helper_buffers->residual_count_buffer.ToCUDA(),
        helper_buffers->residual_buffer.ToCUDA());
  } else {
    ComputeCostAndResidualCountFromImagesCUDAKernel_GradientXY(
        stream,
        use_depth_residuals,
        use_descriptor_residuals,
        CreatePixelCornerProjector(depth_camera),
        CreatePixelCenterUnprojector(depth_camera),
        baseline_fx,
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        threshold_factor,
        estimate_frame_T_surfel_frame,
        surfel_depth.ToCUDA(),
        surfel_normals.ToCUDA(),
        surfel_color.ToCUDA(),
        downsampled_depth.ToCUDA(),
        downsampled_normals.ToCUDA(),
        downsampled_color,
        helper_buffers->residual_count_buffer.ToCUDA(),
        helper_buffers->residual_buffer.ToCUDA());
  }
  CUDA_CHECK();
  
  helper_buffers->residual_count_buffer.DownloadAsync(stream, residual_count);
  helper_buffers->residual_buffer.DownloadAsync(stream, residual_sum);
  cudaStreamSynchronize(stream);
}

}
