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

#include "badslam/keyframe.h"

#include "badslam/surfel_projection.h"

namespace vis {

Keyframe::Keyframe(
    cudaStream_t stream,
    u32 frame_index,
    float min_depth,
    float max_depth,
    const CUDABuffer<u16>& depth_buffer,
    const CUDABuffer<u16>& normals_buffer,
    const CUDABuffer<u16>& radius_buffer,
    const CUDABuffer<uchar4>& color_buffer,
    const ImageFramePtr<u16, SE3f>& depth_frame,
    const ImageFramePtr<Vec3u8, SE3f>& color_frame)
    : frame_index_(frame_index),
      last_active_in_ba_iteration_(-1),
      last_covis_in_ba_iteration_(-1),
      min_depth_(min_depth),
      max_depth_(max_depth),
      depth_buffer_(depth_buffer.height(), depth_buffer.width()),
      normals_buffer_(normals_buffer.height(), normals_buffer.width()),
      radius_buffer_(radius_buffer.height(), radius_buffer.width()),
      color_buffer_(color_buffer.height(), color_buffer.width()),
      depth_frame_(depth_frame),
      color_frame_(color_frame) {
  CHECK_GT(min_depth, 0.f)
      << "Keyframe min depth must be larger than 0 since the frustum checks"
          " do not work properly otherwise.";
  
  // TODO: Avoid these copies by taking buffer ownership instead?
  depth_buffer_.SetTo(depth_buffer, stream);
  normals_buffer_.SetTo(normals_buffer, stream);
  radius_buffer_.SetTo(radius_buffer, stream);
  
  color_buffer_.SetTo(color_buffer, stream);
  color_buffer_.CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      &color_texture_);
  
  activation_ = Activation::kActive;
  
  // Make sure that any derived transformations are cached
  set_frame_T_global(frame_T_global());
}

Keyframe::Keyframe(
    cudaStream_t stream,
    u32 frame_index,
    const DepthParameters& depth_params,
    const PinholeCamera4f& depth_camera,
    const Image<u16>& depth_image,
    const Image<Vec3u8>& color_image,
    const SE3f& global_tr_frame)
    : frame_index_(frame_index),
      last_active_in_ba_iteration_(-1),
      last_covis_in_ba_iteration_(-1),
      depth_buffer_(depth_image.height(), depth_image.width()),
      normals_buffer_(depth_image.height(), depth_image.width()),
      radius_buffer_(depth_image.height(), depth_image.width()),
      color_buffer_(color_image.height(), color_image.width()) {
  // Perform color image preprocessing.
  CUDABuffer<uchar3> rgb_buffer(color_image.height(), color_image.width());
  rgb_buffer.UploadAsync(stream, reinterpret_cast<const Image<uchar3>&>(color_image));
  ComputeBrightnessCUDA(
      stream,
      rgb_buffer.ToCUDA(),
      &color_buffer_.ToCUDA());
  color_buffer_.CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      &color_texture_);
  
  // Perform depth image preprocessing.
  CUDABuffer<u16> depth_buffer(depth_image.height(), depth_image.width());
  depth_buffer_.UploadAsync(stream, depth_image);
  
  ComputeNormalsCUDA(
      stream,
      CreatePixelCenterUnprojector(depth_camera),
      depth_params,
      depth_buffer_.ToCUDA(),
      &depth_buffer.ToCUDA(),
      &normals_buffer_.ToCUDA());
  
  CUDABufferPtr<float> min_max_depth_init_buffer_;
  CUDABufferPtr<float> min_max_depth_result_buffer_;
  ComputeMinMaxDepthCUDA_InitializeBuffers(
      &min_max_depth_init_buffer_,
      &min_max_depth_result_buffer_);
  
  ComputePointRadiiAndRemoveIsolatedPixelsCUDA(
      stream,
      CreatePixelCenterUnprojector(depth_camera),
      depth_params.raw_to_float_depth,
      depth_buffer.ToCUDA(),
      &radius_buffer_.ToCUDA(),
      &depth_buffer_.ToCUDA());
  
  ComputeMinMaxDepthCUDA(
      stream,
      depth_buffer.ToCUDA(),
      depth_params.raw_to_float_depth,
      min_max_depth_init_buffer_->ToCUDA(),
      &min_max_depth_result_buffer_->ToCUDA(),
      &min_depth_,
      &max_depth_);
  
  // Create new ImageFramePtr for the CPU data.
  // NOTE: We do not actually store the images in these frames. This could be
  //       done such that it would be possible to clear the GPU buffers if
  //       GPU memory needs to be freed, and be able to re-create them
  //       afterwards from the CPU data if they are needed again.
  depth_frame_.reset(new ImageFrame<u16, SE3f>());
  color_frame_.reset(new ImageFrame<Vec3u8, SE3f>());
  
  // Make sure that any derived transformations are cached
  set_global_T_frame(global_tr_frame);
  
  activation_ = Activation::kActive;
}

}
