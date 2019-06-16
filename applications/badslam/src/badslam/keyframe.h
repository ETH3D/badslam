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

#include <libvis/cuda/cuda_buffer.h>
#include <libvis/cuda/cuda_util.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>

#include "badslam/cuda_depth_processing.cuh"
#include "badslam/cuda_depth_processing.h"
#include "badslam/cuda_image_processing.cuh"
#include "badslam/cuda_matrix.cuh"

namespace vis {

// Represents a keyframe which stores the measured (preprocessed) depth image,
// as well as a normal and radius image derived from it. Furthermore, the
// measured color image is stored, and the current estimate for the keyframe's
// global pose.
class Keyframe {
 public:
  // Keyframe activation state, not used by default and not described in the
  // paper.
  enum class Activation {
    // A keyframe is active if it moved more than some epsilon in the last BA
    // iteration.
    kActive = 0,
    
    // A keyframe has the state "covisible active" if it did not move more than
    // some epsilon in the last BA iteration, but one of the keyframes with
    // which it has co-visibility did.
    kCovisibleActive = 1,
    
    // A keyframe is inactive if neither itself nor any keyframe with which it
    // has co-visibility moved in the last BA iteration.
    kInactive = 2
  };
  
  
  // Creates a keyframe using existing GPU buffers of depth, normal, radius,
  // and color data, as well as existing depth and color frames which may
  // reference the original depth and color data on the CPU respectively disk.
  Keyframe(
      cudaStream_t stream,
      u32 frame_index,
      float min_depth,
      float max_depth,
      const CUDABuffer<u16>& depth_buffer,
      const CUDABuffer<u16>& normals_buffer,
      const CUDABuffer<u16>& radius_buffer,
      const CUDABuffer<uchar4>& color_buffer,
      const ImageFramePtr<u16, SE3f>& depth_frame,
      const ImageFramePtr<Vec3u8, SE3f>& color_frame);
  
  // Creates a keyframe from depth and color data. Derives the normal and radius
  // data from the depth image. This function is slow (since it involves
  // temporary GPU memory allocations) and thus should not be used for
  // real-time functionality, but is provided for convenience (e.g., for testing).
  // raw_to_float_depth is the factor which is applied to values of depth_image
  // to obtain metric depths in meters. Notice that this constructor does not
  // apply the full preprocessing pipeline to the depth images that BAD SLAM
  // would apply (e.g., it does not apply bilateral filtering to the depth).
  Keyframe(
      cudaStream_t stream,
      u32 frame_index,
      const DepthParameters& depth_params,
      const PinholeCamera4f& depth_camera,
      const Image<u16>& depth_image,
      const Image<Vec3u8>& color_image,
      const SE3f& global_tr_frame);
  
  inline ~Keyframe() {
    cudaDestroyTextureObject(color_texture_);
  }
  
  inline void SetID(int id) {
    id_ = id;
  }
  
  inline int id() const {
    return id_;
  }
  
  inline int last_active_in_ba_iteration() const {
    return last_active_in_ba_iteration_;
  }
  
  inline void SetLastActiveInBAIteration(int iteration) {
    last_active_in_ba_iteration_ = iteration;
  }
  
  inline int last_covis_in_ba_iteration() const {
    return last_covis_in_ba_iteration_;
  }
  
  inline void SetLastCovisInBAIteration(int iteration) {
    last_covis_in_ba_iteration_ = iteration;
  }
  
  inline float min_depth() const {
    return min_depth_;
  }
  
  inline float max_depth() const {
    return max_depth_;
  }
  
  inline vector<int>& co_visibility_list() {
    return co_visibility_list_;
  }
  
  inline const vector<int>& co_visibility_list() const {
    return co_visibility_list_;
  }
  
  inline Activation activation() const {
    return activation_;
  }
  
  void SetActivation(Activation activation) {
    activation_ = activation;
  }
  
  // Returns the video frame index of the frame from which this keyframe was
  // created.
  inline u32 frame_index() const {
    return frame_index_;
  }
  
  inline void set_global_T_frame(const SE3f& global_T_frame) {
    depth_frame_->SetGlobalTFrame(global_T_frame);
    color_frame_->SetGlobalTFrame(global_T_frame);
    frame_T_global_cuda_ = CUDAMatrix3x4(frame_T_global().matrix3x4());
    global_R_frame_cuda_ = CUDAMatrix3x3(global_T_frame.rotationMatrix());
  }
  
  inline void set_frame_T_global(const SE3f& frame_T_global) {
    depth_frame_->SetFrameTGlobal(frame_T_global);
    color_frame_->SetFrameTGlobal(frame_T_global);
    frame_T_global_cuda_ = CUDAMatrix3x4(frame_T_global.matrix3x4());
    global_R_frame_cuda_ = CUDAMatrix3x3(global_T_frame().rotationMatrix());
  }
  
  inline const SE3f& global_T_frame() const {
    return depth_frame_->global_T_frame();
  }
  
  inline const SE3f& frame_T_global() const {
    return depth_frame_->frame_T_global();
  }
  
  inline const CUDAMatrix3x4& frame_T_global_cuda() const {
    return frame_T_global_cuda_;
  }
  
  inline const CUDAMatrix3x3& global_R_frame_cuda() const {
    return global_R_frame_cuda_;
  }
  
  inline const CUDABuffer<u16>& depth_buffer() const {
    return depth_buffer_;
  }
  
  inline const CUDABuffer<u16>& normals_buffer() const {
    return normals_buffer_;
  }
  
  inline const CUDABuffer<u16>& radius_buffer() const {
    return radius_buffer_;
  }
  
  inline const CUDABuffer<uchar4>& color_buffer() const {
    return color_buffer_;
  }
  
  inline cudaTextureObject_t color_texture() const {
    return color_texture_;
  }
  
 private:
  int id_;
  u32 frame_index_;
  int last_active_in_ba_iteration_;
  int last_covis_in_ba_iteration_;
  
  float min_depth_;
  float max_depth_;
  
  CUDAMatrix3x4 frame_T_global_cuda_;
  CUDAMatrix3x3 global_R_frame_cuda_;
  
  // List of frames having co-visibility with this frame.
  vector<int> co_visibility_list_;
  
  Activation activation_;
  
  CUDABuffer<u16> depth_buffer_;
  CUDABuffer<u16> normals_buffer_;  // (more or less) derived from the depth. TODO: Re-compute this from the depth buffer, if required, to save memory?
  CUDABuffer<u16> radius_buffer_;  // (more or less) derived from the depth. TODO: Re-compute this from the depth buffer, if required, to save memory?
  CUDABuffer<uchar4> color_buffer_;
  cudaTextureObject_t color_texture_;
  
  // Reference to depth data on the CPU / disk
  ImageFramePtr<u16, SE3f> depth_frame_;
  // Reference to color data on the CPU / disk
  ImageFramePtr<Vec3u8, SE3f> color_frame_;
};

}
