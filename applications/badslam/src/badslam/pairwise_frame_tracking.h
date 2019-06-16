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
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>

#include "badslam/kernels.h"
#include "badslam/surfel_projection.cuh"

namespace vis {

class BadSlamRenderWindow;

// Collection of GPU buffers used by pairwise frame tracking that should stay
// allocated over subsequent calls to avoid repeated GPU memory allocations.
struct PairwiseFrameTrackingBuffers {
  PairwiseFrameTrackingBuffers(int depth_width, int depth_height, int num_scales);
  
  vector<CUDABufferPtr<float>> tracked_depth;
  vector<CUDABufferPtr<u16>> tracked_normals;
  vector<CUDABufferPtr<uchar>> tracked_color;
  vector<cudaTextureObject_t> tracked_color_texture;
  
  vector<CUDABufferPtr<float>> base_depth;
  vector<CUDABufferPtr<u16>> base_normals;
  vector<CUDABufferPtr<uchar>> base_color;
  vector<cudaTextureObject_t> base_color_texture;
};

// Prepares GPU buffers and textures for pairwise frame tracking, for the given
// image sizes.
void CreatePairwiseTrackingInputBuffersAndTextures(
    int depth_width,
    int depth_height,
    int color_width,
    int color_height,
    CUDABufferPtr<float>* calibrated_depth,
    CUDABufferPtr<uchar>* calibrated_gradmag,
    CUDABufferPtr<uchar>* base_kf_gradmag,
    CUDABufferPtr<uchar>* tracked_gradmag,
    cudaTextureObject_t* calibrated_gradmag_texture,
    cudaTextureObject_t* base_kf_gradmag_texture,
    cudaTextureObject_t* tracked_gradmag_texture);

// Tracks the pose of an RGB-D frame relative to another RGB-D frame.
// TODO: If possible, simplify the function signature?
//       Maybe create a "PairwiseFrameTracker" class which contains the helper buffers?
void TrackFramePairwise(
    PairwiseFrameTrackingBuffers* buffers,
    cudaStream_t stream,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const CUDABuffer<float>& cfactor_buffer,
    PoseEstimationHelperBuffers* helper_buffers,
    const shared_ptr<BadSlamRenderWindow>& render_window,
    std::ofstream* convergence_samples_file,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    bool use_pyramid_level_0,
    bool use_gradmag,
    /* tracked frame */
    const CUDABuffer<u16>& tracked_depth_buffer,
    const CUDABuffer<u16>& tracked_normals_buffer,
    const cudaTextureObject_t tracked_color_texture,
    /* base frame */
    const CUDABuffer<float>& base_depth_buffer,
    const CUDABuffer<u16>& base_normals_buffer,
    const CUDABuffer<uchar>& base_color_buffer,
    const cudaTextureObject_t base_color_texture,
    /* input / output poses */
    const SE3f& global_T_base,  // for debugging only!
    bool test_different_initial_estimates,
    const SE3f& base_T_frame_initial_estimate_1,
    const SE3f& base_T_frame_initial_estimate_2,
    SE3f* out_base_T_frame_estimate);

}
