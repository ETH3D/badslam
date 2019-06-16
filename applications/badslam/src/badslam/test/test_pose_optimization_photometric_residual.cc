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


#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <libvis/eigen.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>

#include "badslam/direct_ba.h"
#include "badslam/kernels.h"
#include "badslam/util.cuh"
#include "badslam/render_window.h"
#include "badslam/cuda_depth_processing.cuh"

using namespace vis;


// This checks that the photometric residual in bundle adjustment works correctly.
// It sets up a keyframe and a scene with surfels that match its depth image.
// Then it perturbs the keyframe's pose and tests whether optimizing the frame's
// pose yields its original pose again, using only photometric residuals.
TEST(Optimization, PoseOptimizationColorOnlyCues) {
  srand(0);
  
  // Initialize camera
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.5f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
  PinholeCamera4f camera(width, height, camera_parameters);
  
  // Set ground truth keyframe pose
  Matrix<float, 6, 1> global_tr_frame_log;
  global_tr_frame_log << 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f;
  SE3f global_tr_frame = SE3f::exp(global_tr_frame_log);
  
  // Initialize CUDA stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Initialize DirectBA
  constexpr float raw_to_float_depth = 1.f / 1000;
  DirectBA direct_ba(
      /*max_surfel_count*/ 1000 * 1000,
      raw_to_float_depth,
      /*baseline_fx*/ 40,
      /*sparse_surfel_cell_size*/ 1.f,
      /*surfel_merge_dist_factor*/ 0.8f,
      /*min_observation_count_while_bootstrapping_1*/ 2,
      /*min_observation_count_while_bootstrapping_2*/ 2,
      /*min_observation_count*/ 2,
      /*color_camera_initial_estimate*/ camera,
      /*depth_camera_initial_estimate*/ camera,
      /*pyramid_level_for_color*/ 0,
      /*use_depth_residuals*/ false,
      /*use_descriptor_residuals*/ true,
      nullptr,
      /*global_T_anchor_frame*/ SE3f());
  
  // Create a depth map that does not fully constrain the pose
  Image<u16> depth_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        depth_image(x, y) = numeric_limits<u16>::max();
      } else {
        depth_image(x, y) = 2 / raw_to_float_depth;
      }
    }
  }
  
  // Create a random color image
  Image<Vec3u8> color_image(width, height);
  for (int y = 0; y < height; ++ y) {
    color_image(0, y) = Vec3u8(0, 0, 0);
    for (int x = 1; x < width; ++ x) {
      u8 i = rand() % 16;
      // Enforce some local consistency by basing the color on the color of the
      // left and top pixels
      const Vec3u8& left = color_image(x - 1, y);
      Vec3u8 top;
      if (y > 0) {
        top = color_image(x, y - 1);
      } else {
        top = Vec3u8(0, 0, 0);
      }
      color_image(x, y) = Vec3u8(
          (left.x() + top.x()) / 2 + i,
          (left.y() + top.y()) / 2 + i,
          (left.z() + top.z()) / 2 + i);
    }
  }
  
//   ImageDisplay debug_display;
//   debug_display.Update(color_image, "debug: color");
//   std::getchar();
//   debug_display.Update(depth_image, "debug: depth map", 0.f, 2 / raw_to_float_depth);
//   std::getchar();
  
  shared_ptr<Keyframe> new_keyframe(new Keyframe(
      stream,
      /*frame_index*/ 0,
      direct_ba.depth_params(),
      direct_ba.depth_camera(),
      depth_image,
      color_image,
      global_tr_frame));
  direct_ba.AddKeyframe(new_keyframe);
  
  // Create surfels from the depth map.
  direct_ba.CreateSurfelsForKeyframe(stream, /*filter_new_surfels*/ false, new_keyframe);
  
  // Attempt to estimate the frame's pose, starting from different offsets.
  constexpr float kTranslationOffset = 0.0005f;
  constexpr float kRotationOffset = 0.001f;
  SE3f offsets[] = {
      // Identity
      SE3f(),
      // Positive offsets
      SE3f::exp((Matrix<float, 6, 1>() << kTranslationOffset, 0, 0, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, kTranslationOffset, 0, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, kTranslationOffset, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, kRotationOffset, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, 0, kRotationOffset, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, 0, 0, kRotationOffset).finished()),
      // Negative offsets
      SE3f::exp((Matrix<float, 6, 1>() << -kTranslationOffset, 0, 0, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, -kTranslationOffset, 0, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, -kTranslationOffset, 0, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, -kRotationOffset, 0, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, 0, -kRotationOffset, 0).finished()),
      SE3f::exp((Matrix<float, 6, 1>() << 0, 0, 0, 0, 0, -kRotationOffset).finished())};
  
  for (int offset_index = 0; offset_index < static_cast<int>(sizeof(offsets) / sizeof(offsets[0])); ++ offset_index) {
    SE3f global_tr_frame_estimate;
    direct_ba.EstimateFramePose(
        stream,
        global_tr_frame * offsets[offset_index],
        new_keyframe->depth_buffer(),
        new_keyframe->normals_buffer(),
        new_keyframe->color_texture(),
        &global_tr_frame_estimate,
        /*called_within_ba*/ false);
    
    Matrix<float, 6, 1> error = (global_tr_frame_estimate.inverse() * global_tr_frame).log();
    // LOG(INFO) << "Case " << offset_index << " error: " << error.transpose();
    for (int i = 0; i < 6; ++ i) {
      constexpr float kTolerance = 8e-5f;
      EXPECT_NEAR(0.f, error(i), kTolerance) << "Error in test case (offset) " << offset_index << ", dimension " << i;
    }
  }
  
  cudaStreamDestroy(stream);
}
