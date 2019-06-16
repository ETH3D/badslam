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

#include "badslam/cuda_depth_processing.cuh"
#include "badslam/direct_ba.h"
#include "badslam/kernels.h"
#include "badslam/util.cuh"
#include "badslam/render_window.h"
#include "badslam/pose_graph_optimizer.h"

using namespace vis;


// Tests that the pose graph optimizer does not crash (i.e., the program is able
// to invoke g2o properly).
TEST(Optimization, PoseGraphOptimizer) {
  srand(0);
  
  // Initialize camera
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.5f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
  PinholeCamera4f camera(width, height, camera_parameters);
  
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
      /*use_depth_residuals*/ true,
      /*use_descriptor_residuals*/ false,
      nullptr,
      /*global_T_anchor_frame*/ SE3f());
  
  // Create dummy image
  Image<Vec3u8> color_image(width, height);
  memset(color_image.data(), 0, color_image.stride() * color_image.height());
  
  Image<u16> depth_image(width, height);
  depth_image.SetTo(numeric_limits<u16>::max());
  
  // Add some dummy keyframes
  for (int i = 0; i < 10; ++ i) {
    SE3f global_tr_frame = SE3f::exp(SE3f::Tangent::Random());
    shared_ptr<Keyframe> new_keyframe(new Keyframe(
        /*stream*/ 0,
        /*frame_index*/ i,
        direct_ba.depth_params(),
        direct_ba.depth_camera(),
        depth_image,
        color_image,
        global_tr_frame));
    direct_ba.AddKeyframe(new_keyframe);
  }
  
  // Initialize PoseGraphOptimizer
  PoseGraphOptimizer optimizer(&direct_ba, /*add_current_state_odometry_constraints*/ true);
  
  // Optimize the graph
  optimizer.Optimize();
}
