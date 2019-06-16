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


#include <complex>
#include <iostream>

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


// Helper function to render a set of planes into a depth and normal image.
void RenderPlanes(
    const SE3f& global_tr_frame,
    int plane_count,
    Hyperplane<float, 3>* planes,
    float raw_to_float_depth,
    const PinholeCamera4f& camera,
    Image<u16>* depth_image,
    Image<Vec3u8>* color_image) {
  const Vec3f& global_t_frame = global_tr_frame.translation();
  const Mat3f global_r_frame = global_tr_frame.rotationMatrix();
  
  float invalid_float_depth = raw_to_float_depth * numeric_limits<u16>::max();
  depth_image->SetTo(numeric_limits<u16>::max());
  color_image->SetTo(Vec3u8(0, 0, 0));
  
  for (int plane_index = 0; plane_index < plane_count; ++ plane_index) {
    for (int y = 0; y < depth_image->height(); ++ y) {
      for (int x = 0; x < depth_image->width(); ++ x) {
        Vec3f frame_direction = camera.UnprojectFromPixelCenterConv(Vec2f(x, y));
        ASSERT_EQ(frame_direction.z(), 1.0f);
        ParametrizedLine<float, 3> ray(global_t_frame, global_r_frame * frame_direction);
        
        float z = ray.intersectionParameter(planes[plane_index]);
        float previous_measurement = raw_to_float_depth * (*depth_image)(x, y);
        if (z > 0 && (previous_measurement == invalid_float_depth || z < previous_measurement)) {
          (*depth_image)(x, y) = std::min<u32>(numeric_limits<u16>::max(), z / raw_to_float_depth + 0.5f);
          
          Vec3f global_position = ray.pointAt(z);
          constexpr float kFactor = 200;
          u8 cx = (255 / 2.f) * (1.f + sin(0.15f * kFactor * global_position.x() + 0.5f * sin(0.25f * kFactor * global_position.y()))) + rand() % 1;
          u8 cy = (255 / 2.f) * (1.f + sin(0.15f * kFactor * global_position.y() + 0.5f * sin(0.25f * kFactor * global_position.z()))) + rand() % 1;
          u8 cz = (255 / 2.f) * (1.f + sin(0.15f * kFactor * global_position.z() + 0.5f * sin(0.25f * kFactor * global_position.x()))) + rand() % 1;
          (*color_image)(x, y) = Vec3u8(cx, cy, cz);
        }
      }
    }
  }
  
  // Uncomment to debug:
//   static ImageDisplay depth_debug_display;
//   depth_debug_display.Update(*depth_image, "debug: depth image", static_cast<u16>(0.f), static_cast<u16>(3.0f / raw_to_float_depth));
//   static ImageDisplay color_debug_display;
//   color_debug_display.Update(*color_image, "debug: color image");
//   std::getchar();
}

// Intrinsics optimization test. Creates a scene consisting of
// a set of planes, then renders some keyframes from this geometry and distorts
// their color intrinsics. Applies bundle adjustment and tests whether this
// distortion can be compensated.
// 
// It should be straightforward to use this test code as a template to apply
// direct bundle adjustment on a set of images. Simply load the depth and color
// images from files instead of generating them, and supply initial estimates
// for the keyframe poses.
static void TestIntrinsicsOptimizationWithPhotometricResidual(bool use_pcg) {
  srand(0);
  
  // Initialize the camera (here we use the same camera intrinsics for the depth
  // and color camera).
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.45f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
  PinholeCamera4f camera(width, height, camera_parameters);
  
  // Initialize a CUDA stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Optional: Create a render window for debugging.
  constexpr bool show_visualization = false;
  
  shared_ptr<BadSlamRenderWindow> render_window;  // render window callbacks
  shared_ptr<RenderWindow> render_window_ui;  // render window UI object
  OpenGLContext opengl_context;  // context for the main thread
  OpenGLContext opengl_context_2;  // context for the BA thread
  OpenGLContext no_opengl_context;  // stores the original (non-existing) context
  
  if (show_visualization) {
    float splat_half_extent_in_pixels = 3.0f;
    int render_window_default_width = 1280;
    int render_window_default_height = 720;
    
    // Warning: do not use make_shared() here to allocate render_window, since
    // it might ignore the class' aligned allocator, leading to potential
    // crashes in Eigen operations later on render_window's members.
    render_window = shared_ptr<BadSlamRenderWindow>(
        new BadSlamRenderWindow(splat_half_extent_in_pixels));
    render_window_ui = RenderWindow::CreateWindow(
        "BAD SLAM Test", render_window_default_width, render_window_default_height,
        RenderWindow::API::kOpenGL, render_window);
    render_window->InitializeForCUDAInterop(
        /*max_surfel_count*/ 1000 * 1000,
        &opengl_context,
        &opengl_context_2,
        camera);
    SwitchOpenGLContext(opengl_context, &no_opengl_context);
  }
  
  // Distort color intrinsics
  const float distorted_camera_parameters[4] = {0.5f * height + 0.5f, 0.45f * height - 0.6f, 0.5f * width - 0.5f + 1.23f, 0.5f * height - 0.5f - 2.17f};
  PinholeCamera4f distorted_color_camera(width, height, distorted_camera_parameters);
  
  // Initialize DirectBA
  constexpr float raw_to_float_depth = 1.f / 1000;
  DirectBA direct_ba(
      /*max_surfel_count*/ 1000 * 1000,
      raw_to_float_depth,
      /*baseline_fx*/ 40,
      /*sparse_surfel_cell_size*/ 2.f,
      /*surfel_merge_dist_factor*/ 0.8f,
      /*min_observation_count_while_bootstrapping_1*/ 2,
      /*min_observation_count_while_bootstrapping_2*/ 2,
      /*min_observation_count*/ 2,
      /*color_camera_initial_estimate*/ camera,
      /*depth_camera_initial_estimate*/ camera,
      /*pyramid_level_for_color*/ 0,
      /*use_depth_residuals*/ false,  // NOTE: disabling geometric residuals!
      /*use_descriptor_residuals*/ true,
      render_window,
      /*global_T_anchor_frame*/ SE3f());
  
  if (show_visualization) {
    render_window->SetDirectBA(&direct_ba);
  }
  
  // Set the first keyframe's pose
  Matrix<float, 6, 1> global_tr_frame_log;
  global_tr_frame_log << 0.01f, 0.02f, 0.03f, 0.004f, 0.005f, 0.006f;
  SE3f global_tr_frame_0 = SE3f::exp(global_tr_frame_log);
  
  // Generate test geometry consisting of some random planes
  constexpr int kPlaneCount = 20;
  Hyperplane<float, 3> planes[kPlaneCount];
  
  for (int plane_index = 0; plane_index < kPlaneCount; ++ plane_index) {
    Vec3f plane_normal = Vec3f::Random();
    plane_normal.z() = -1.f;
    plane_normal.normalize();
    planes[plane_index] = Hyperplane<float, 3>(plane_normal, 2.5f);
  }
  
  // Allocate buffers for rendering the planes into
  Image<u16> depth_image(width, height);
  Image<Vec3u8> color_image(width, height);
  
  // Create keyframes observing the surface from different distances
  constexpr int kNumKeyframes = 12;
  RGBDVideo<Vec3u8, u16> rgbd_video;
  
  SE3f last_frame_tr_global = SE3f();
  for (int i = 0; i < kNumKeyframes; ++ i) {
    Matrix<float, 6, 1> frame_0_T_frame_log;
    frame_0_T_frame_log <<
        3.0f * ((rand() % 200) / 200.f - 0.5f),
        3.0f * ((rand() % 200) / 200.f - 0.5f),
        3.0f * ((rand() % 200) / 200.f - 0.5f),
        3.5f * (((rand() % 200) - 100) / 500.f),
        3.5f * (((rand() % 200) - 100) / 500.f),
        3.5f * (((rand() % 200) - 100) / 500.f);
    SE3f frame_0_T_frame = SE3f::exp(frame_0_T_frame_log);
    SE3f global_tr_frame = global_tr_frame_0 * frame_0_T_frame;
    
    RenderPlanes(global_tr_frame, kPlaneCount, planes,
                 raw_to_float_depth, camera, &depth_image, &color_image);
    
    shared_ptr<Keyframe> keyframe(new Keyframe(
        stream,
        /*frame_index*/ i,
        direct_ba.depth_params(),
        direct_ba.depth_camera(),
        depth_image,
        color_image,
        global_tr_frame));
    direct_ba.AddKeyframe(keyframe);
    last_frame_tr_global = keyframe->frame_T_global();
  }
  
  for (auto& keyframe : direct_ba.keyframes()) {
    direct_ba.CreateSurfelsForKeyframe(stream, true, keyframe);
  }
  direct_ba.SetColorCamera(distorted_color_camera);
  
  // Optimize intrinsics
  for (int i = 0; i < 10; ++ i) {
    direct_ba.BundleAdjustment(
        stream,
        /*optimize_depth_intrinsics*/ false,
        /*optimize_color_intrinsics*/ true,
        /*do_surfel_updates*/ true,
        /*optimize_poses*/ false,
        /*optimize_geometry*/ false,
        /*min_iterations*/ 1,
        /*max_iterations*/ 10,
        use_pcg,
        /*active_keyframe_window_start*/ 0,
        /*active_keyframe_window_end*/ direct_ba.keyframes().size() - 1,
        /*increase_ba_iteration_count*/ i != 0);
    
    if (show_visualization) {
      std::getchar();
    }
    
    // Uncomment this to print the current state at each iteration:
    PinholeCamera4f estimated_camera = direct_ba.color_camera();
    LOG(INFO) << "camera_difference: " << (estimated_camera.parameters()[0] - camera.parameters()[0])
                                      << ", " << (estimated_camera.parameters()[1] - camera.parameters()[1])
                                      << ", " << (estimated_camera.parameters()[2] - camera.parameters()[2])
                                      << ", " << (estimated_camera.parameters()[3] - camera.parameters()[3]);
  }
  
  PinholeCamera4f estimated_camera = direct_ba.color_camera();
  EXPECT_NEAR(camera.parameters()[0], estimated_camera.parameters()[0], 0.03);
  EXPECT_NEAR(camera.parameters()[1], estimated_camera.parameters()[1], 0.03);
  EXPECT_NEAR(camera.parameters()[2], estimated_camera.parameters()[2], 0.15);
  EXPECT_NEAR(camera.parameters()[3], estimated_camera.parameters()[3], 0.15);
  
  if (show_visualization) {
    SwitchOpenGLContext(no_opengl_context);
    opengl_context.Deinitialize();
    opengl_context_2.Deinitialize();
  }
  
  cudaStreamDestroy(stream);
}

TEST(Optimization, AlternatingIntrinsicsOptimizationWithPhotometricResidual) {
  TestIntrinsicsOptimizationWithPhotometricResidual(/*use_pcg*/ false);
}

TEST(Optimization, PCGIntrinsicsOptimizationWithPhotometricResidual) {
  TestIntrinsicsOptimizationWithPhotometricResidual(/*use_pcg*/ true);
}
