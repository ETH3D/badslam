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


// This checks that the surfel depth optimization in bundle adjustment works
// correctly. It sets up a keyframe and a scene with surfels that match its
// depth image. The it perturbs the keyframe's depth map and tests whether
// optimizing the surfel positions moves them accordingly.
static void TestGeometryOptimizationWithGeometricResidual(bool use_pcg) {
  srand(0);
  
  // Initialize camera
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.5f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
  PinholeCamera4f camera(width, height, camera_parameters);
  
  // Set keyframe pose
  Matrix<float, 6, 1> global_tr_frame_log;
  global_tr_frame_log << 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f;
  SE3f global_tr_frame = SE3f::exp(global_tr_frame_log);
  
  // Initialize CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  // Initialize DirectBA
  constexpr float raw_to_float_depth = 1.f / 5000;
  DirectBA direct_ba(
      /*max_surfel_count*/ 1000 * 1000,
      raw_to_float_depth,
      /*baseline_fx*/ 40,
      /*sparse_surfel_cell_size*/ 1.f,
      /*surfel_merge_dist_factor*/ 0.8f,
      /*min_observation_count_while_bootstrapping_1*/ 1,
      /*min_observation_count_while_bootstrapping_2*/ 1,
      /*min_observation_count*/ 1,
      /*color_camera_initial_estimate*/ camera,
      /*depth_camera_initial_estimate*/ camera,
      /*pyramid_level_for_color*/ 0,
      /*use_depth_residuals*/ true,
      /*use_descriptor_residuals*/ true,
      nullptr,
      /*global_T_anchor_frame*/ SE3f());
  
  // Create a depth image.
  Image<u16> depth_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        depth_image(x, y) = numeric_limits<u16>::max();
      } else {
        depth_image(x, y) = (1 + 0.01f * (rand() % 100)) / raw_to_float_depth + 0.5f;
      }
    }
  }
  
  // Create a homogeneous color image.
  Image<Vec3u8> color_image(width, height);
  memset(color_image.data(), 0, color_image.stride() * color_image.height());
  
  shared_ptr<Keyframe> new_keyframe(new Keyframe(
      stream,
      /*frame_index*/ 0,
      direct_ba.depth_params(),
      direct_ba.depth_camera(),
      depth_image,
      color_image,
      global_tr_frame));
  
  // Force normals to surfel<->camera direction such that the optimization does
  // not change the pixel correspondence of surfels.
  Image<u16> normals_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      Vec3f normal = -1 * camera.UnprojectFromPixelCenterConv(Vec2f(x, y)).normalized();
      normals_image(x, y) = ImageSpaceNormalToU16(normal.x(), normal.y());
    }
  }
  CUDABuffer<u16>* kf_normals_buffer = const_cast<CUDABuffer<u16>*>(&new_keyframe->normals_buffer());
  kf_normals_buffer->UploadAsync(/*stream*/ 0, normals_image);
  
  direct_ba.AddKeyframe(new_keyframe);
  
  // Create surfels from the depth map.
  direct_ba.CreateSurfelsForKeyframe(stream, /*filter_new_surfels*/ false, new_keyframe);
  
  // Change the depth map and test whether the depth optimization changes the surfels accordingly.
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      depth_image(x, y) += (0.0001f * (rand() % 50)) / raw_to_float_depth;
    }
  }
  CUDABuffer<u16>* kf_depth_buffer = const_cast<CUDABuffer<u16>*>(&new_keyframe->depth_buffer());
  kf_depth_buffer->UploadAsync(stream, depth_image);
  
  constexpr int kIterationCount = 10;
  for (int i = 0; i < kIterationCount; ++ i) {
    direct_ba.BundleAdjustment(
        stream,
        /*optimize_depth_intrinsics*/ false,
        /*optimize_color_intrinsics*/ false,
        /*do_surfel_updates*/ false,
        /*optimize_poses*/ false,
        /*optimize_geometry*/ true,
        /*min_iterations*/ 10,
        /*max_iterations*/ 10,
        use_pcg,
        /*active_keyframe_window_start*/ 0,
        /*active_keyframe_window_end*/ direct_ba.keyframes().size() - 1,
        /*increase_ba_iteration_count*/ true);
  }
  
  vector<float> surfel_x(direct_ba.surfel_count());
  vector<float> surfel_y(direct_ba.surfel_count());
  vector<float> surfel_z(direct_ba.surfel_count());
  
  direct_ba.surfels()->DownloadPartAsync(kSurfelX * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_x.data());
  direct_ba.surfels()->DownloadPartAsync(kSurfelY * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_y.data());
  direct_ba.surfels()->DownloadPartAsync(kSurfelZ * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_z.data());
  
  cudaStreamSynchronize(stream);
  
  // NOTE: There is an error pattern visible in this visualization. I think
  //       it comes from limited numerical accuracy.
  constexpr bool kShowErrorVisualization = false;
  Image<Vec3u8> error_visualization(width, height);
  if (kShowErrorVisualization) {
    for (int y = 0; y < height; ++ y) {
      for (int x = 0; x < width; ++ x) {
        error_visualization(x, y) = Vec3u8(0, 0, 0);
      }
    }
  }
  
  SE3f frame_tr_global = global_tr_frame.inverse();
  Mat3f frame_r_global = frame_tr_global.rotationMatrix();
  const Vec3f& frame_t_global = frame_tr_global.translation();
  
  int num_fails = 0;
  for (usize i = 0; i < direct_ba.surfel_count(); ++ i) {
    Vec3f cam_space_surfel = frame_r_global * Vec3f(surfel_x[i], surfel_y[i], surfel_z[i]) + frame_t_global;
    Vec2f pxy;
    if (direct_ba.depth_camera().ProjectToPixelCornerConvIfVisible(cam_space_surfel, 0, &pxy)) {
      float expected_z = raw_to_float_depth * depth_image(pxy.x(), pxy.y());
      float actual_z = cam_space_surfel.z();
      float error = fabs(actual_z - expected_z);
      
      constexpr float kEpsilon = 1e-4f;
      if (error > kEpsilon) {
        ++ num_fails;
        if (kShowErrorVisualization) {
          error_visualization(pxy.x(), pxy.y()) = Vec3u8(255, 0, 0);
        }
      } else {
        if (kShowErrorVisualization) {
          u8 intensity = std::min<float>(255, (255.99f / kEpsilon) * error);
          error_visualization(pxy.x(), pxy.y()) = Vec3u8::Constant(intensity);
        }
      }
    }
  }
  
  EXPECT_EQ(0, num_fails);
  
  if (kShowErrorVisualization) {
    ImageDisplay debug_display;
    debug_display.Update(error_visualization, "Surfel depth error visualization");
    std::getchar();
  }
  
  cudaStreamDestroy(stream);
}

TEST(Optimization, AlternatingGeometryOptimizationWithGeometricResidual) {
  TestGeometryOptimizationWithGeometricResidual(/*use_pcg*/ false);
}

TEST(Optimization, PCGGeometryOptimizationWithGeometricResidual) {
  TestGeometryOptimizationWithGeometricResidual(/*use_pcg*/ true);
}
