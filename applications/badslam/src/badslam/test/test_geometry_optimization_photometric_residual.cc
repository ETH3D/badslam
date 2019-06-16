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


static void ComputeError(
    int width,
    int height,
    cudaStream_t stream,
    DirectBA& direct_ba,
    const SE3f& global_tr_frame_1,
    float expected_z) {
  vector<float> surfel_x(direct_ba.surfel_count());
  vector<float> surfel_y(direct_ba.surfel_count());
  vector<float> surfel_z(direct_ba.surfel_count());
  
  direct_ba.surfels()->DownloadPartAsync(kSurfelX * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_x.data());
  direct_ba.surfels()->DownloadPartAsync(kSurfelY * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_y.data());
  direct_ba.surfels()->DownloadPartAsync(kSurfelZ * direct_ba.surfels()->ToCUDA().pitch(), direct_ba.surfel_count() * sizeof(float), stream, surfel_z.data());
  
  cudaStreamSynchronize(stream);
  
  constexpr bool kShowErrorVisualization = false;
  Image<Vec3u8> error_visualization(width, height);
  if (kShowErrorVisualization) {
    for (int y = 0; y < height; ++ y) {
      for (int x = 0; x < width; ++ x) {
        error_visualization(x, y) = Vec3u8(0, 0, 255);
      }
    }
  }
  
  SE3f frame_tr_global = global_tr_frame_1.inverse();
  Mat3f frame_r_global = frame_tr_global.rotationMatrix();
  const Vec3f& frame_t_global = frame_tr_global.translation();
  
  int num_fails = 0;
  int num_correct = 0;
  for (usize i = 0; i < direct_ba.surfel_count(); ++ i) {
    Vec3f cam_space_surfel = frame_r_global * Vec3f(surfel_x[i], surfel_y[i], surfel_z[i]) + frame_t_global;
    Vec2f pxy;
    if (direct_ba.depth_camera().ProjectToPixelCornerConvIfVisible(cam_space_surfel, 0, &pxy)) {
      float actual_z = cam_space_surfel.z();
      float error = fabs(expected_z - actual_z);
      
      constexpr float kEpsilon = 1e-3f;
      if (error > kEpsilon) {
        ++ num_fails;
        if (kShowErrorVisualization) {
          error_visualization(pxy.x(), pxy.y()) = Vec3u8(255, 0, 0);
        }
      } else {
        ++ num_correct;
        if (kShowErrorVisualization) {
          u8 intensity = std::min<float>(255, (255.99f / kEpsilon) * error);
          error_visualization(pxy.x(), pxy.y()) = Vec3u8::Constant(intensity);
        }
      }
    }
  }
  
  EXPECT_GE(num_correct, 100000);
  EXPECT_LE(num_fails, 75000);
  if (num_fails > 0) {
    LOG(INFO) << "num_fails: " << num_fails << " (" << ((100.f * num_fails) / direct_ba.surfel_count()) << "% of surfels; surfel_count: " << direct_ba.surfel_count() << ")";
  }
  
  if (kShowErrorVisualization) {
    ImageDisplay debug_display;
    debug_display.Update(error_visualization, "Surfel depth error visualization");
    std::getchar();
  }
}


// This checks that the surfel depth optimization in bundle adjustment works
// correctly. It sets up a keyframe and a scene with surfels that match its
// depth image. The it perturbs the keyframe's image and tests whether
// optimizing the surfel positions moves them accordingly.
static void TestGeometryOptimizationWithPhotometricResidual(bool use_pcg) {
  srand(0);
  
  // Initialize camera
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.5f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
  PinholeCamera4f camera(width, height, camera_parameters);
  
  // Set keyframe pose
  Matrix<float, 6, 1> global_tr_frame_log;
  global_tr_frame_log << 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f;
  SE3f global_tr_frame_0 = SE3f::exp(global_tr_frame_log);
  
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
      /*use_depth_residuals*/ false,
      /*use_descriptor_residuals*/ true,
      nullptr,
      /*global_T_anchor_frame*/ SE3f());
  
  constexpr int kFrameOffsetPx = 100;
  constexpr float kDepth = 2.f;
  
  // Create a known depth map
  Image<u16> depth_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      if (x < kFrameOffsetPx || y == 0 || x == width - 1 || y == height - 1) {
        depth_image(x, y) = numeric_limits<u16>::max();
      } else {
        depth_image(x, y) = (kDepth / raw_to_float_depth) + 0.5f;
      }
    }
  }
  
  Image<Vec3u8> color_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      u8 i = (255 / 2.f) * (1.f + sin(0.15f * x + 0.5f * sin(0.25f * y))) + rand() % 1;
      color_image(x, y) = Vec3u8(i, i, i);
    }
  }
  
//   ImageDisplay debug_display;
//   debug_display.Update(color_image, "debug: color");
//   std::getchar();
  
//   ImageDisplay debug_display;
//   debug_display.Update(inv_depth_image, "debug: inv depth map", 0.f, 0.5f);
//   std::getchar();
  
  RGBDVideo<Vec3u8, u16> rgbd_video;
  shared_ptr<Keyframe> keyframe_0(new Keyframe(
      stream,
      /*frame_index*/ 0,
      direct_ba.depth_params(),
      direct_ba.depth_camera(),
      depth_image,
      color_image,
      global_tr_frame_0));
  
  // Force normals to fronto-parallel.
  Image<u16> normals_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      normals_image(x, y) = ImageSpaceNormalToU16(0, 0);
    }
  }
  CUDABuffer<u16>* kf_normals_buffer = const_cast<CUDABuffer<u16>*>(&keyframe_0->normals_buffer());
  kf_normals_buffer->UploadAsync(/*stream*/ 0, normals_image);
  
  direct_ba.AddKeyframe(keyframe_0);
  
  // Reproject the first keyframe into a second one
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      if (x == 0 || y == 0 || x >= width - kFrameOffsetPx || y == height - 1) {
        depth_image(x, y) = numeric_limits<u16>::max();
      } else {
        // NOTE: adding depth noise
        depth_image(x, y) = depth_image(x + kFrameOffsetPx, y) + 0.0001f * (rand() % 100) / raw_to_float_depth;
        color_image(x, y) = color_image(x + kFrameOffsetPx, y);
      }
    }
  }
  
  // kFrameOffsetPx * fx_inv / 1 == offset_x / kDepth
  float offset_x = kDepth * kFrameOffsetPx / camera.parameters()[0];
  SE3f frame_0_T_frame_1;
  frame_0_T_frame_1.translation().x() = offset_x;
  SE3f global_tr_frame_1 = global_tr_frame_0 * frame_0_T_frame_1;
  
  shared_ptr<Keyframe> keyframe_1(new Keyframe(
      stream,
      /*frame_index*/ 1,
      direct_ba.depth_params(),
      direct_ba.depth_camera(),
      depth_image,
      color_image,
      global_tr_frame_1));
  
  // Force normals to fronto-parallel.
  kf_normals_buffer = const_cast<CUDABuffer<u16>*>(&keyframe_1->normals_buffer());
  kf_normals_buffer->UploadAsync(/*stream*/ 0, normals_image);
  
  direct_ba.AddKeyframe(keyframe_1);
  
  // Create surfels from keyframe 1 (having noisy depth values)
  direct_ba.CreateSurfelsForKeyframe(stream, /*filter_new_surfels*/ false, keyframe_1);
  
//   // Remove depth noise in the keyframe depth map again
//   for (int y = 0; y < height; ++ y) {
//     for (int x = 0; x < width; ++ x) {
//       if (x == 0 || y == 0 || x >= width - kFrameOffsetPx || y == height - 1) {
//         depth_image(x, y) = numeric_limits<u16>::max();
//       } else {
//         depth_image(x, y) = (kDepth / raw_to_float_depth) + 0.5f;
//       }
//     }
//   }
//   const_cast<CUDABuffer<u16>&>(keyframe_1->depth_buffer()).UploadAsync(0, depth_image);
  
  // Optimize surfel positions
  direct_ba.BundleAdjustment(
      stream,
      /*optimize_depth_intrinsics*/ false,
      /*optimize_color_intrinsics*/ false,
      /*do_surfel_updates*/ false,
      /*optimize_poses*/ false,
      /*optimize_geometry*/ true,
      /*min_iterations*/ 60,
      /*max_iterations*/ 60,
      use_pcg,
      /*active_keyframe_window_start*/ 0,
      /*active_keyframe_window_end*/ direct_ba.keyframes().size() - 1,
      /*increase_ba_iteration_count*/ true);
  
  ComputeError(width, height, stream, direct_ba, global_tr_frame_1, kDepth);
  
  cudaStreamDestroy(stream);
}

TEST(Optimization, AlternatingGeometryOptimizationWithPhotometricResidual) {
  TestGeometryOptimizationWithPhotometricResidual(/*use_pcg*/ false);
}

TEST(Optimization, PCGGeometryOptimizationWithPhotometricResidual) {
  TestGeometryOptimizationWithPhotometricResidual(/*use_pcg*/ true);
}
