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


// LambertW function (and helper functions to compute it) from
// https://github.com/IstvanMezo/LambertW-function . It is required for this
// test to invert the depth distortion function.

// z * exp(z)
complex<double> zexpz(complex<double> z) {
  return z * exp(z);
}

// The derivative of z * exp(z) = exp(z) + z * exp(z)
complex<double> zexpz_d(complex<double> z) {
  return exp(z) + z * exp(z);
}

// The second derivative of z * exp(z) = 2. * exp(z) + z * exp(z)
complex<double> zexpz_dd(complex<double> z) {
  return 2. * exp(z) + z * exp(z);
}

// Determine the initial point for the root finding
complex<double> LambertWInitPoint(complex<double> z, int k) {
  const double pi{ 3.14159265358979323846 };
  const double e{ 2.71828182845904523536 };
  complex<double> I{0, 1};
  complex<double> two_pi_k_I{ 0., 2. * pi * k };
  complex<double> ip{ log(z) + two_pi_k_I - log(log(z) + two_pi_k_I) };// initial point coming from the general asymptotic approximation
  complex<double> p{ sqrt( 2. * (e * z + 1.) ) };// used when we are close to the branch cut around zero and when k=0,-1

  if (abs(z - (-exp(-1.))) <= 1.) { //we are close to the branch cut, the initial point must be chosen carefully
    if (k == 0) ip = -1. + p - 1./3. * pow(p, 2) + 11./72. * pow(p, 3);
    if (k == 1 && z.imag() < 0.) ip = -1. - p - 1./3. * pow(p, 2) - 11. / 72. * pow(p, 3);
    if (k == -1 && z.imag() > 0.) ip = -1. - p - 1./3. * pow(p, 2) - 11./72. * pow(p, 3);
  }

  if (k == 0 && abs(z - .5) <= .5) ip = (0.35173371 * (0.1237166 + 7.061302897 * z)) / (2. + 0.827184 * (1. + 2. * z));// (1,1) Pade approximant for W(0,a)

  if (k == -1 && abs(z - .5) <= .5) ip = -(((2.2591588985 +
    4.22096*I) * ((-14.073271 - 33.767687754*I) * z - (12.7127 -
    19.071643*I) * (1. + 2.*z))) / (2. - (17.23103 - 10.629721*I) * (1. + 2.*z)));// (1,1) Pade approximant for W(-1,a)

  return ip;
}

complex<double> LambertW(complex<double> z, int k = 0) {
  // For some particular z and k W(z,k) has simple value:
  if (z == 0.) return (k == 0) ? 0. : -INFINITY;
  if (z == -exp(-1.) && (k == 0 || k == -1)) return -1.;
  if (z == exp(1.) && k == 0) return 1.;

  // Halley method begins
  complex<double> w{ LambertWInitPoint(z, k) }, wprev{ LambertWInitPoint(z, k) }; // intermediate values in the Halley method
  const unsigned int maxiter = 30; // max number of iterations. This eliminates improbable infinite loops
  unsigned int iter = 0; // iteration counter
  double prec = 1.E-30; // difference threshold between the last two iteration results (or the iter number of iterations is taken)

  do {
    wprev = w;
    w -= 2.*((zexpz(w) - z) * zexpz_d(w)) /
      (2.*pow(zexpz_d(w),2) - (zexpz(w) - z)*zexpz_dd(w));
    iter++;
  } while ((abs(w - wprev) > prec) && iter < maxiter);
  return w;
}

// Helper function to render a set of planes into a depth and normal image.
void RenderPlanes(
    const SE3f& global_tr_frame,
    int plane_count,
    Hyperplane<float, 3>* planes,
    float true_a,
    float true_cfactor,
    float raw_to_float_depth,
    const PinholeCamera4f& camera,
    Image<float>* plane_depth_image,
    Image<u16>* depth_image) {
  const int width = plane_depth_image->width();
  const int height = plane_depth_image->height();
  
  const Vec3f& global_t_frame = global_tr_frame.translation();
  const Mat3f global_r_frame = global_tr_frame.rotationMatrix();
  
  plane_depth_image->SetTo(0.f);
  depth_image->SetTo(static_cast<u16>(numeric_limits<u16>::max()));
  
  for (int plane_index = 0; plane_index < plane_count; ++ plane_index) {
    for (int y = 0; y < height; ++ y) {
      for (int x = 0; x < width; ++ x) {
        Vec3f frame_direction = camera.UnprojectFromPixelCenterConv(Vec2f(x, y));
        ASSERT_EQ(frame_direction.z(), 1.0f);
        ParametrizedLine<float, 3> ray(global_t_frame, global_r_frame * frame_direction);
        float z = ray.intersectionParameter(planes[plane_index]);
        float previous_measurement = (*plane_depth_image)(x, y);
        if (z > 0 && (previous_measurement == 0 || z < previous_measurement)) {
          (*plane_depth_image)(x, y) = z;
        }
      }
    }
  }
  
  // Convert raw to distorted depth
  for (int y = 1; y < height - 1; ++ y) {
    for (int x = 1; x < width - 1; ++ x) {
      float true_depth = (*plane_depth_image)(x, y);
      if (true_depth == 0) {
        (*depth_image)(x, y) = numeric_limits<u16>::max();
      } else if (true_a == 0 && true_cfactor == 0) {
        (*depth_image)(x, y) = std::min<u32>(numeric_limits<u16>::max(), true_depth / raw_to_float_depth + 0.5f);
      } else {
        float distorted_depth = 1.0 / ((true_a + true_depth * LambertW(complex<double>(-true_a * true_cfactor * exp(- true_a / true_depth))).real()) / (true_a * true_depth));
        (*depth_image)(x, y) = std::min<u32>(numeric_limits<u16>::max(), distorted_depth / raw_to_float_depth + 0.5f);
      }
    }
  }
  
  // Uncomment to debug:
//   static ImageDisplay debug_display;
//   debug_display.Update(*depth_image, "debug: depth map", static_cast<u16>(0.f), static_cast<u16>(3.0f / raw_to_float_depth));
//   std::getchar();
}

// Intrinsics optimization test. Creates a scene consisting of
// a set of planes, then renders some keyframes from this geometry and distorts
// their depth images. Applies bundle adjustment and tests whether this
// distortion can be compensated.
// 
// It should be straightforward to use this test code as a template to apply
// direct bundle adjustment on a set of images. Simply load the depth and color
// images from files instead of generating them, and supply initial estimates
// for the keyframe poses.
static void TestDepthDeformationOptimizationWithGeometricResidual(bool use_pcg) {
  srand(0);
  
  // Initialize the camera (here we use the same camera intrinsics for the depth
  // and color camera).
  constexpr int width = 640;
  constexpr int height = 480;
  const float camera_parameters[4] = {0.5f * height, 0.5f * height, 0.5f * width - 0.5f, 0.5f * height - 0.5f};
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
      /*use_depth_residuals*/ true,
      /*use_descriptor_residuals*/ false,  // NOTE: disabling photometric residuals!
      render_window,
      /*global_T_anchor_frame*/ SE3f());
  
  if (show_visualization) {
    render_window->SetDirectBA(&direct_ba);
  }
  
  // Create a known depth map with a number of planes and known distortion
  constexpr float true_a = 0.03f;
  constexpr float true_cfactor = 0.005f;
  
  // Set initial values for the optimization
  direct_ba.a() = 0;
  direct_ba.cfactor_buffer()->Clear(0, stream);
  
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
  Image<float> plane_depth_image(width, height);
  Image<u16> depth_image(width, height);
  
  // Use a homogeneous color image (descriptor residuals are disabled)
  Image<Vec3u8> color_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      color_image(x, y) = Vec3u8::Zero();
    }
  }
  
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
    
    RenderPlanes(global_tr_frame, kPlaneCount, planes, true_a, true_cfactor,
                 raw_to_float_depth, camera, &plane_depth_image, &depth_image);
    
    shared_ptr<Keyframe> keyframe(new Keyframe(
        stream,
        /*frame_index*/ i,
        direct_ba.depth_params(),
        direct_ba.depth_camera(),
        depth_image,
        color_image,  // NOTE: Colors are not reprojected, but this does not matter since they are not used
        global_tr_frame));
    direct_ba.AddKeyframe(keyframe);
    last_frame_tr_global = keyframe->frame_T_global();
  }
  
  // The 'c' factor is tested on a random pixel
  constexpr int kCFactorTestX = 50;
  constexpr int kCFactorTestY = 50;
  Image<float> cfactor_image(direct_ba.cfactor_buffer()->width(),
                             direct_ba.cfactor_buffer()->height());
  ASSERT_LT(kCFactorTestX, cfactor_image.width());
  ASSERT_LT(kCFactorTestY, cfactor_image.height());
  
  // Optimize intrinsics
  for (int i = 0; i < (use_pcg ? 20 : 400); ++ i) {
    direct_ba.BundleAdjustment(
        stream,
        /*optimize_depth_intrinsics*/ i != 0,
        /*optimize_color_intrinsics*/ false,
        /*do_surfel_updates*/ true,
        /*optimize_poses*/ false,
        /*optimize_geometry*/ true,
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
//     LOG(INFO) << "estimated a: " << direct_ba.a() << ", true a: " << true_a;
//     direct_ba.cfactor_buffer()->DownloadAsync(0, &cfactor_image);
//     LOG(INFO) << "estimated cfactor: " << cfactor_image(kCFactorTestX, kCFactorTestY) << ", true cfactor: " << true_cfactor;
  }
  
  EXPECT_NEAR(true_a, direct_ba.a(), 1e-2f);  // NOTE: loose threshold since this seems to converge extremely slowly (and probably does not matter much)
  direct_ba.cfactor_buffer()->DownloadAsync(0, &cfactor_image);
  EXPECT_NEAR(true_cfactor, cfactor_image(kCFactorTestX, kCFactorTestY), 1e-3f);
  
  if (show_visualization) {
    SwitchOpenGLContext(no_opengl_context);
    opengl_context.Deinitialize();
    opengl_context_2.Deinitialize();
  }
  
  cudaStreamDestroy(stream);
}

TEST(Optimization, AlternatingDepthDeformationOptimizationWithGeometricResidual) {
  TestDepthDeformationOptimizationWithGeometricResidual(/*use_pcg*/ false);
}

TEST(Optimization, PCGDepthDeformationOptimizationWithGeometricResidual) {
  TestDepthDeformationOptimizationWithGeometricResidual(/*use_pcg*/ true);
}


// This test specifically tests for pinhole camera intrinsics (fx, fy, cx, cy)
// optimization instead of depth deformation optimization.
static void TestIntrinsicsOptimizationWithGeometricResidual(bool use_pcg) {
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
  
  // Distort depth intrinsics
  const float distorted_camera_parameters[4] = {0.5f * height + 0.5f, 0.45f * height - 0.6f, 0.5f * width - 0.5f + 1.23f, 0.5f * height - 0.5f - 2.17f};
  PinholeCamera4f distorted_depth_camera(width, height, distorted_camera_parameters);
  
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
      /*use_depth_residuals*/ true,
      /*use_descriptor_residuals*/ false,  // NOTE: disabling photometric residuals!
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
  Image<float> plane_depth_image(width, height);
  Image<u16> depth_image(width, height);
  
  // Use a homogeneous color image (descriptor residuals are disabled)
  Image<Vec3u8> color_image(width, height);
  for (int y = 0; y < height; ++ y) {
    for (int x = 0; x < width; ++ x) {
      color_image(x, y) = Vec3u8::Zero();
    }
  }
  
  // Create keyframes observing the surface from different distances
  constexpr int kNumKeyframes = 3 * 12;
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
    
    RenderPlanes(global_tr_frame, kPlaneCount, planes, /*true_a*/ 0, /*true_cfactor*/ 0,
                 raw_to_float_depth, camera, &plane_depth_image, &depth_image);
    
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
  direct_ba.SetDepthCamera(distorted_depth_camera);
  
  // Optimize intrinsics
  // NOTE: We are testing many more iterations than necessary here since this
  //       triggered some issues where the a parameter could become strongly
  //       negative and then numerical issues destroyed the reconstruction.
  for (int i = 0; i < 100; ++ i) {
    direct_ba.BundleAdjustment(
        stream,
        /*optimize_depth_intrinsics*/ true,
        /*optimize_color_intrinsics*/ false,
        /*do_surfel_updates*/ false,
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
    PinholeCamera4f estimated_camera = direct_ba.depth_camera();
    LOG(INFO) << "camera_difference: " << (estimated_camera.parameters()[0] - camera.parameters()[0])
                                      << ", " << (estimated_camera.parameters()[1] - camera.parameters()[1])
                                      << ", " << (estimated_camera.parameters()[2] - camera.parameters()[2])
                                      << ", " << (estimated_camera.parameters()[3] - camera.parameters()[3]);
  }
  
  PinholeCamera4f estimated_camera = direct_ba.depth_camera();
  EXPECT_NEAR(camera.parameters()[0], estimated_camera.parameters()[0], 0.001);
  EXPECT_NEAR(camera.parameters()[1], estimated_camera.parameters()[1], 0.001);
  EXPECT_NEAR(camera.parameters()[2], estimated_camera.parameters()[2], 0.001);
  EXPECT_NEAR(camera.parameters()[3], estimated_camera.parameters()[3], 0.001);
  
  if (show_visualization) {
    SwitchOpenGLContext(no_opengl_context);
    opengl_context.Deinitialize();
    opengl_context_2.Deinitialize();
  }
  
  cudaStreamDestroy(stream);
}

TEST(Optimization, AlternatingIntrinsicsOptimizationWithGeometricResidual) {
  TestIntrinsicsOptimizationWithGeometricResidual(/*use_pcg*/ false);
}

TEST(Optimization, PCGIntrinsicsOptimizationWithGeometricResidual) {
  TestIntrinsicsOptimizationWithGeometricResidual(/*use_pcg*/ true);
}
