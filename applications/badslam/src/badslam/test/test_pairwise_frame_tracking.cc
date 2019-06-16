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

#include <random>

#include <Eigen/Geometry>
#include <gtest/gtest.h>
#include <libvis/eigen.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/logging.h>
#include <libvis/mesh.h>
#include <libvis/mesh_opengl.h>
#include <libvis/renderer.h>

#include "badslam/cuda_depth_processing.cuh"
#include "badslam/direct_ba.h"
#include "badslam/kernels.h"
#include "badslam/pairwise_frame_tracking.h"
#include "badslam/util.cuh"
#include "badslam/render_window.h"


// Some test results:
// 
// 22:15:20.153 test_pairwise_frame_tra:343   INFO| --- Testing with gradients x / y ---
// 22:15:20.253 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.5 --> average error norm (convergence): 1.23053 --> average error norm (accuracy): 0.0275803
// 22:15:20.366 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.25 --> average error norm (convergence): 0.543878 --> average error norm (accuracy): 0.000406534
// 22:15:20.484 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.1 --> average error norm (convergence): 0.260922 --> average error norm (accuracy): 0.000267356
// 22:15:20.592 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.08 --> average error norm (convergence): 0.167879 --> average error norm (accuracy): 0.000228615
// 22:15:20.696 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.07 --> average error norm (convergence): 0.0626805 --> average error norm (accuracy): 0.000187236
// 22:15:20.792 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.06 --> average error norm (convergence): 0.0174566 --> average error norm (accuracy): 0.000157755
// 22:15:20.881 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.05 --> average error norm (convergence): 0.000274997 --> average error norm (accuracy): 0.000209946
// 22:15:20.968 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.025 --> average error norm (convergence): 0.000221819 --> average error norm (accuracy): 0.000190435
// 22:15:21.053 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.01 --> average error norm (convergence): 0.000229535 --> average error norm (accuracy): 0.000222226
// 22:15:21.131 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.001 --> average error norm (convergence): 0.000232132 --> average error norm (accuracy): 0.00109521
// 
// 22:15:21.131 test_pairwise_frame_tra:343   INFO| --- Testing with brightness constancy assumption ---
// 22:15:21.197 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.5 --> average error norm (convergence): 1.74595 --> average error norm (accuracy): 0.00198717
// 22:15:21.275 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.25 --> average error norm (convergence): 1.85936 --> average error norm (accuracy): 0.000113487
// 22:15:21.354 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.1 --> average error norm (convergence): 0.279781 --> average error norm (accuracy): 4.22934e-05
// 22:15:21.435 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.08 --> average error norm (convergence): 0.0521283 --> average error norm (accuracy): 5.52821e-05
// 22:15:21.511 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.07 --> average error norm (convergence): 5.90658e-05 --> average error norm (accuracy): 3.67257e-05
// 22:15:21.583 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.06 --> average error norm (convergence): 5.90927e-05 --> average error norm (accuracy): 3.76441e-05
// 22:15:21.654 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.05 --> average error norm (convergence): 5.91582e-05 --> average error norm (accuracy): 5.6894e-05
// 22:15:21.721 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.025 --> average error norm (convergence): 5.93368e-05 --> average error norm (accuracy): 3.62716e-05
// 22:15:21.788 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.01 --> average error norm (convergence): 5.9312e-05 --> average error norm (accuracy): 5.09321e-05
// 22:15:21.852 test_pairwise_frame_tra:459   INFO| Distortion strenght: 0.001 --> average error norm (convergence): 5.93804e-05 --> average error norm (accuracy): 9.19018e-05
// 
// 22:15:21.131 test_pairwise_frame_tra:343   INFO| --- Testing with gradient magnitudes ---
// 22:28:18.486 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.5 --> average error norm (convergence): 1.57414 --> average error norm (accuracy): 0.512028
// 22:28:18.597 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.25 --> average error norm (convergence): 0.903301 --> average error norm (accuracy): 1.59576
// 22:28:18.699 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.1 --> average error norm (convergence): 0.15644 --> average error norm (accuracy): 0.00174811
// 22:28:18.800 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.08 --> average error norm (convergence): 0.0837946 --> average error norm (accuracy): 0.00135416
// 22:28:18.896 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.07 --> average error norm (convergence): 0.0398299 --> average error norm (accuracy): 0.00130194
// 22:28:18.985 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.06 --> average error norm (convergence): 0.0124953 --> average error norm (accuracy): 0.00104814
// 22:28:19.070 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.05 --> average error norm (convergence): 0.00111669 --> average error norm (accuracy): 0.000687964
// 22:28:19.152 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.025 --> average error norm (convergence): 0.00112156 --> average error norm (accuracy): 0.000537154
// 22:28:19.229 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.01 --> average error norm (convergence): 0.00112214 --> average error norm (accuracy): 0.000340628
// 22:28:19.302 test_pairwise_frame_tra:504   INFO| Distortion strenght: 0.001 --> average error norm (convergence): 0.00111539 --> average error norm (accuracy): 0.000932258


using namespace vis;

static void EstimateNormals(
    const PinholeCamera4f& camera,
    Image<float>* depth,
    Image<u16>* normals) {
  normals->SetSize(depth->size());
  
  for (int y = 0; y < depth->height(); ++ y) {
    for (int x = 0; x < depth->width(); ++ x) {
      if (x < 1 || y < 1 || x >= depth->width() - 1 || y >= depth->height() - 1) {
        (*depth)(x, y) = 0;
        continue;
      }
      
      float float_depth = (*depth)(x, y);
      if (float_depth <= 0 || std::isinf(float_depth) || std::isnan(float_depth)) {
        (*depth)(x, y) = 0;
        continue;
      }
      
      float right_depth = (*depth)(x + 1, y);
      if (right_depth <= 0 || std::isinf(right_depth) || std::isnan(right_depth)) {
        (*depth)(x, y) = 0;
        continue;
      }
      
      float down_depth = (*depth)(x, y + 1);
      if (down_depth <= 0 || std::isinf(down_depth) || std::isnan(down_depth)) {
        (*depth)(x, y) = 0;
        continue;
      }
      
      Vec3f center_point = float_depth * camera.UnprojectFromPixelCenterConv(Vec2i(x, y));
      Vec3f right_point = right_depth * camera.UnprojectFromPixelCenterConv(Vec2i(x + 1, y));
      Vec3f down_point = down_depth * camera.UnprojectFromPixelCenterConv(Vec2i(x, y + 1));
      
      Vec3f to_right = right_point - center_point;
      Vec3f to_down = down_point - center_point;
      
      Vec3f normal = to_down.cross(to_right).normalized();
      (*normals)(x, y) = ImageSpaceNormalToU16(normal.x(), normal.y());
    }
  }
}

static void RenderImages(
    const PinholeCamera4f& camera,
    float min_render_depth,
    float max_render_depth,
    const shared_ptr<Renderer>& renderer,
    const shared_ptr<Mesh3fC3u8OpenGL>& mesh_opengl,
    bool use_gradmag,
    SE3f images_T_global[2],
    float raw_to_float_depth,
    CUDABuffer<u16>* tracked_depth_buffer,
    CUDABuffer<u16>* tracked_normals_buffer,
    CUDABufferPtr<uchar> tracked_color_buffer,
    CUDABuffer<u16>* base_normals_buffer,
    CUDABufferPtr<float> base_depth_buffer,
    CUDABufferPtr<uchar> base_color_buffer) {
  const int camera_width = camera.width();
  const int camera_height = camera.height();
  
  Image<float> depth_images[2];
  Image<Vec3u8> color_images[2];
  
  for (int i = 0; i < 2; ++ i) {
    renderer->BeginRendering(images_T_global[i], camera, min_render_depth, max_render_depth);
    mesh_opengl->Render(&renderer->shader_program());
    renderer->EndRendering();
    
    depth_images[i].SetSize(camera_width, camera_height);
    renderer->DownloadDepthResult(camera_width, camera_height, depth_images[i].data());
    
    color_images[i].SetSize(camera_width, camera_height);
    renderer->DownloadColorResult(camera_width, camera_height, reinterpret_cast<u8*>(color_images[i].data()));
    
    // Debug: show images
//       static ImageDisplay depth_image_display;
//       depth_image_display.Update(depth_images[i], "Rendered depth image", min_render_depth, max_render_depth);
//       static ImageDisplay color_image_display;
//       color_image_display.Update(color_images[i], "Rendered color image");
//       std::getchar();
  }
  
  // images[0] == tracked
  // images[1] == base
  
  // Derive tracked_normals_buffer and remove depth where no normal can be computed
  Image<u16> normal_image_0;
  EstimateNormals(camera, &depth_images[0], &normal_image_0);
  tracked_normals_buffer->UploadAsync(/*stream*/ 0, normal_image_0);
  
  // Upload depth_images[0] to tracked_depth_buffer. Need to scale to u16.
  Image<u16> tracked_depth_cpu(camera_width, camera_height);
  for (int y = 0; y < camera_height; ++ y) {
    for (int x = 0; x < camera_width; ++ x) {
      float float_depth = depth_images[0](x, y);
      if (float_depth <= 0 || std::isinf(float_depth) || std::isnan(float_depth)) {
        tracked_depth_cpu(x, y) = 0;
      } else {
        tracked_depth_cpu(x, y) = static_cast<u16>((depth_images[0](x, y) / raw_to_float_depth) + 0.5f);
      }
    }
  }
  tracked_depth_buffer->UploadAsync(/*stream*/ 0, tracked_depth_cpu);
  

  
  // Upload color_images[0] to a buffer for which tracked_color_texture is a texture
  if (use_gradmag) {
    Image<Vec4u8> tracked_color_cpu(camera_width, camera_height);
    for (int y = 0; y < camera_height; ++ y) {
      for (int x = 0; x < camera_width; ++ x) {
        const Vec3u8& color = color_images[0](x, y);
        tracked_color_cpu(x, y) = Vec4u8(color.x(), color.y(), color.z(), 0.299f * color.x() + 0.587f * color.y() + 0.114f * color.z());
      }
    }
    CUDABuffer<uchar4> temp_buffer(camera_height, camera_width);
    temp_buffer.UploadAsync(/*stream*/ 0, reinterpret_cast<Image<uchar4>&>(tracked_color_cpu));
    cudaTextureObject_t temp_texture;
    temp_buffer.CreateTextureObject(
        cudaAddressModeClamp, cudaAddressModeClamp,
        cudaFilterModeLinear, cudaReadModeNormalizedFloat,
        false, &temp_texture);
    ComputeSobelGradientMagnitudeCUDA(
        /*stream*/ 0,
        temp_texture,
        &tracked_color_buffer->ToCUDA());
    cudaDestroyTextureObject(temp_texture);
  } else {
    Image<u8> tracked_color_cpu(camera_width, camera_height);
    for (int y = 0; y < camera_height; ++ y) {
      for (int x = 0; x < camera_width; ++ x) {
        const Vec3u8& color = color_images[0](x, y);
        tracked_color_cpu(x, y) = 0.299f * color.x() + 0.587f * color.y() + 0.114f * color.z();
      }
    }
    tracked_color_buffer->UploadAsync(/*stream*/ 0, tracked_color_cpu);
  }
  
  
  // Derive base_normals_buffer and remove depth where no normal can be computed
  Image<u16> normal_image_1;
  EstimateNormals(camera, &depth_images[1], &normal_image_1);
  base_normals_buffer->UploadAsync(/*stream*/ 0, normal_image_1);
  
  // Upload data of image 1 for the base frame
  for (int y = 0; y < camera_height; ++ y) {
    for (int x = 0; x < camera_width; ++ x) {
      float float_depth = depth_images[1](x, y);
      if (float_depth < 0 || std::isinf(float_depth) || std::isnan(float_depth)) {
        depth_images[1](x, y) = 0;
      }
    }
  }
  base_depth_buffer->UploadAsync(/*stream*/ 0, depth_images[1]);
  
  // Upload color_images[1] to a buffer for which base_color_texture is a texture
  if (use_gradmag) {
    Image<uchar4> tracked_color_cpu(camera_width, camera_height);
    for (int y = 0; y < camera_height; ++ y) {
      for (int x = 0; x < camera_width; ++ x) {
        const Vec3u8& color = color_images[1](x, y);
        tracked_color_cpu(x, y) = make_uchar4(color.x(), color.y(), color.z(), 0.299f * color.x() + 0.587f * color.y() + 0.114f * color.z());
      }
    }
    CUDABuffer<uchar4> temp_buffer(camera_height, camera_width);
    temp_buffer.UploadAsync(/*stream*/ 0, tracked_color_cpu);
    cudaTextureObject_t temp_texture;
    temp_buffer.CreateTextureObject(
        cudaAddressModeClamp, cudaAddressModeClamp,
        cudaFilterModeLinear, cudaReadModeNormalizedFloat,
        false, &temp_texture);
    ComputeSobelGradientMagnitudeCUDA(
        /*stream*/ 0,
        temp_texture,
        &base_color_buffer->ToCUDA());
    cudaDestroyTextureObject(temp_texture);
  } else {
    Image<u8> base_color_cpu(camera_width, camera_height);
    for (int y = 0; y < camera_height; ++ y) {
      for (int x = 0; x < camera_width; ++ x) {
        const Vec3u8& color = color_images[1](x, y);
        base_color_cpu(x, y) = 0.299f * color.x() + 0.587f * color.y() + 0.114f * color.z();
      }
    }
    base_color_buffer->UploadAsync(/*stream*/ 0, base_color_cpu);
  }
}

TEST(Optimization, PairwiseFrameTracking) {
  srand(0);
  
  std::mt19937 generator(/*seed*/ 0);
  
  // Initialize and switch to an OpenGL context.
  OpenGLContext opengl_context;
  OpenGLContext no_opengl_context;
  
  opengl_context.InitializeWindowless();
  SwitchOpenGLContext(opengl_context, &no_opengl_context);
  
  // Create a synthetic scene using a heightmap. First generate vertices.
  constexpr int kHeightmapVerticesX = 61;
  constexpr int kHeightmapVerticesY = 61;
  constexpr float kHeightmapWidth = 5.f;
  constexpr float kHeightmapHeight = 5.f;
  constexpr float kHeightmapZDistance = 1.f;
  // Low variation to avoid occlusions.
  constexpr float kHeightmapZVariation = 0.05f;
  
  std::uniform_real_distribution<> z_distribution(-kHeightmapZVariation,
                                                  kHeightmapZVariation);
  std::uniform_int_distribution<> color_distribution(0, 255);
  
  shared_ptr<PointCloud<Point3fC3u8>> mesh_vertex_cloud(new PointCloud<Point3fC3u8>());
  mesh_vertex_cloud->Resize(kHeightmapVerticesY * kHeightmapVerticesX);
  
  int index = 0;
  for (int y = 0; y < kHeightmapVerticesY; ++ y) {
    for (int x = 0; x < kHeightmapVerticesX; ++ x) {
      Vec3f position;
      position.x() = ((x / (1.f * kHeightmapVerticesX - 1.f)) - 0.5f) * kHeightmapWidth;
      position.y() = ((y / (1.f * kHeightmapVerticesY - 1.f)) - 0.5f) * kHeightmapHeight;
      position.z() = kHeightmapZDistance + z_distribution(generator);
      // Make surface without occlusions by pulling back the surface at the
      // borders.
      position.z() -= 6 * sqrt(pow((x / (1.f * kHeightmapVerticesX - 1.f)) - 0.5f, 2) +
                               pow((y / (1.f * kHeightmapVerticesY - 1.f)) - 0.5f, 2));
      
      Vec3u8 color;
      color.x() = color_distribution(generator);
      color.y() = color_distribution(generator);
      color.z() = color_distribution(generator);
      
      mesh_vertex_cloud->at(index) = Point3fC3u8(position, color);
      ++ index;
    }
  }
  
  // Allocate mesh and insert vertices.
  Mesh3fCu8 mesh;
  *mesh.vertices_mutable() = mesh_vertex_cloud;
  vector<Triangle<u32>>& triangles = *mesh.triangles_mutable();
  
  // Write faces into mesh.
  int num_faces = 2 * (kHeightmapVerticesX - 1) * (kHeightmapVerticesY - 1);
  triangles.reserve(num_faces);
  for (int y = 0; y < kHeightmapVerticesY - 1; ++ y) {
    for (int x = 0; x < kHeightmapVerticesX - 1; ++ x) {
      // Top left.
      Triangle<u32> face;
      face.index(0) = x + (y + 1) * kHeightmapVerticesX;
      face.index(1) = (x + 1) + y * kHeightmapVerticesX;
      face.index(2) = x + y * kHeightmapVerticesX;
      triangles.push_back(face);
      
      // Bottom right.
      face.index(0) = x + (y + 1) * kHeightmapVerticesX;
      face.index(1) = (x + 1) + (y + 1) * kHeightmapVerticesX;
      face.index(2) = (x + 1) + y * kHeightmapVerticesX;
      triangles.push_back(face);
    }
  }
  
  // Transfer mesh to GPU memory.
  shared_ptr<Mesh3fC3u8OpenGL> mesh_opengl(new Mesh3fC3u8OpenGL(mesh));
  
  // Create camera.
  constexpr int kCameraWidth = 256;
  constexpr int kCameraHeight = 256;
  constexpr float kCameraFX = 0.5f * kCameraWidth;
  constexpr float kCameraFY = 0.5f * kCameraHeight;
  constexpr float kCameraCX = 0.5f * kCameraWidth - 0.5f;
  constexpr float kCameraCY = 0.5f * kCameraHeight - 0.5f;
  const float camera_parameters[4] = {kCameraFX, kCameraFY, kCameraCX, kCameraCY};
  PinholeCamera4f camera(kCameraWidth, kCameraHeight, camera_parameters);
  
  // Create RGB & depth renderer.
  RendererProgramStoragePtr renderer_program_storage(new RendererProgramStorage());
  shared_ptr<Renderer> renderer(new Renderer(
      /*render_color*/ true,
      /*render_depth*/ true,
      kCameraWidth, kCameraHeight,
      renderer_program_storage));
  
  // Initialize pairwise tracking.
  PairwiseFrameTrackingBuffers pairwise_tracking_buffers(kCameraWidth, kCameraHeight, /*num_scales*/ 3);
  PoseEstimationHelperBuffers pose_estimation_helper_buffers;
  
  int sparse_surfel_cell_size = 4;
  
  CUDABufferPtr<float> cfactor_buffer;
  cfactor_buffer.reset(new CUDABuffer<float>(
      (camera.height() - 1) / sparse_surfel_cell_size + 1,
      (camera.width() - 1) / sparse_surfel_cell_size + 1));
  cfactor_buffer->Clear(0, /*stream*/ 0);
  cudaDeviceSynchronize();
  
  DepthParameters depth_params;
  depth_params.a = 0;
  depth_params.cfactor_buffer = cfactor_buffer->ToCUDA();
  depth_params.raw_to_float_depth = 1. / 1000.;
  depth_params.baseline_fx = 40.f;
  depth_params.sparse_surfel_cell_size = sparse_surfel_cell_size;
  
  CUDABuffer<u16> tracked_depth_buffer(kCameraHeight, kCameraWidth);
  CUDABuffer<u16> tracked_normals_buffer(kCameraHeight, kCameraWidth);
  CUDABufferPtr<uchar> tracked_color_buffer;
  cudaTextureObject_t tracked_color_texture;
  
  CUDABuffer<u16> base_normals_buffer(kCameraHeight, kCameraWidth);
  CUDABufferPtr<uchar> base_color_buffer;
  cudaTextureObject_t base_color_texture;
  
  CUDABufferPtr<float> base_depth_buffer;
  CUDABufferPtr<uchar> base_calibrated_color_buffer;
  cudaTextureObject_t base_calibrated_color_texture;
  
  CreatePairwiseTrackingInputBuffersAndTextures(
      /*depth_width*/ kCameraWidth,
      /*depth_height*/ kCameraHeight,
      /*color_width*/ kCameraWidth,
      /*color_height*/ kCameraHeight,
      &base_depth_buffer,
      &base_calibrated_color_buffer,
      &base_color_buffer,
      &tracked_color_buffer,
      &base_calibrated_color_texture,
      &base_color_texture,
      &tracked_color_texture);
  
  LOG(INFO) << "NOTE: Testing photometric residuals only.";
  
  // Loop over tests.
  vector<bool> use_gradmag_values = {false, true};
  for (bool use_gradmag : use_gradmag_values) {
    LOG(INFO) << "--- Testing with use_gradmag: " << use_gradmag << " ---";
    
    vector<float> distortion_strengths = {/*2, 1, 0.75,*/ 0.5, 0.25, 0.1, 0.08, 0.07, 0.06, 0.05, 0.025, 0.01, 0.001};
    for (float distortion_strength : distortion_strengths) {
      srand(0);
      
      float convergence_error_norm_sum = 0;
      float accuracy_error_norm_sum = 0;
      constexpr int kNumTests = 10;
      for (int test = 0; test < kNumTests; ++ test) {
        // 1. Convergence test: distort the initial estimate, try to obtain the correct transformation.
        SE3f images_T_global[2];
        for (int i = 0; i < 2; ++ i) {
          images_T_global[i] = SE3f::exp(0.1f * SE3f::Tangent::Random());
        }
        
        RenderImages(
            camera,
            /*min_render_depth*/ 0.1f,
            /*max_render_depth*/ 2.0f * (kHeightmapZDistance + kHeightmapZVariation),
            renderer, mesh_opengl, use_gradmag,
            images_T_global, depth_params.raw_to_float_depth,
            &tracked_depth_buffer, &tracked_normals_buffer, tracked_color_buffer,
            &base_normals_buffer, base_depth_buffer, base_color_buffer);
        
        SE3f base_T_tracked_ground_truth = images_T_global[1] * images_T_global[0].inverse();
        SE3f base_T_tracked_initial_estimate = base_T_tracked_ground_truth * SE3f::exp(distortion_strength * SE3f::Tangent::Random());
        SE3f base_T_tracked_estimate;
        
        TrackFramePairwise(
            &pairwise_tracking_buffers,
            /*stream*/ 0,
            /*color_camera*/ camera,
            /*depth_camera*/ camera,
            depth_params,
            *cfactor_buffer,
            &pose_estimation_helper_buffers,
            /*render_window*/ nullptr,
            /*convergence_samples_file*/ nullptr,
            /*use_depth_residuals*/ false,
            /*use_descriptor_residuals*/ true,
            /*use_pyramid_level_0*/ true,
            use_gradmag,
            /* tracked frame */
            tracked_depth_buffer,
            tracked_normals_buffer,
            tracked_color_texture,
            /* base frame */
            *base_depth_buffer,
            base_normals_buffer,
            *base_color_buffer,
            base_color_texture,
            /* input / output poses */
            images_T_global[1].inverse(),
            /*test_different_initial_estimates*/ false,
            base_T_tracked_initial_estimate,
            base_T_tracked_initial_estimate,
            &base_T_tracked_estimate);
        
        // Check base_T_tracked_estimate against the ground truth
        Matrix<float, 6, 1> error = (base_T_tracked_estimate.inverse() * base_T_tracked_ground_truth).log();
        convergence_error_norm_sum += error.norm();
        
        
        // 2. Accuracy test: give the ground truth as initial estimate, verify that the tracking stays there.
        for (int i = 0; i < 2; ++ i) {
          images_T_global[i] = SE3f::exp(distortion_strength * SE3f::Tangent::Random());
        }
        
        RenderImages(
            camera,
            /*min_render_depth*/ 0.1f,
            /*max_render_depth*/ 2.0f * (kHeightmapZDistance + kHeightmapZVariation),
            renderer, mesh_opengl, use_gradmag,
            images_T_global, depth_params.raw_to_float_depth,
            &tracked_depth_buffer, &tracked_normals_buffer, tracked_color_buffer,
            &base_normals_buffer, base_depth_buffer, base_color_buffer);
        
        base_T_tracked_ground_truth = images_T_global[1] * images_T_global[0].inverse();
        base_T_tracked_initial_estimate = base_T_tracked_ground_truth;
        
        TrackFramePairwise(
            &pairwise_tracking_buffers,
            /*stream*/ 0,
            /*color_camera*/ camera,
            /*depth_camera*/ camera,
            depth_params,
            *cfactor_buffer,
            &pose_estimation_helper_buffers,
            /*render_window*/ nullptr,
            /*convergence_samples_file*/ nullptr,
            /*use_depth_residuals*/ false,
            /*use_descriptor_residuals*/ true,
            /*use_pyramid_level_0*/ true,
            use_gradmag,
            /* tracked frame */
            tracked_depth_buffer,
            tracked_normals_buffer,
            tracked_color_texture,
            /* base frame */
            *base_depth_buffer,
            base_normals_buffer,
            *base_color_buffer,
            base_color_texture,
            /* input / output poses */
            images_T_global[1].inverse(),
            /*test_different_initial_estimates*/ false,
            base_T_tracked_initial_estimate,
            base_T_tracked_initial_estimate,
            &base_T_tracked_estimate);
        
        // Check base_T_tracked_estimate against the ground truth
        error = (base_T_tracked_estimate.inverse() * base_T_tracked_ground_truth).log();
        accuracy_error_norm_sum += error.norm();
      }  // loop over tests
      
      LOG(INFO) << "Distortion strenght: " << distortion_strength
                << " --> average error norm (convergence): " << (convergence_error_norm_sum / kNumTests)
                << " --> average error norm (accuracy): " << (accuracy_error_norm_sum / kNumTests);
    }  // loop over distortion_strengths
  }  // loop over use_gradmag_values
  
  // Delete OpenGL objects before losing the OpenGL context
  renderer.reset();
  renderer_program_storage.reset();
  mesh_opengl.reset();
  
  SwitchOpenGLContext(no_opengl_context);
  opengl_context.Deinitialize();
}
