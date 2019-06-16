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


#include "badslam/pairwise_frame_tracking.h"

#include "badslam/convergence_analysis.h"
#include "badslam/kernels.h"
#include "badslam/render_window.h"
#include "badslam/util.cuh"

namespace vis {

PairwiseFrameTrackingBuffers::PairwiseFrameTrackingBuffers(
    int depth_width, int depth_height, int num_scales) {
  base_depth.resize(num_scales);
  base_normals.resize(num_scales);
  base_color.resize(num_scales);
  base_color_texture.resize(num_scales);
  
  tracked_depth.resize(num_scales);
  tracked_normals.resize(num_scales);
  tracked_color.resize(num_scales);
  tracked_color_texture.resize(num_scales);
  
  for (u32 scale = 0; scale < num_scales; ++ scale) {
    int scale_width = depth_width / pow(2, scale);
    int scale_height = depth_height / pow(2, scale);
    
    base_depth[scale].reset(new CUDABuffer<float>(scale_height, scale_width));
    base_normals[scale].reset(new CUDABuffer<u16>(scale_height, scale_width));
    base_color[scale].reset(new CUDABuffer<uchar>(scale_height, scale_width));
    base_color[scale]->CreateTextureObject(
        cudaAddressModeClamp,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeNormalizedFloat,
        /*use_normalized_coordinates*/ false,
        &base_color_texture[scale]);
    
    // Scale 0 is not used in these arrays for multi-res tracking, but is used for loop closure tracking (except normals)
    tracked_depth[scale].reset(new CUDABuffer<float>(scale_height, scale_width));
    if (scale >= 1) {
      tracked_normals[scale].reset(new CUDABuffer<u16>(scale_height, scale_width));
    }
    tracked_color[scale].reset(new CUDABuffer<uchar>(scale_height, scale_width));
    tracked_color[scale]->CreateTextureObject(
        cudaAddressModeClamp,
        cudaAddressModeClamp,
        cudaFilterModeLinear,
        cudaReadModeNormalizedFloat,
        /*use_normalized_coordinates*/ false,
        &tracked_color_texture[scale]);
  }
}

void TrackFramePairwise_DebugDisplayNormalImage(
    cudaStream_t stream,
    const CUDABuffer<u16>& normals_buffer,
    ImageDisplay* display,
    const char* window_title) {
  Image<u16> normals_cpu(normals_buffer.width(), normals_buffer.height());
  normals_buffer.DownloadAsync(stream, &normals_cpu);
  
  Image<Vec3u8> debug_image(normals_buffer.width(), normals_buffer.height());
  for (u32 y = 0; y < debug_image.height(); ++ y) {
    for (u32 x = 0; x < debug_image.width(); ++ x) {
      float3 normal = U16ToImageSpaceNormal(normals_cpu(x, y));
      debug_image(x, y) = Vec3u8(255.99f * 0.5f * (normal.x + 1),
                                 255.99f * 0.5f * (normal.y + 1),
                                 255.99f * 0.5f * (normal.z + 1));
    }
  }
  
  display->Update(debug_image, window_title);
}

void TrackFramePairwise_DebugDisplayGradmagImage(
    cudaStream_t stream,
    const CUDABuffer<uchar>& color_buffer,
    ImageDisplay* display,
    const char* window_title) {
  Image<uchar> colors_cpu(color_buffer.width(), color_buffer.height());
  color_buffer.DownloadAsync(stream, &colors_cpu);
  display->Update(colors_cpu, window_title);
}

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
    cudaTextureObject_t* tracked_gradmag_texture) {
  calibrated_depth->reset(new CUDABuffer<float>(depth_height, depth_width));
  calibrated_gradmag->reset(new CUDABuffer<uchar>(depth_height, depth_width));
  base_kf_gradmag->reset(new CUDABuffer<uchar>(color_height, color_width));
  tracked_gradmag->reset(new CUDABuffer<uchar>(color_height, color_width));
  
  (*calibrated_gradmag)->CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      calibrated_gradmag_texture);
  (*base_kf_gradmag)->CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      base_kf_gradmag_texture);
  (*tracked_gradmag)->CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      tracked_gradmag_texture);
}

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
    /* tracked frame
     * NOTE: Color must be in the color camera intrinsics.
     */
    const CUDABuffer<u16>& tracked_depth_buffer,
    const CUDABuffer<u16>& tracked_normals_buffer,
    const cudaTextureObject_t tracked_color_texture,
    /* base frame
     * NOTE: Color must have been transformed to the depth camera intrinsics.
     */
    const CUDABuffer<float>& base_depth_buffer,
    const CUDABuffer<u16>& base_normals_buffer,
    const CUDABuffer<uchar>& base_color_buffer,
    const cudaTextureObject_t base_color_texture,
    /* input / output poses */
    const SE3f& global_T_base,  // for debugging only!
    bool test_different_initial_estimates,
    const SE3f& base_T_frame_initial_estimate_1,
    const SE3f& base_T_frame_initial_estimate_2,
    SE3f* out_base_T_frame_estimate) {
  static int call_counter = 0;
  ++ call_counter;
  
  // The following can be uncommented to debug specific calls to this function:
  // LOG(INFO) << "call_counter: " << call_counter;
  bool kDebug = false;  // bool kDebug = call_counter >= 1213;
  
  // Set this to true to debug calls that lead to apparent divergence:
  constexpr bool kDebugDivergence = false;
  
  const int num_scales = buffers->base_depth.size();
  
repeat_pose_estimation:;
  
  shared_ptr<Point3fC3u8Cloud> debug_frame_cloud;
  if (kDebug) {
    // Show point cloud.
    Image<u16> tracked_depth_buffer_cpu(tracked_depth_buffer.width(), tracked_depth_buffer.height());
    tracked_depth_buffer.DownloadAsync(stream, &tracked_depth_buffer_cpu);
    cudaStreamSynchronize(stream);
    
    Image<float> cfactor_buffer_cpu(cfactor_buffer.width(), cfactor_buffer.height());
    cfactor_buffer.DownloadAsync(stream, &cfactor_buffer_cpu);
    cudaStreamSynchronize(stream);
    
    usize point_count = 0;
    for (u32 y = 0; y < tracked_depth_buffer_cpu.height(); ++ y) {
      const u16* ptr = tracked_depth_buffer_cpu.row(y);
      const u16* end = ptr + tracked_depth_buffer_cpu.width();
      while (ptr < end) {
        if (!(*ptr & kInvalidDepthBit)) {
          ++ point_count;
        }
        ++ ptr;
      }
    }
    
    debug_frame_cloud.reset(new Point3fC3u8Cloud(point_count));
    usize point_index = 0;
    for (int y = 0; y < tracked_depth_buffer.height(); ++ y) {
      for (int x = 0; x < tracked_depth_buffer.width(); ++ x) {
        if (kInvalidDepthBit & tracked_depth_buffer_cpu(x, y)) {
          continue;
        }
        float depth = RawToCalibratedDepth(
            depth_params.a,
            cfactor_buffer_cpu(x / depth_params.sparse_surfel_cell_size,
                               y / depth_params.sparse_surfel_cell_size),
            depth_params.raw_to_float_depth,
            tracked_depth_buffer_cpu(x, y));
        
        Point3fC3u8& point = debug_frame_cloud->at(point_index);
        point.position() = depth * depth_camera.UnprojectFromPixelCenterConv(Vec2f(x, y));
        point.color() = Vec3u8(255, 80, 80);
        ++ point_index;
      }
    }
    
    if (render_window) {
      render_window->SetCurrentFramePose((global_T_base * base_T_frame_initial_estimate_1).matrix());  // TODO: Display this using a different frustum than that used to show the current frame?
      
      render_window->SetFramePointCloud(
          debug_frame_cloud,
          global_T_base * base_T_frame_initial_estimate_1);
      
      render_window->RenderFrame();
    }
    std::getchar();
  }
  
  const int kMaxIterationsPerScale = convergence_samples_file ? 100 : 30;
  
  // Specify which buffers to use.
  vector<CUDABuffer<float>*> base_depth(num_scales);
  vector<CUDABuffer<u16>*> base_normals(num_scales);
  vector<CUDABuffer<uchar>*> base_color(num_scales);
  vector<cudaTextureObject_t*> base_color_textures(num_scales);
  
  vector<CUDABuffer<float>*> tracked_depth(num_scales);
  vector<CUDABuffer<u16>*> tracked_normals(num_scales);
  vector<CUDABuffer<uchar>*> tracked_color(num_scales);
  vector<cudaTextureObject_t*> tracked_color_textures(num_scales);
  
  for (u32 scale = 0; scale < num_scales; ++ scale) {
    base_depth[scale] = buffers->base_depth[scale].get();
    base_normals[scale] = buffers->base_normals[scale].get();
    base_color[scale] = buffers->base_color[scale].get();
    base_color_textures[scale] = &buffers->base_color_texture[scale];
    
    tracked_depth[scale] = buffers->tracked_depth[scale].get();
    tracked_normals[scale] = (scale >= 1) ? buffers->tracked_normals[scale].get() : const_cast<CUDABuffer<u16>*>(&tracked_normals_buffer);  // TODO: avoid const_cast
    tracked_color[scale] = buffers->tracked_color[scale].get();
    tracked_color_textures[scale] = &buffers->tracked_color_texture[scale];
  }
  
  
  static CUDABufferPtr<float> debug_residual_image;
  if ((kDebug || convergence_samples_file) && !debug_residual_image) {
    debug_residual_image.reset(new CUDABuffer<float>(tracked_depth_buffer.height() / 2, tracked_depth_buffer.width() / 2));
  }
  
  // Set pointers to use provided base image
  base_depth[0] = const_cast<CUDABuffer<float>*>(&base_depth_buffer);  // TODO: avoid const_cast
  base_normals[0] = const_cast<CUDABuffer<u16>*>(&base_normals_buffer);
  base_color[0] = const_cast<CUDABuffer<uchar>*>(&base_color_buffer);
  base_color_textures[0] = const_cast<cudaTextureObject_t*>(&base_color_texture);
  
  if (use_pyramid_level_0) {
    // Convert tracked frame input to the expected format to use it on scale 0.
    CalibrateDepthCUDA(
        stream,
        depth_params,
        tracked_depth_buffer.ToCUDA(),
        &tracked_depth[0]->ToCUDA());
    
    tracked_color[/*TODO: should be pyramid_level_for_color*/ 0]->SetToReadModeNormalized(tracked_color_texture, stream);
    // TODO: Later, downsampling for the color must only start at pyramid_level_for_color!
  } else {  // if (!use_pyramid_level_0) {
    // Downsample input images
    if (depth_camera.width() != color_camera.width() &&
        depth_camera.width() != 2 * color_camera.width()) {
      // TODO: Inelegant. Also see another special case for this situation below.
      LOG(FATAL) << "The chosen depth / color pyramid level combination is not supported here.";
    }
    
    CalibrateAndDownsampleImagesCUDA(
        stream,
        depth_camera.width() == color_camera.width(),
        depth_params,
        tracked_depth_buffer.ToCUDA(),
        tracked_normals_buffer.ToCUDA(),
        tracked_color_texture,
        &tracked_depth[1]->ToCUDA(),
        &tracked_normals[1]->ToCUDA(),
        &tracked_color[1]->ToCUDA(),
        kDebug);
  }
  
  for (u32 scale = 0; scale < num_scales; ++ scale) {
    if (scale >= 1) {
      if (scale >= 2 || use_pyramid_level_0) {
        DownsampleImagesCUDA(
            stream,
            tracked_depth[scale - 1]->ToCUDA(),
            tracked_normals[scale - 1]->ToCUDA(),
            *tracked_color_textures[scale - 1],
            &tracked_depth[scale]->ToCUDA(),
            &tracked_normals[scale]->ToCUDA(),
            &tracked_color[scale]->ToCUDA(),
            kDebug);
      }
      
      DownsampleImagesCUDA(
          stream,
          base_depth[scale - 1]->ToCUDA(),
          base_normals[scale - 1]->ToCUDA(),
          *base_color_textures[scale - 1],
          &base_depth[scale]->ToCUDA(),
          &base_normals[scale]->ToCUDA(),
          &base_color[scale]->ToCUDA(),
          kDebug);
    }
  }
  
  if (convergence_samples_file) {
    *convergence_samples_file << "EstimateFramePoseMultiRes()" << std::endl;
  }
  
  // Coefficients for update equation: H * x = b
  Eigen::Matrix<float, 6, 6> H;
  Eigen::Matrix<float, 6, 1> b;
  
  SE3f base_T_frame_estimate;
  SE3f base_T_frame_chosen_initial_estimate;
  if (!test_different_initial_estimates) {
    base_T_frame_estimate = base_T_frame_initial_estimate_1;
    base_T_frame_chosen_initial_estimate = base_T_frame_initial_estimate_1;
  }
  
  // Iterate over scales
//   bool converged;
  for (int scale = num_scales - 1; scale >= (use_pyramid_level_0 ? 0 : 1); -- scale) {
    if (kDebug) {
      LOG(INFO) << "Debug: scale " << scale;
      
      static ImageDisplay base_depth_display;
      base_depth[scale]->DebugDisplay(stream, &base_depth_display, "debug base depth", 0.f, 3.f);
      
      static ImageDisplay tracked_depth_display;
      tracked_depth[scale]->DebugDisplay(stream, &tracked_depth_display, "debug tracked depth", 0.f, 3.f);
      
      static ImageDisplay base_normals_display;
      TrackFramePairwise_DebugDisplayNormalImage(
          stream,
          *base_normals[scale],
          &base_normals_display,
          "debug base normals");
      
      static ImageDisplay tracked_normals_display;
      TrackFramePairwise_DebugDisplayNormalImage(
          stream,
          *tracked_normals[scale],
          &tracked_normals_display,
          "debug tracked normals");
      
      static ImageDisplay base_gradmag_display;
      TrackFramePairwise_DebugDisplayGradmagImage(
          stream,
          *base_color[scale],
          &base_gradmag_display,
          "debug base gradmag");
      
      static ImageDisplay tracked_gradmag_display;
      TrackFramePairwise_DebugDisplayGradmagImage(
          stream,
          *tracked_color[scale],
          &tracked_gradmag_display,
          "debug tracked gradmag");
      
      LOG(INFO) << "Debug: showing images for scale " << scale;
    }
    
    if (convergence_samples_file) {
      *convergence_samples_file << "scale " << scale << std::endl;
    }
    
    float scaling_factor = pow(2, scale);
    // TODO: Inelegant special case for differing pyramid levels where only a color
    //       pyramid level that is higher than the depth pyramid level by one is supported.
    //       Also see another special case related to this situation above.
    shared_ptr<PinholeCamera4f> tracked_color_camera(color_camera.Scaled(
        (depth_camera.width() == color_camera.width()) ? (1.f / scaling_factor) : (2.f / scaling_factor)));
    shared_ptr<PinholeCamera4f> tracked_depth_camera(depth_camera.Scaled(1.f / scaling_factor));
    
    float threshold_factor = scaling_factor;
    
    // If this is not the first scale, test whether the costs are better at the
    // last scale's result or at the initial estimate, and continue with the
    // better pose. This hopefully avoids problems with divergence on small
    // scales. If this is the first scale, do an analogous choice between the
    // two provided initial estimates (TODO: in that case, the variable naming
    // is off).
    if (scale != num_scales - 1 || test_different_initial_estimates) {
  //     static int times_last_scale_chosen = 0;
  //     static int times_initial_estimate_chosen = 0;
      
      u32 residual_count_last_scale;
      float cost_last_scale;
      SE3f base_T_frame_last_scale = (scale != num_scales - 1) ? base_T_frame_estimate : base_T_frame_initial_estimate_1;
      ComputeCostAndResidualCountFromImagesCUDA(
          stream,
          use_depth_residuals,
          use_descriptor_residuals,
          *tracked_color_camera,
          *tracked_depth_camera,
          depth_params.baseline_fx,
          threshold_factor,
          *tracked_depth[scale],
          *tracked_normals[scale],
          *tracked_color_textures[scale],
          CUDAMatrix3x4(base_T_frame_last_scale.inverse().matrix3x4()),
          *base_depth[scale],
          *base_normals[scale],
          *base_color[scale],
          &residual_count_last_scale,
          &cost_last_scale,
          helper_buffers,
          use_gradmag);
      
      u32 residual_count_initial_estimate;
      float cost_initial_estimate;
      SE3f base_T_frame_initial_estimate = (scale != num_scales - 1) ? base_T_frame_chosen_initial_estimate : base_T_frame_initial_estimate_2;
      ComputeCostAndResidualCountFromImagesCUDA(
          stream,
          use_depth_residuals,
          use_descriptor_residuals,
          *tracked_color_camera,
          *tracked_depth_camera,
          depth_params.baseline_fx,
          threshold_factor,
          *tracked_depth[scale],
          *tracked_normals[scale],
          *tracked_color_textures[scale],
          CUDAMatrix3x4(base_T_frame_initial_estimate.inverse().matrix3x4()),
          *base_depth[scale],
          *base_normals[scale],
          *base_color[scale],
          &residual_count_initial_estimate,
          &cost_initial_estimate,
          helper_buffers,
          use_gradmag);
      
      // Selection heuristic based on residual count and cost:
      if (residual_count_last_scale > 2 * residual_count_initial_estimate) {
  //       if (scale != num_scales - 1) {
  //         ++ times_last_scale_chosen;
  //       }
        base_T_frame_estimate = base_T_frame_last_scale;
      } else if (residual_count_initial_estimate > 2 * residual_count_last_scale) {
  //       if (scale != num_scales - 1) {
  //         ++ times_initial_estimate_chosen;
  //       }
        base_T_frame_estimate = base_T_frame_initial_estimate;
      } else if (cost_last_scale < cost_initial_estimate) {
  //       if (scale != num_scales - 1) {
  //         ++ times_last_scale_chosen;
  //       }
        base_T_frame_estimate = base_T_frame_last_scale;
      } else {
  //       if (scale != num_scales - 1) {
  //         ++ times_initial_estimate_chosen;
  //       }
        base_T_frame_estimate = base_T_frame_initial_estimate;
      }
      
      if (scale == num_scales - 1) {
        base_T_frame_chosen_initial_estimate = base_T_frame_estimate;
      }
      
  //     static int counter = 0;
  //     ++ counter;
  //     if (counter % 50 == 0) {
  //       LOG(INFO) << "DEBUG: times_last_scale_chosen = " << times_last_scale_chosen;
  //       LOG(INFO) << "DEBUG: times_initial_estimate_chosen = " << times_initial_estimate_chosen;
  //     }
    }
    
//     converged = false;
    int iteration;
    for (iteration = 0; iteration < kMaxIterationsPerScale; ++ iteration) {
      if (kDebug) {
        LOG(INFO) << "Debug: iteration " << iteration;
      }
      
      // Accumulate coefficients from cost term Jacobians.
      // NOTE: We handle Eigen objects outside of code compiled by the CUDA
      //       compiler only, since otherwise there were wrong results on my
      //       laptop.
      u32 residual_count;
      float residual_sum;
      float H_temp[6 * (6 + 1) / 2];
      AccumulatePoseEstimationCoeffsFromImagesCUDA(
          stream,
          use_depth_residuals,
          use_descriptor_residuals,
          *tracked_color_camera,
          *tracked_depth_camera,
          depth_params.baseline_fx,
          threshold_factor,
          *tracked_depth[scale],
          *tracked_normals[scale],
          *tracked_color_textures[scale],
          CUDAMatrix3x4(base_T_frame_estimate.inverse().matrix3x4()),
          *base_depth[scale],
          *base_normals[scale],
          *base_color[scale],
          &residual_count,
          &residual_sum,
          H_temp,
          b.data(),
          kDebug || convergence_samples_file,
          debug_residual_image.get(),
          helper_buffers,
          use_gradmag);
      
      int index = 0;
      for (int row = 0; row < 6; ++ row) {
        for (int col = row; col < 6; ++ col) {
          H(row, col) = H_temp[index];
          ++ index;
        }
      }
      
      // Solve for the update x
      // NOTE: Not sure if using double is helpful here
      Eigen::Matrix<float, 6, 1> x = H.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(b.cast<double>()).cast<float>();
      
      if (kDebug) {
        LOG(INFO) << "Debug: x = " << std::endl << x;
        LOG(INFO) << "residual_count = " << residual_count;
        LOG(INFO) << "residual_sum = " << residual_sum;
        
        // Condition number of H using Frobenius norm. In octave / Matlab:
        // cond(H, "fro") = norm(H, "fro") * norm(inv(H), "fro")
        // If this is very high, the pose is probably not well constrained.
        // However, noise can make it appear well-conditioned when it should not be!
        float cond_H_fro = H.norm() * H.inverse().norm();
        LOG(INFO) << "Debug: cond(H, \"fro\") = " << cond_H_fro;
      }
      
      // TODO: The damping was introduced to combat the jittering; maybe Levenberg-Marquardt would be a better alternative?
      float damping = 1.f;
      if (scale == num_scales - 2) {
        damping = 0.5f;
      } else if (scale == num_scales - 1) {
        damping = 0.25f;
      }
      
      // Apply the (negative) update -x.
      base_T_frame_estimate = base_T_frame_estimate * SE3f::exp(-damping * x);
      
      if (kDebug) {
        Image<float> debug_residual_image_cpu(debug_residual_image->width(), debug_residual_image->height());
        debug_residual_image->DownloadAsync(stream, &debug_residual_image_cpu);
        Image<Vec3u8> residual_visualization(debug_residual_image->width(), debug_residual_image->height());
        for (int y = 0; y < debug_residual_image->height(); ++ y) {
          for (int x = 0; x < debug_residual_image->width(); ++ x) {
            // constexpr float kMaxResidual = 0.1f;  // for depth
            constexpr float kMaxResidual = 1e-7f * 20.f;  // for color
            float r = debug_residual_image_cpu(x, y);
            if (std::isnan(r)) {
              residual_visualization(x, y) = Vec3u8(255, 0, 0);
            } else {
              u8 intensity = std::min<float>(255, 255.99f * fabs(r) / kMaxResidual);
              residual_visualization(x, y) = Vec3u8(intensity, intensity, intensity);
            }
          }
        }
        static ImageDisplay residual_visualization_display;
        residual_visualization_display.Update(residual_visualization, "residual visualization");
        
        LOG(INFO) << "Debug: relative camera position: " << base_T_frame_estimate.translation().transpose();
        if (render_window) {
          render_window->SetCurrentFramePose((global_T_base * base_T_frame_estimate).matrix());  // TODO: Display this with a different frustum than the one used to show the current frame pose?
          
          render_window->SetFramePointCloud(
              debug_frame_cloud,
              global_T_base * base_T_frame_estimate);
          
          render_window->RenderFrame();
        }
        std::getchar();
      }
      
      // Check for convergence
      bool converged = IsScaleNPoseEstimationConverged(x, scaling_factor);
      if (!convergence_samples_file && converged) {
        if (kDebug) {
          LOG(INFO) << "Debug: Assuming convergence.";
        }
//         converged = true;
        ++ iteration;
        break;
      } else if (convergence_samples_file) {
        *convergence_samples_file << "iteration " << iteration << std::endl;
        *convergence_samples_file << "x " << x.transpose() << std::endl;
        *convergence_samples_file << "residual_sum " << residual_sum << std::endl;
      }
    }
  }
  
//   if (!converged) {
//     static int not_converged_count = 0;
//     ++ not_converged_count;
//     LOG(ERROR) << "not_converged_count increased to " << not_converged_count << " at call_counter value: " << call_counter;
//   }
  
//   static float average_iteration_count = 0;
//   average_iteration_count = ((call_counter - 1) * average_iteration_count + iteration) / (1.0f * call_counter);
//   if (call_counter % 200 == 0) {
//     LOG(INFO) << "Average pose optimization iteration count: " << average_iteration_count;
//   }
  
  // Debug check for divergence
  constexpr float kDebugDivergenceCheckThresholdDistance = 0.3f;
  if (kDebugDivergence &&
      (base_T_frame_chosen_initial_estimate.translation() - base_T_frame_estimate.translation()).squaredNorm() >=
       kDebugDivergenceCheckThresholdDistance * kDebugDivergenceCheckThresholdDistance) {
    LOG(ERROR) << "Pose estimation divergence detected (movement from initial estimate is larger than the threshold of " << kDebugDivergenceCheckThresholdDistance << " meters).";
    LOG(ERROR) << "(Use a backtrace to see in which part it occurred.)";
    LOG(ERROR) << "Chosen initial camera position: " << base_T_frame_chosen_initial_estimate.translation().transpose();
    LOG(ERROR) << "Current camera position: " << base_T_frame_estimate.translation().transpose();
    LOG(ERROR) << "Would you like to debug it (type y + Return for yes, n + Return for no)?";
    while (true) {
      int response = std::getchar();
      if (response == 'y' || response == 'Y') {
        // Repeat with debug enabled.
        kDebug = true;
        goto repeat_pose_estimation;
      } else if (response == 'n' || response == 'N') {
        break;
      }
    }
  }
  
  if (kDebug && render_window) {
    render_window->UnsetFramePointCloud();
  }
  
  *out_base_T_frame_estimate = base_T_frame_estimate;
}

}
