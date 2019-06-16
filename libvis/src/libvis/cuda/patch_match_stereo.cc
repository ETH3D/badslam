// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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


#include "libvis/cuda/patch_match_stereo.h"
#include "libvis/cuda/patch_match_stereo.cuh"

#include "libvis/cuda/pixel_corner_projector.h"

// #include "libvis/point_cloud.h"  // for debugging only
// #include "libvis/render_display.h"  // for debugging only

namespace vis {

struct ConnectedComponent {
  ConnectedComponent(int _parent, int _pixel_count)
      : parent(_parent), pixel_count(_pixel_count) {}

  int parent;
  int pixel_count;
  bool should_be_removed;
};

int PatchMatchStereoCUDA::ConvertMatchMetric(MatchMetric metric) {
  if (metric == MatchMetric::kSSD) {
    return kPatchMatchStereo_MatchMetric_SSD;
  } else if (metric == MatchMetric::kZNCC) {
    return kPatchMatchStereo_MatchMetric_ZNCC;
  } else if (metric == MatchMetric::kCensus) {
    return kPatchMatchStereo_MatchMetric_Census;
  } else {
    LOG(FATAL) << "Invalid argument";
    return -1;
  }
}

void PatchMatchStereoCUDA::RemoveSmallConnectedComponentsInInvDepthMap(
    float separator_value,
    int min_component_size,
    int min_x,
    int min_y,
    int max_x,
    int max_y,
    Image<float>* inv_depth_map) {
  // Find connected components and calculate number of certain pixels.
  vector<ConnectedComponent> components;
  constexpr size_t kPreallocationSize = 4096u;
  components.reserve(kPreallocationSize);
  Image<int> component_image(inv_depth_map->width(), inv_depth_map->height());
  
  for (int y = min_y; y <= max_y; ++ y) {
    for (int x = min_x; x <= max_x; ++ x) {
      // Mark pixel as invalid if it has the separator value.
      if (inv_depth_map->operator()(x, y) == separator_value) {
        component_image(x, y) = -1;
        continue;
      }
      
      if (x > min_x &&
          component_image(x - 1, y) != -1 &&
          DepthIsSimilar(inv_depth_map->operator()(x - 1, y), inv_depth_map->operator()(x, y))) {
        // Merge into left component.
        component_image(x, y) = component_image(x - 1, y);
        ConnectedComponent* const component = &components[component_image(x - 1, y)];
        ConnectedComponent* parent = component;
        while (&components[parent->parent] != parent) {
          ConnectedComponent* const higher_parent = &components[parent->parent];
          parent->parent = higher_parent->parent;
          parent = higher_parent;
        }
        parent->pixel_count += 1;
        
        if (y > min_y &&
            component_image(x, y - 1) != -1 &&
            DepthIsSimilar(inv_depth_map->operator()(x, y - 1), inv_depth_map->operator()(x, y))) {
          // Merge left into top component.
          // Notice: leaf, parent and target components may be the same.
          ConnectedComponent* const left_component = &components[component_image(x - 1, y)];
          ConnectedComponent* parent = left_component;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          int certain_pixels = left_component->pixel_count;
          left_component->pixel_count = 0;
          certain_pixels += parent->pixel_count;
          parent->pixel_count = 0;
          
          ConnectedComponent* const top_component = &components[component_image(x, y - 1)];
          parent->parent = top_component->parent;
          while (&components[parent->parent] != parent) {
            ConnectedComponent* const higher_parent = &components[parent->parent];
            parent->parent = higher_parent->parent;
            parent = higher_parent;
          }
          components[parent->parent].pixel_count += certain_pixels;
        }
      } else if (y > min_y &&
                 component_image(x, y - 1) != -1 &&
                 DepthIsSimilar(inv_depth_map->operator()(x, y - 1), inv_depth_map->operator()(x, y))) {
        // Merge into top component.
        component_image(x, y) = component_image(x, y - 1);
        ConnectedComponent* const component = &components[component_image(x, y - 1)];
        ConnectedComponent* parent = component;
        while (&components[parent->parent] != parent) {
          ConnectedComponent* const higher_parent = &components[parent->parent];
          parent->parent = higher_parent->parent;
          parent = higher_parent;
        }
        parent->pixel_count += 1;
      } else {
        // Create a new component.
        components.emplace_back(components.size(), 1);
        component_image(x, y) = components.size() - 1;
      }
    }
  }
  
  // Resolve parents until the root and decide on which components to remove.
  for (size_t i = 0u, end = components.size(); i < end; ++i) {
    ConnectedComponent* const component = &components[i];
    ConnectedComponent* parent = &components[component->parent];
    while (&components[parent->parent] != parent) {
      ConnectedComponent* const higher_parent = &components[parent->parent];
      parent->parent = higher_parent->parent;
      parent = higher_parent;
    }
    component->should_be_removed = parent->pixel_count < min_component_size;
  }
  
  // Remove bad connected components from image.
  for (int y = min_y; y <= max_y; ++y) {
    for (int x = min_x; x <= max_x; ++x) {
      if (component_image(x, y) >= 0 &&
          components[component_image(x, y)].should_be_removed) {
        inv_depth_map->operator()(x, y) = 0.f;
      }
    }
  }
}


PatchMatchStereoCUDA::PatchMatchStereoCUDA(int width, int height) {
  reference_image_gpu_.reset(new CUDABuffer<u8>(height, width));
  stereo_image_gpu_.reset(new CUDABuffer<u8>(height, width));
  inv_depth_map_gpu_.reset(new CUDABuffer<float>(height, width));
  inv_depth_map_gpu_2_.reset(new CUDABuffer<float>(height, width));
  
  normals_.reset(new CUDABuffer<char2>(height, width));
  costs_.reset(new CUDABuffer<float>(height, width));
  costs_2_.reset(new CUDABuffer<float>(height, width));
  random_states_.reset(new CUDABuffer<curandState>(height, width));
  lambda_.reset(new CUDABuffer<float>(height, width));
  
  second_best_inv_depth_map_gpu_.reset(new CUDABuffer<float>(height, width));
  second_best_normals_.reset(new CUDABuffer<char2>(height, width));
  second_best_costs_.reset(new CUDABuffer<float>(height, width));
  second_best_costs_2_.reset(new CUDABuffer<float>(height, width));
}

void PatchMatchStereoCUDA::ComputeDepthMap(
//     const Camera& reference_camera,
    const CUDAUnprojectionLookup2D& reference_unprojection,
    const Image<u8>& reference_image,
    const SE3f& reference_image_tr_global,
    const Camera& stereo_camera,
    const Image<u8>& stereo_image,
    const SE3f& stereo_image_tr_global,
    Image<float>* inv_depth_map,
    Image<float>* lr_consistency_inv_depth_map) {
  CHECK_EQ(reference_image.width(), reference_image_gpu_->width());
  CHECK_EQ(reference_image.height(), reference_image_gpu_->height());
  
  CHECK_EQ(stereo_image.width(), stereo_image_gpu_->width());
  CHECK_EQ(stereo_image.height(), stereo_image_gpu_->height());
  
  // NOTE: Allocating buffers and textures each time this is called is slow. Cache them for speedup.
  
  inv_depth_map->SetSize(reference_image.width(), reference_image.height());
  
  PixelCornerProjector stereo_camera_projector(stereo_camera);
  
  SE3f stereo_image_tr_reference_image_se3 = stereo_image_tr_global * reference_image_tr_global.inverse();
  CUDAMatrix3x4 stereo_image_tr_reference_image = CUDAMatrix3x4(stereo_image_tr_reference_image_se3.matrix3x4());
  CUDAMatrix3x4 reference_image_tr_stereo_image = CUDAMatrix3x4(stereo_image_tr_reference_image_se3.inverse().matrix3x4());
  
  // Upload color images to the GPU
  reference_image_gpu_->UploadAsync(0, reference_image);
  stereo_image_gpu_->UploadAsync(0, stereo_image);
  
  cudaTextureObject_t stereo_texture;
  stereo_image_gpu_->CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      &stereo_texture);
  
  cudaTextureObject_t reference_texture;
  reference_image_gpu_->CreateTextureObject(
      cudaAddressModeClamp,
      cudaAddressModeClamp,
      cudaFilterModeLinear,
      cudaReadModeNormalizedFloat,
      /*use_normalized_coordinates*/ false,
      &reference_texture);
  
  // Initialize the depth and normals randomly, and compute initial matching costs.
  InitPatchMatchCUDA(
      /*stream*/ 0,
      ConvertMatchMetric(match_metric_),
      context_radius_,
      max_normal_2d_length_,
      reference_unprojection.lookup_texture(),
      reference_image_gpu_->ToCUDA(),
      reference_texture,
      stereo_image_tr_reference_image,
      stereo_camera_projector.ToCUDA(),
      stereo_texture,
      1.0f / min_initial_depth_,
      1.0f / max_initial_depth_,
      &inv_depth_map_gpu_->ToCUDA(),
      &normals_->ToCUDA(),
      &costs_->ToCUDA(),
      &random_states_->ToCUDA(),
      &lambda_->ToCUDA());
  
  // Perform PatchMatch iterations
  for (int iteration = 0; iteration < iteration_count_; ++ iteration) {
    // Attempt mutations.
    PatchMatchMutationStepCUDA(
        /*stream*/ 0,
        ConvertMatchMetric(match_metric_),
        context_radius_,
        max_normal_2d_length_,
        reference_unprojection.lookup_texture(),
        reference_image_gpu_->ToCUDA(),
        reference_texture,
        stereo_image_tr_reference_image,
        stereo_camera_projector.ToCUDA(),
        stereo_texture,
        std::pow(0.5f, std::min(iteration + 1, 6 /*TODO: Make parameter*/)) * (1.0f / min_initial_depth_ - 1.0f / max_initial_depth_),
        &inv_depth_map_gpu_->ToCUDA(),
        &normals_->ToCUDA(),
        &costs_->ToCUDA(),
        &random_states_->ToCUDA());
    
//     // Optimize locally.
//     // TODO: This is only implemented for SSD cost with pinhole cameras.
//     //       It will not give proper results for other settings!
//     //       Currently, we use many iterations in order to test many hypotheses
//     //       and a final refinement step to essentially get the same result
//     //       as with local optimization.
//     PatchMatchOptimizationStepCUDA(
//         /*stream*/ 0,
//         ConvertMatchMetric(match_metric_),
//         context_radius_,
//         max_normal_2d_length_,
//         reference_unprojection.lookup_texture(),
//         *reference_image_gpu_,
//         reference_texture,
//         stereo_image_tr_reference_image,
//         stereo_camera_projector.ToCUDA(),
//         stereo_texture,
//         inv_depth_map_gpu_.get(),
//         normals_.get(),
//         costs_.get(),
//         random_states_.get(),
//         lambda_.get());
    
    // Attempt propagations.
    PatchMatchPropagationStepCUDA(
        /*stream*/ 0,
        ConvertMatchMetric(match_metric_),
        context_radius_,
        reference_unprojection.lookup_texture(),
        reference_image_gpu_->ToCUDA(),
        reference_texture,
        stereo_image_tr_reference_image,
        stereo_camera_projector.ToCUDA(),
        stereo_texture,
        &inv_depth_map_gpu_->ToCUDA(),
        &normals_->ToCUDA(),
        &costs_->ToCUDA(),
        &random_states_->ToCUDA());
    
//     // DEBUG
//     inv_depth_map_gpu_->DownloadAsync(0, inv_depth_map);
//     static ImageDisplay debug_display;
//     debug_display.Update(*inv_depth_map, "depth debug", 0.f, 1.5f);
//     
//     Image<char2> normals_cpu(normals_->width(), normals_->height());
//     normals_->DownloadAsync(0, &normals_cpu);
//     Image<Vec3u8> normals_visualization(normals_->width(), normals_->height());
//     for (int y = 0; y < normals_->height(); ++ y) {
//       for (int x = 0; x < normals_->width(); ++ x) {
//         const char2& n = normals_cpu(x, y);
//         normals_visualization(x, y) = Vec3u8(n.x + 127, n.y + 127, 127);
//       }
//     }
//     static ImageDisplay debug_display_2;
//     debug_display_2.Update(normals_visualization, "normals debug");
//     
//     // Show point cloud
//     static shared_ptr<RenderDisplay> render_display = make_shared<RenderDisplay>();
//     static shared_ptr<RenderWindow> render_window = RenderWindow::CreateWindow("PatchMatch debug Visualization", 1280, 720, RenderWindow::API::kOpenGL, render_display);
//     
//     shared_ptr<Point3fCu8Cloud> cloud(new Point3fCu8Cloud());
//     inv_depth_map_gpu_->DownloadAsync(0, inv_depth_map);
//     IDENTIFY_CAMERA(reference_camera,
//         cloud->SetFromRGBDImage(*inv_depth_map, true, 0.f, reference_image, _reference_camera));
//     
//     render_display->SetUpDirection(Vec3f(0, 0, 1));
//     render_display->Update(cloud, "visualization cloud");
//     
//     std::getchar();
//     // END DEBUG
  }
  
  // Depth refinement
  constexpr int kRefinementSteps = 20;  // TODO: Make parameter
  constexpr float kRefinementRangeFactor = 0.02;  // Factor on inverse depth. TODO: Make parameter
  
  PatchMatchDiscreteRefinementStepCUDA(
      /*stream*/ 0,
      ConvertMatchMetric(match_metric_),
      context_radius_,
      reference_unprojection.lookup_texture(),
      reference_image_gpu_->ToCUDA(),
      reference_texture,
      stereo_image_tr_reference_image,
      stereo_camera_projector.ToCUDA(),
      stereo_texture,
      kRefinementSteps,
      kRefinementRangeFactor,
      &inv_depth_map_gpu_->ToCUDA(),
      &normals_->ToCUDA(),
      &costs_->ToCUDA());
  
  // For outlier filtering, do a second round of PatchMatch Stereo while
  // excluding the depth ranges next to the results of the first round. This
  // will find the places with the second best cost, which allows to compare the
  // best and the second best cost to reject depth estimates for pixels where
  // those costs are similar and the depth is thus ambiguous.
  if (second_best_min_cost_factor_ > 1) {
    constexpr int kSecondBestIterations = 30;  // TODO: Make parameter
    
    InitPatchMatchCUDA(
        /*stream*/ 0,
        ConvertMatchMetric(match_metric_),
        context_radius_,
        max_normal_2d_length_,
        reference_unprojection.lookup_texture(),
        reference_image_gpu_->ToCUDA(),
        reference_texture,
        stereo_image_tr_reference_image,
        stereo_camera_projector.ToCUDA(),
        stereo_texture,
        1.0f / min_initial_depth_,
        1.0f / max_initial_depth_,
        &second_best_inv_depth_map_gpu_->ToCUDA(),
        &second_best_normals_->ToCUDA(),
        &second_best_costs_->ToCUDA(),
        &random_states_->ToCUDA(),
        &lambda_->ToCUDA(),
        second_best_min_distance_factor_,
        &inv_depth_map_gpu_->ToCUDA());
    
    // Perform PatchMatch iterations
    for (int iteration = 0; iteration < kSecondBestIterations; ++ iteration) {
      // Attempt mutations.
      PatchMatchMutationStepCUDA(
          /*stream*/ 0,
          ConvertMatchMetric(match_metric_),
          context_radius_,
          max_normal_2d_length_,
          reference_unprojection.lookup_texture(),
          reference_image_gpu_->ToCUDA(),
          reference_texture,
          stereo_image_tr_reference_image,
          stereo_camera_projector.ToCUDA(),
          stereo_texture,
          std::pow(0.5f, std::min(iteration + 1, 6 /*TODO: Make parameter*/)) * (1.0f / min_initial_depth_ - 1.0f / max_initial_depth_),
          &second_best_inv_depth_map_gpu_->ToCUDA(),
          &second_best_normals_->ToCUDA(),
          &second_best_costs_->ToCUDA(),
          &random_states_->ToCUDA(),
          second_best_min_distance_factor_,
          &inv_depth_map_gpu_->ToCUDA());
      
      // Attempt propagations.
      PatchMatchPropagationStepCUDA(
          /*stream*/ 0,
          ConvertMatchMetric(match_metric_),
          context_radius_,
          reference_unprojection.lookup_texture(),
          reference_image_gpu_->ToCUDA(),
          reference_texture,
          stereo_image_tr_reference_image,
          stereo_camera_projector.ToCUDA(),
          stereo_texture,
          &second_best_inv_depth_map_gpu_->ToCUDA(),
          &second_best_normals_->ToCUDA(),
          &second_best_costs_->ToCUDA(),
          &random_states_->ToCUDA(),
          second_best_min_distance_factor_,
          &inv_depth_map_gpu_->ToCUDA());
    }
    
  //   // DEBUG: Visualize the ratio second_best_costs_ / costs_.
  //   Image<float> costs_cpu(costs_->width(), costs_->height());
  //   costs_->DownloadAsync(0, &costs_cpu);
  //   
  //   Image<float> second_best_costs_cpu(second_best_costs_->width(), second_best_costs_->height());
  //   second_best_costs_->DownloadAsync(0, &second_best_costs_cpu);
  //   
  //   Image<Vec3u8> cost_factor_visualization(costs_->width(), costs_->height());
  //   for (int y = 0; y < costs_->height(); ++ y) {
  //     for (int x = 0; x < costs_->width(); ++ x) {
  //       float cost = costs_cpu(x, y);
  //       float second_best_cost = second_best_costs_cpu(x, y);
  //       
  //       if (!(second_best_cost >= second_best_min_cost_factor_ * cost)) {
  //         cost_factor_visualization(x, y) = Vec3u8(255, 0, 0);  // red: filtered because of this criterion
  //       } else {
  //         float factor = second_best_cost / cost;
  //         cost_factor_visualization(x, y) = Vec3u8::Constant(255.99f * (factor - 1) / (second_best_min_cost_factor_ - 1));  // black to white: far to close from filtering
  //       }
  //     }
  //   }
  //   
  //   static ImageDisplay cost_factor_display;
  //   cost_factor_display.Update(cost_factor_visualization, "Cost factor visualization");
  //   std::getchar();
  //   // END DEBUG
  }
  
//   // DEBUG: Visualize the depth image before outlier filtering.
//   inv_depth_map_gpu_->DownloadAsync(0, inv_depth_map);
//   static ImageDisplay raw_inv_depth_display;
//   raw_inv_depth_display.Update(*inv_depth_map, "Raw depth before filtering", 0.f, 1.5f);
  
  // Automatic ping-pong buffer handling.
  CUDABuffer<float>* cur_inv_depth = inv_depth_map_gpu_.get();
  CUDABuffer<float>* other_inv_depth = inv_depth_map_gpu_2_.get();
  
  // Post-processing step: median filtering (excluding invalid pixels).
  MedianFilterDepthMap3x3CUDA(
      /*stream*/ 0,
      context_radius_,
      &cur_inv_depth->ToCUDA(),
      &other_inv_depth->ToCUDA(),
      &costs_->ToCUDA(),
      &costs_2_->ToCUDA(),
      &second_best_costs_->ToCUDA(),
      &second_best_costs_2_->ToCUDA());
  std::swap(cur_inv_depth, other_inv_depth);
  
//   // DEBUG: Visualize the depth image before outlier filtering.
//   cur_inv_depth->DownloadAsync(0, inv_depth_map);
//   static ImageDisplay median_filtered_inv_depth_display;
//   median_filtered_inv_depth_display.Update(*inv_depth_map, "Raw depth after median filtering", 0.f, 1.5f);
  
  // Left-right consistency check (optional)
  if (lr_consistency_inv_depth_map) {
    CUDABuffer<float> lr_consistency_inv_depth(lr_consistency_inv_depth_map->height(), lr_consistency_inv_depth_map->width());
    lr_consistency_inv_depth.UploadAsync(0, *lr_consistency_inv_depth_map);
    
    PatchMatchLeftRightConsistencyCheckCUDA(
        /*stream*/ 0,
        context_radius_,
        lr_consistency_factor_threshold_,
        reference_unprojection.lookup_texture(),
        stereo_image_tr_reference_image,
        stereo_camera_projector.ToCUDA(),
        lr_consistency_inv_depth.ToCUDA(),
        &cur_inv_depth->ToCUDA(),
        &other_inv_depth->ToCUDA());
    std::swap(cur_inv_depth, other_inv_depth);
  }
  
  // Outlier filtering
  const float epipolar_gradient_threshold =
      (1 + 2 * context_radius_) *
      (1 + 2 * context_radius_) *
      min_epipolar_gradient_per_pixel_;
  
  PatchMatchFilterOutliersCUDA(
      /*stream*/ 0,
      context_radius_,
      1.f / max_depth_,
      required_range_min_depth_,
      required_range_max_depth_,
      reference_unprojection.lookup_texture(),
      reference_image_gpu_->ToCUDA(),
      reference_texture,
      stereo_image_tr_reference_image,
      reference_image_tr_stereo_image,
      stereo_camera_projector.ToCUDA(),
      stereo_texture,
      &cur_inv_depth->ToCUDA(),
      &other_inv_depth->ToCUDA(),
      &normals_->ToCUDA(),
      &costs_2_->ToCUDA(),
      cost_threshold_,
      epipolar_gradient_threshold,
      -cos(angle_threshold_),
      &second_best_costs_2_->ToCUDA(),
      second_best_min_cost_factor_);
  std::swap(cur_inv_depth, other_inv_depth);
  
//   // DEBUG: Visualize the depth image after partial outlier filtering.
//   cur_inv_depth->DownloadAsync(0, inv_depth_map);
//   static ImageDisplay partial_filtered_inv_depth_display;
//   partial_filtered_inv_depth_display.Update(*inv_depth_map, "Raw depth after partial filtering", 0.f, 1.5f);
  
  // Post-processing step: bilateral filtering (excluding invalid pixels).
  BilateralFilterCUDA(
      /*stream*/ 0,
      bilateral_filter_sigma_xy_,
      bilateral_filter_sigma_inv_depth_,
      bilateral_filter_radius_factor_,
      cur_inv_depth->ToCUDA(),
      &other_inv_depth->ToCUDA());
  std::swap(cur_inv_depth, other_inv_depth);
  
//   // DEBUG: Visualize the depth image after bilateral filtering.
//   cur_inv_depth->DownloadAsync(0, inv_depth_map);
//   static ImageDisplay bilateral_filtered_inv_depth_display;
//   bilateral_filtered_inv_depth_display.Update(*inv_depth_map, "Raw depth after bilateral filtering", 0.f, 1.5f);
  
  // Post-processing step: small hole filling (some iterations).
  constexpr int kNumHoleFillingIterations = 3;  // TODO: make parameter
  for (int iteration = 0; iteration < kNumHoleFillingIterations; ++ iteration) {
    FillHolesCUDA(
        /*stream*/ 0,
        cur_inv_depth->ToCUDA(),
        &other_inv_depth->ToCUDA());
    std::swap(cur_inv_depth, other_inv_depth);
  }
  
  // Download result to CPU
  cur_inv_depth->DownloadAsync(0, inv_depth_map);
  
  // Outlier filtering by removing small connected components.
  RemoveSmallConnectedComponentsInInvDepthMap(
      0, min_component_size_,
      context_radius_, context_radius_,
      inv_depth_map->width() - 1 - context_radius_,
      inv_depth_map->height() - 1 - context_radius_,
      inv_depth_map);
  
  cudaDestroyTextureObject(reference_texture);
  cudaDestroyTextureObject(stereo_texture);
}

void PatchMatchStereoCUDA::GetNormals(Image<Vec3f>* normals) {
  Image<char2> normals_cpu(normals_->width(), normals_->height());
  normals_->DownloadAsync(0, &normals_cpu);
  
  normals->SetSize(normals_->width(), normals_->height());
  
  // For convenience, convert the compressed normals to 3D vectors
  for (int y = 0; y < normals_->height(); ++ y) {
    for (int x = 0; x < normals_->width(); ++ x) {
      const char2& normal_xy_char = normals_cpu(x, y);
      float2 normal_xy = make_float2(
          normal_xy_char.x * (1 / 127.f), normal_xy_char.y * (1 / 127.f));
      float normal_z = -sqrtf(1.f - normal_xy.x * normal_xy.x - normal_xy.y * normal_xy.y);
      
      (*normals)(x, y) = Vec3f(normal_xy.x, normal_xy.y, normal_z);
    }
  }
}

}
