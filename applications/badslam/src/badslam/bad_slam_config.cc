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

#include "badslam/bad_slam_config.h"

namespace vis {

bool BadSlamConfig::Save(FILE* file) const {
  auto SaveInt32 = [&](int value) {
    i32 int32 = value;
    fwrite(&int32, sizeof(i32), 1, file);
  };
  
  auto SaveBool = [&](bool value) {
    u8 unsigned8 = value ? 1 : 0;
    fwrite(&unsigned8, sizeof(u8), 1, file);
  };
  
  auto SaveString = [&](const string& value) {
    u32 size = value.size();
    fwrite(&size, sizeof(u32), 1, file);
    fwrite(value.data(), 1, size, file);
  };
  
  
  fwrite(&raw_to_float_depth, sizeof(float), 1, file);
  SaveInt32(start_frame);
  SaveInt32(end_frame);
  fwrite(&target_frame_rate, sizeof(float), 1, file);
  SaveInt32(fps_restriction);
  SaveInt32(pyramid_level_for_depth);
  SaveInt32(pyramid_level_for_color);
  
  fwrite(&max_depth, sizeof(float), 1, file);
  fwrite(&baseline_fx, sizeof(float), 1, file);
  SaveInt32(median_filter_and_densify_iterations);
  fwrite(&bilateral_filter_sigma_xy, sizeof(float), 1, file);
  fwrite(&bilateral_filter_radius_factor, sizeof(float), 1, file);
  fwrite(&bilateral_filter_sigma_inv_depth, sizeof(float), 1, file);
  
  SaveInt32(max_surfel_count);
  SaveInt32(sparse_surfel_cell_size);
  fwrite(&surfel_merge_dist_factor, sizeof(float), 1, file);
  SaveInt32(min_observation_count_while_bootstrapping_1);
  SaveInt32(min_observation_count_while_bootstrapping_2);
  SaveInt32(min_observation_count);
  
  SaveInt32(num_scales);
  SaveBool(use_motion_model);
  
  SaveInt32(keyframe_interval);
  SaveInt32(max_num_ba_iterations_per_keyframe);
  SaveBool(disable_deactivation);
  SaveBool(use_geometric_residuals);
  SaveBool(use_photometric_residuals);
  SaveBool(optimize_intrinsics);
  SaveInt32(intrinsics_optimization_interval);
  SaveBool(do_surfel_updates);
  SaveBool(parallel_ba);
  SaveBool(use_pcg);
  SaveBool(estimate_poses);
  
  SaveInt32(min_free_gpu_memory_mb);
  
  SaveBool(enable_loop_detection);
  SaveBool(parallel_loop_detection);
  
  SaveString(loop_detection_vocabulary_path);
  SaveString(loop_detection_pattern_path);
  fwrite(&loop_detection_image_frequency, sizeof(float), 1, file);
  SaveInt32(loop_detection_images_width);
  SaveInt32(loop_detection_images_height);
  
  return true;
}

bool BadSlamConfig::Load(FILE* file) {
  auto LoadInt32 = [&]() {
    i32 value;
    if (fread(&value, sizeof(i32), 1, file) != 1) {
      LOG(ERROR) << "Encountered unexpected end of file";
    }
    return value;
  };
  
  auto LoadBool = [&]() {
    u8 value;
    if (fread(&value, sizeof(u8), 1, file) != 1) {
      LOG(ERROR) << "Encountered unexpected end of file";
    }
    return value != 0;
  };
  
  auto LoadFloat = [&]() {
    float value;
    if (fread(&value, sizeof(float), 1, file) != 1) {
      LOG(ERROR) << "Encountered unexpected end of file";
    }
    return value;
  };
  
  auto LoadString = [&]() {
    u32 size;
    if (fread(&size, sizeof(u32), 1, file) != 1) {
      LOG(ERROR) << "Encountered unexpected end of file";
    }
    if (size > 100000) {
      LOG(ERROR) << "Encountered a too long string (size: " << size << "), refusing to load it. Returning an empty string instead.";
      return string("");
    } else {
      // TODO: How to read a std::string without making a copy, such as with the
      //       temporary buffer here?
      vector<char> buffer(size + 1);
      buffer[size] = 0;
      if (fread(buffer.data(), 1, size, file) != size) {
        LOG(ERROR) << "Encountered unexpected end of file";
      }
      return string(buffer.data());
    }
  };
  
  
  raw_to_float_depth = LoadFloat();
  start_frame = LoadInt32();
  end_frame = LoadInt32();
  target_frame_rate = LoadFloat();
  fps_restriction = LoadInt32();
  pyramid_level_for_depth = LoadInt32();
  pyramid_level_for_color = LoadInt32();
  
  max_depth = LoadFloat();
  baseline_fx = LoadFloat();
  median_filter_and_densify_iterations = LoadInt32();
  bilateral_filter_sigma_xy = LoadFloat();
  bilateral_filter_radius_factor = LoadFloat();
  bilateral_filter_sigma_inv_depth = LoadFloat();
  
  max_surfel_count = LoadInt32();
  sparse_surfel_cell_size = LoadInt32();
  surfel_merge_dist_factor = LoadFloat();
  min_observation_count_while_bootstrapping_1 = LoadInt32();
  min_observation_count_while_bootstrapping_2 = LoadInt32();
  min_observation_count = LoadInt32();
  
  num_scales = LoadInt32();
  use_motion_model = LoadBool();
  
  keyframe_interval = LoadInt32();
  max_num_ba_iterations_per_keyframe = LoadInt32();
  disable_deactivation = LoadBool();
  use_geometric_residuals = LoadBool();
  use_photometric_residuals = LoadBool();
  optimize_intrinsics = LoadBool();
  intrinsics_optimization_interval = LoadInt32();
  do_surfel_updates = LoadBool();
  parallel_ba = LoadBool();
  use_pcg = LoadBool();
  estimate_poses = LoadBool();
  
  min_free_gpu_memory_mb = LoadInt32();
  
  enable_loop_detection = LoadBool();
  parallel_loop_detection = LoadBool();
  
  loop_detection_vocabulary_path = LoadString();
  loop_detection_pattern_path = LoadString();
  loop_detection_image_frequency = LoadFloat();
  loop_detection_images_width = LoadInt32();
  loop_detection_images_height = LoadInt32();
  
  return true;
}

}
