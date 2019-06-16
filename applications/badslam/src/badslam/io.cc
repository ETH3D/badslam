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

#include "badslam/io.h"

#include <fstream>
#include <iomanip>

#include "badslam/bad_slam.h"

namespace vis {

bool SaveState(
    const BadSlam& slam,
    const std::string& path) {
  FILE* file = fopen(path.c_str(), "wb");
  if (!file) {
    return false;
  }
  
  auto SaveInt32 = [&](int value) {
    i32 int32 = value;
    fwrite(&int32, sizeof(i32), 1, file);
  };
  
  auto SaveBool = [&](bool value) {
    u8 unsigned8 = value ? 1 : 0;
    fwrite(&unsigned8, sizeof(u8), 1, file);
  };
  
  auto SaveSE3f = [&](const SE3f& value) {
    fwrite(value.data(), sizeof(float), 7, file);
  };
  
  // Header
  const char* identifier = "BADSLAM";
  fwrite(identifier, 1, 7, file);
  
  u8 version = 1;
  fwrite(&version, 1, 1, file);
  
  
  // BadSlam
  // NOTE: Not saving parallel_ba_iteration_queue_.
  i32 base_kf_id = slam.base_kf() ? slam.base_kf()->id() : -1;
  fwrite(&base_kf_id, sizeof(i32), 1, file);
  
  vector<SE3f> motion_model_base_kf_tr_frame = slam.motion_model_base_kf_tr_frame();
  u32 size = motion_model_base_kf_tr_frame.size();
  fwrite(&size, sizeof(u32), 1, file);
  for (u32 i = 0; i < size; ++ i) {
    SaveSE3f(motion_model_base_kf_tr_frame[i]);
  }
  
  vector<shared_ptr<Keyframe>> queued_keyframes;
  vector<SE3f> queued_keyframes_last_kf_tr_this_kf;
  slam.GetQueuedKeyframes(
      &queued_keyframes,
      &queued_keyframes_last_kf_tr_this_kf);
  size = queued_keyframes.size();
  fwrite(&size, sizeof(u32), 1, file);
  for (u32 i = 0; i < size; ++ i) {
    i32 kf_frame_index = queued_keyframes[i] ? queued_keyframes[i]->frame_index() : -1;
    fwrite(&kf_frame_index, sizeof(i32), 1, file);
    SaveSE3f(queued_keyframes_last_kf_tr_this_kf[i]);
  }
  
  i32 last_frame_index = slam.last_frame_index();
  fwrite(&last_frame_index, sizeof(i32), 1, file);
  
  
  // Config
  if (!slam.config().Save(file)) {
    fclose(file);
    return false;
  }
  
  
  // RGBDVideo (frame poses)
  size = slam.rgbd_video()->frame_count();
  fwrite(&size, sizeof(u32), 1, file);
  for (u32 i = 0; i < size; ++ i) {
    SaveSE3f(slam.rgbd_video()->depth_frame(i)->global_T_frame());
  }
  
  
  // Direct BA
  // NOTE: Not saving active_surfels_.
  auto& ba = slam.direct_ba();
  
  PinholeCamera4f color_camera = ba.color_camera();
  SaveInt32(color_camera.type_int());
  SaveInt32(color_camera.width());
  SaveInt32(color_camera.height());
  SaveInt32(color_camera.parameter_count());
  fwrite(color_camera.parameters(), sizeof(float), color_camera.parameter_count(), file);
  
  SaveInt32(ba.pyramid_level_for_color());
  
  PinholeCamera4f depth_camera = ba.depth_camera();
  SaveInt32(depth_camera.type_int());
  SaveInt32(depth_camera.width());
  SaveInt32(depth_camera.height());
  SaveInt32(depth_camera.parameter_count());
  fwrite(depth_camera.parameters(), sizeof(float), depth_camera.parameter_count(), file);
  
  CUDABufferConstPtr<float> cfactor_buffer = ba.cfactor_buffer();
  Image<float> cfactor_cpu(cfactor_buffer->width(), cfactor_buffer->height());
  cfactor_buffer->DownloadAsync(0, &cfactor_cpu);
  SaveInt32(cfactor_cpu.width());
  SaveInt32(cfactor_cpu.height());
  SaveInt32(cfactor_cpu.stride());
  fwrite(cfactor_cpu.data(), 1, cfactor_cpu.height() * cfactor_cpu.stride(), file);
  
  DepthParameters depth_params = ba.depth_params();
  fwrite(&depth_params.a, sizeof(float), 1, file);
  fwrite(&depth_params.raw_to_float_depth, sizeof(float), 1, file);
  fwrite(&depth_params.baseline_fx, sizeof(float), 1, file);
  SaveInt32(depth_params.sparse_surfel_cell_size);
  
  SaveInt32(ba.keyframes().size());
  for (usize i = 0; i < ba.keyframes().size(); ++ i) {
    auto& keyframe = ba.keyframes()[i];
    SaveInt32(keyframe ? keyframe->id() : -1);
    if (keyframe) {
      SaveInt32(keyframe->frame_index());
      SaveInt32(static_cast<int>(keyframe->activation()));
      SaveInt32(keyframe->last_active_in_ba_iteration());
      SaveInt32(keyframe->last_covis_in_ba_iteration());
    }
  }
  
  SaveInt32(ba.surfel_count());
  SaveInt32(ba.surfels_size());
  
  CUDABufferConstPtr<float> surfels = ba.surfels();
  vector<float> surfel_data(ba.surfels_size());
  for (int i = 0; i < kSurfelDataAttributeCount; ++ i) {
    surfels->DownloadPartAsync(i * surfels->ToCUDA().pitch(), ba.surfels_size() * sizeof(float), 0, surfel_data.data());
    fwrite(surfel_data.data(), sizeof(float), ba.surfels_size(), file);
  }
  
  SaveInt32(ba.ba_iteration_count());
  SaveInt32(ba.last_ba_iteration_count());
  
  SaveBool(ba.use_depth_residuals());
  SaveBool(ba.use_descriptor_residuals());
  
  SaveInt32(ba.min_observation_count_while_bootstrapping_1());
  SaveInt32(ba.min_observation_count_while_bootstrapping_2());
  SaveInt32(ba.min_observation_count());
  
  float surfel_merge_dist_factor = ba.surfel_merge_dist_factor();
  fwrite(&surfel_merge_dist_factor, sizeof(float), 1, file);
  
  fclose(file);
  return true;
}

bool LoadState(
    BadSlam* slam,
    const std::string& path,
    std::function<bool (int, int)> progress_function) {
  // TODO: If loading is aborted, this function should ideally not make any changes
  //       to the SLAM object. So, all possible reasons for aborting should be checked before making changes.
  
  FILE* file = fopen(path.c_str(), "rb");
  if (!file) {
    return false;
  }
  
  auto LoadInt32 = [&]() {
    i32 int32;
    if (fread(&int32, sizeof(i32), 1, file) != 1) {
      LOG(ERROR) << "Unexpected end of file.";
    }
    return int32;
  };
  
  auto LoadU32 = [&]() {
    u32 value;
    if (fread(&value, sizeof(u32), 1, file) != 1) {
      LOG(ERROR) << "Unexpected end of file.";
    }
    return value;
  };
  
  auto LoadFloat = [&]() {
    float value;
    if (fread(&value, sizeof(float), 1, file) != 1) {
      LOG(ERROR) << "Unexpected end of file.";
    }
    return value;
  };
  
  auto LoadBool = [&]() {
    u8 unsigned8;
    if (fread(&unsigned8, sizeof(u8), 1, file) != 1) {
      LOG(ERROR) << "Unexpected end of file.";
    }
    return unsigned8 != 0;
  };
  
  auto LoadSE3f = [&]() {
    SE3f value;
    if (fread(value.data(), sizeof(float), 7, file) != 7) {
      LOG(ERROR) << "Unexpected end of file.";
    }
    return value;
  };
  
  // Header
  char identifier[7];
  if (fread(identifier, 1, 7, file) != 7) {
    LOG(ERROR) << "Encountered the end of file before finishing reading the file identifier.";
    fclose(file); return false;
  }
  if (identifier[0] != 'B' ||
      identifier[1] != 'A' ||
      identifier[2] != 'D' ||
      identifier[3] != 'S' ||
      identifier[4] != 'L' ||
      identifier[5] != 'A' ||
      identifier[6] != 'M') {
    LOG(ERROR) << "File identifier does not match.";
    fclose(file); return false;
  }
  
  u8 version;
  if (fread(&version, 1, 1, file) != 1) {
    LOG(ERROR) << "Unexpected end of file.";
    fclose(file); return false;
  }
  if (version != 1) {
    LOG(ERROR) << "Unknown file format version.";
    fclose(file); return false;
  }
  
  
  // BadSlam
  int base_kf_id = LoadInt32();
  
  u32 size = LoadU32();
  if (size > 1000) {
    LOG(ERROR) << "Excessive motion model size, refusing to load.";
    fclose(file); return false;
  }
  vector<SE3f> motion_model_base_kf_tr_frame(size);
  for (u32 i = 0; i < size; ++ i) {
    motion_model_base_kf_tr_frame[i] = LoadSE3f();
  }
  slam->SetMotionModelBaseKFTrFrame(motion_model_base_kf_tr_frame);
  
  size = LoadU32();
  if (size > 10000) {
    LOG(ERROR) << "Excessive queued keyframes size, refusing to load.";
    fclose(file); return false;
  }
  vector<int> queued_keyframes_frame_indices(size);
  vector<SE3f> queued_keyframes_last_kf_tr_this_kf(size);
  for (u32 i = 0; i < size; ++ i) {
    queued_keyframes_frame_indices[i] = LoadInt32();
    queued_keyframes_last_kf_tr_this_kf[i] = LoadSE3f();
  }
  
  slam->SetLastIndexInVideo(LoadInt32());
  
  // Config
  if (!slam->config().Load(file)) {
    fclose(file); return false;
  }
  
  // TODO: We should ensure that *all* parameters used by DirectBA are set properly.
  //       It might be best to have a separate struct of BA options.
  auto& ba = slam->direct_ba();
  ba.SetUseDescriptorResiduals(slam->config().use_photometric_residuals);
  ba.SetUseDepthResiduals(slam->config().use_geometric_residuals);
  ba.SetMinObservationCount(slam->config().min_observation_count);
  ba.SetMinObservationCountWhileBootstrapping1(slam->config().min_observation_count_while_bootstrapping_1);
  ba.SetMinObservationCountWhileBootstrapping2(slam->config().min_observation_count_while_bootstrapping_2);
  
  
  // RGBDVideo (frame poses)
  size = LoadU32();
  if (size != slam->rgbd_video()->frame_count()) {
    LOG(ERROR) << "Loaded frame count does not match the existing frame count in the dataset.";
    fclose(file); return false;
  }
  for (u32 i = 0; i < size; ++ i) {
    SE3f global_T_frame = LoadSE3f();
    slam->rgbd_video()->color_frame_mutable(i)->SetGlobalTFrame(global_T_frame);
    slam->rgbd_video()->depth_frame_mutable(i)->SetGlobalTFrame(global_T_frame);
  }
  
  
  // Direct BA
  ba.keyframes_mutable()->clear();
  
  int color_camera_type_int = LoadInt32();
  int color_camera_width = LoadInt32();
  int color_camera_height = LoadInt32();
  int color_camera_parameter_count = LoadInt32();
  if (color_camera_type_int != static_cast<int>(Camera::Type::kPinholeCamera4f) ||
      color_camera_parameter_count != 4) {
    LOG(ERROR) << "Unexpected color camera type or parameter count.";
    fclose(file); return false;
  }
  float color_camera_parameters[4];
  if (fread(color_camera_parameters, sizeof(float), color_camera_parameter_count, file) != color_camera_parameter_count) {
    LOG(ERROR) << "Unexpected end of file.";
    fclose(file); return false;
  }
  ba.SetColorCamera(PinholeCamera4f(color_camera_width, color_camera_height, color_camera_parameters));
  
  ba.SetPyramidLevelForColor(LoadInt32());
  
  int depth_camera_type_int = LoadInt32();
  int depth_camera_width = LoadInt32();
  int depth_camera_height = LoadInt32();
  int depth_camera_parameter_count = LoadInt32();
  if (depth_camera_type_int != static_cast<int>(Camera::Type::kPinholeCamera4f) ||
      depth_camera_parameter_count != 4) {
    LOG(ERROR) << "Unexpected depth camera type or parameter count.";
    fclose(file); return false;
  }
  float depth_camera_parameters[4];
  if (fread(depth_camera_parameters, sizeof(float), depth_camera_parameter_count, file) != depth_camera_parameter_count) {
    LOG(ERROR) << "Unexpected end of file.";
    fclose(file); return false;
  }
  ba.SetDepthCamera(PinholeCamera4f(depth_camera_width, depth_camera_height, depth_camera_parameters));
  
  int cfactor_buffer_width = LoadInt32();
  int cfactor_buffer_height = LoadInt32();
  if (cfactor_buffer_width != ba.cfactor_buffer()->width() ||
      cfactor_buffer_height != ba.cfactor_buffer()->height()) {
    LOG(ERROR) << "cfactor_buffer size does not match.";
    fclose(file); return false;
  }
  int cfactor_buffer_stride = LoadInt32();
  if (cfactor_buffer_stride > 100000 * sizeof(float)) {
    LOG(ERROR) << "Excessive cfactor_buffer stride, refusing to load.";
    fclose(file); return false;
  }
  Image<float> cfactor_cpu(cfactor_buffer_width, cfactor_buffer_height, cfactor_buffer_stride, 1);
  if (fread(cfactor_cpu.data(), 1, cfactor_cpu.height() * cfactor_cpu.stride(), file) != cfactor_cpu.height() * cfactor_cpu.stride()) {
    LOG(ERROR) << "Unexpected end of file.";
    fclose(file); return false;
  }
  ba.cfactor_buffer()->UploadAsync(0, cfactor_cpu);
  
  DepthParameters depth_params = ba.depth_params();
  depth_params.a = LoadFloat();
  depth_params.raw_to_float_depth = LoadFloat();
  depth_params.baseline_fx = LoadFloat();
  depth_params.sparse_surfel_cell_size = LoadInt32();
  ba.SetDepthParams(depth_params);
  
  // Load keyframes. Relevant parameters must be set beforehand (e.g., RGBDVideo poses).
  size = LoadInt32();
  if (size > slam->rgbd_video()->frame_count()) {
    LOG(ERROR) << "More keyframes than frames in the video.";
    fclose(file); return false;
  }
  // Force the new keyframes to get processed immediately.
  bool old_parallel_ba = slam->config().parallel_ba;
  bool old_parallel_loop_detection = slam->config().parallel_loop_detection;
  bool old_estimate_poses = slam->config().estimate_poses;
  slam->config().parallel_ba = false;
  slam->config().parallel_loop_detection = false;
  slam->config().estimate_poses = false;
  for (usize i = 0; i < size; ++ i) {
    if (progress_function) {
      if (!progress_function(i, size)) {
        // Aborted.
        ba.keyframes_mutable()->clear();
        fclose(file); return false;
      }
    }
    
    int keyframe_id = LoadInt32();
    if (keyframe_id < 0) {
      ba.keyframes_mutable()->push_back(nullptr);
    } else {
      if (keyframe_id != i) {
        LOG(ERROR) << "Unexpected keyframe id.";
        fclose(file); return false;
      }
      
      int frame_index = LoadInt32();
      int activation = LoadInt32();
      int last_active_in_ba_iteration = LoadInt32();
      int last_covis_in_ba_iteration = LoadInt32();
      
      // Create a keyframe with the loaded properties
      const Image<Vec3u8>* rgb_image =
          slam->rgbd_video()->color_frame_mutable(frame_index)->GetImage().get();
      
      shared_ptr<Image<u16>> final_cpu_depth_map;
      CUDABuffer<u16>* final_depth_buffer;
      slam->PreprocessFrame(
          frame_index,
          &final_depth_buffer,
          &final_cpu_depth_map);
      
      shared_ptr<Keyframe> new_keyframe = slam->CreateKeyframe(
          frame_index,
          rgb_image,
          final_cpu_depth_map,
          *final_depth_buffer);
      
      CHECK_EQ(new_keyframe->id(), keyframe_id);
      new_keyframe->SetActivation(static_cast<Keyframe::Activation>(activation));
      new_keyframe->SetLastActiveInBAIteration(last_active_in_ba_iteration);
      new_keyframe->SetLastCovisInBAIteration(last_covis_in_ba_iteration);
      
      slam->rgbd_video()->color_frame_mutable(frame_index)->ClearImageAndDerivedData();
      slam->rgbd_video()->depth_frame_mutable(frame_index)->ClearImageAndDerivedData();
    }
  }
  slam->config().parallel_ba = old_parallel_ba;
  slam->config().parallel_loop_detection = old_parallel_loop_detection;
  slam->config().estimate_poses = old_estimate_poses;
  
  int surfel_count = LoadInt32();
  int surfels_size = LoadInt32();
  if (surfel_count != surfels_size) {
    LOG(ERROR) << "surfel_count != surfels_size";
    fclose(file); return false;
  }
  if (surfel_count > slam->config().max_surfel_count) {
    LOG(ERROR) << "surfel_count > slam->config().max_surfel_count";
    fclose(file); return false;
  }
  if (surfel_count < 0) {
    LOG(ERROR) << "surfel_count < 0";
    fclose(file); return false;
  }
  ba.SetSurfelCount(surfel_count, surfels_size);
  CUDABufferPtr<float> surfels = ba.surfels();
  vector<float> surfel_data(surfels_size);
  for (int i = 0; i < kSurfelDataAttributeCount; ++ i) {
    if (fread(surfel_data.data(), sizeof(float), ba.surfels_size(), file) != ba.surfels_size()) {
      LOG(ERROR) << "Unexpected end of file.";
      fclose(file); return false;
    }
    surfels->UploadPartAsync(i * surfels->ToCUDA().pitch(), ba.surfels_size() * sizeof(float), 0, surfel_data.data());
  }
  
  ba.SetBAIterationCount(LoadInt32());
  ba.SetLastBAIterationCount(LoadInt32());
  
  ba.SetUseDepthResiduals(LoadBool());
  ba.SetUseDescriptorResiduals(LoadBool());
  
  ba.SetMinObservationCountWhileBootstrapping1(LoadInt32());
  ba.SetMinObservationCountWhileBootstrapping2(LoadInt32());
  ba.SetMinObservationCount(LoadInt32());
  
  ba.SetSurfelMergeDistFactor(LoadFloat());
  
  // Assign BadSlam::base_kf_.
  if (base_kf_id < 0) {
    slam->SetBaseKF(nullptr);
  } else if (base_kf_id < ba.keyframes().size()) {
    slam->SetBaseKF(ba.keyframes()[base_kf_id].get());
  } else {
    LOG(ERROR) << "Invalid base_kf_id.";
    fclose(file); return false;
  }
  
  // Load queued keyframes.
  usize num_queued_keyframes = queued_keyframes_frame_indices.size();
  vector<shared_ptr<Keyframe>> queued_keyframes(num_queued_keyframes);
  vector<cv::Mat_<u8>> queued_keyframe_gray_images(num_queued_keyframes);
  vector<shared_ptr<Image<u16>>> queued_keyframe_depth_images(num_queued_keyframes);
  for (usize i = 0; i < queued_keyframes_frame_indices.size(); ++ i) {
    int frame_index = queued_keyframes_frame_indices[i];
    
    const Image<Vec3u8>* rgb_image =
        slam->rgbd_video()->color_frame_mutable(frame_index)->GetImage().get();
    
    shared_ptr<Image<u16>> final_cpu_depth_map;
    CUDABuffer<u16>* final_depth_buffer;
    slam->PreprocessFrame(
        frame_index,
        &final_depth_buffer,
        &final_cpu_depth_map);
    
    queued_keyframes[i] = slam->CreateKeyframe(
        frame_index,
        rgb_image,
        final_cpu_depth_map,
        *final_depth_buffer);
    
    queued_keyframe_gray_images[i] = slam->CreateGrayImageForLoopDetection(*rgb_image);
    queued_keyframe_depth_images[i] = final_cpu_depth_map;
    
    slam->rgbd_video()->color_frame_mutable(frame_index)->ClearImageAndDerivedData();
    slam->rgbd_video()->depth_frame_mutable(frame_index)->ClearImageAndDerivedData();
  }
  slam->SetQueuedKeyframes(
      queued_keyframes,
      queued_keyframes_last_kf_tr_this_kf,
      queued_keyframe_gray_images,
      queued_keyframe_depth_images);
  
  fclose(file);
  return true;
}

bool SavePoses(
    const RGBDVideo<Vec3u8, u16>& rgbd_video,
    bool use_depth_timestamps,
    int start_frame,
    const std::string& export_poses_path) {
  SE3f start_frame_T_global = rgbd_video.depth_frame(start_frame)->frame_T_global();
  
  std::ofstream poses_file(export_poses_path, std::ios::out);
  if (!poses_file) {
    return false;
  }
  poses_file << std::setprecision(numeric_limits<double>::digits10 + 1);
  
  poses_file << "# Format: Each line gives one global_T_frame pose with values: tx ty tz qx qy qz qw" << std::endl;
  
  for (usize frame_index = 0; frame_index < rgbd_video.frame_count(); ++ frame_index) {
    SE3f global_T_frame = start_frame_T_global * rgbd_video.depth_frame(frame_index)->global_T_frame();
    
    poses_file << (use_depth_timestamps ? rgbd_video.depth_frame(frame_index)->timestamp_string() :
                                          rgbd_video.color_frame(frame_index)->timestamp_string()) << " "
               << global_T_frame.translation().x() << " "
               << global_T_frame.translation().y() << " "
               << global_T_frame.translation().z() << " "
               << global_T_frame.unit_quaternion().x() << " "
               << global_T_frame.unit_quaternion().y() << " "
               << global_T_frame.unit_quaternion().z() << " "
               << global_T_frame.unit_quaternion().w() << std::endl;
  }
  
  poses_file.close();
  return true;
}

bool SaveCalibration(
    cudaStream_t stream,
    const DirectBA& direct_ba,
    const string& export_base_path) {
  {
    std::string intrinsics_path = export_base_path + ".depth_intrinsics.txt";
    std::ofstream calibration_file(intrinsics_path, std::ios::out);
    if (!calibration_file) {
      return false;
    }
    calibration_file << direct_ba.depth_camera().parameters()[0] << " "
                     << direct_ba.depth_camera().parameters()[1] << " "
                     << (direct_ba.depth_camera().parameters()[2] - 0.5) << " "
                     << (direct_ba.depth_camera().parameters()[3] - 0.5);
    calibration_file.close();
    LOG(INFO) << "Wrote depth intrinsics to: " << intrinsics_path;
  }
  
  {
    std::string intrinsics_path = export_base_path + ".color_intrinsics.txt";
    std::ofstream calibration_file(intrinsics_path, std::ios::out);
    if (!calibration_file) {
      return false;
    }
    calibration_file << direct_ba.color_camera().parameters()[0] << " "
                     << direct_ba.color_camera().parameters()[1] << " "
                     << (direct_ba.color_camera().parameters()[2] - 0.5) << " "
                     << (direct_ba.color_camera().parameters()[3] - 0.5);
    calibration_file.close();
    LOG(INFO) << "Wrote color intrinsics to: " << intrinsics_path;
  }
  
  const CUDABufferConstPtr<float>& cfactor_buffer = direct_ba.cfactor_buffer();
  
  std::string deformation_path = export_base_path + ".deformation.txt";
  std::ofstream deformation_file(deformation_path, std::ios::out);
  if (!deformation_file) {
    return false;
  }
  deformation_file.precision(8);
  deformation_file << cfactor_buffer->width() << " " << cfactor_buffer->height() << std::endl;
  deformation_file << direct_ba.a() << std::endl;
  Image<float> cfactor_buffer_cpu(cfactor_buffer->width(), cfactor_buffer->height());
  cfactor_buffer->DownloadAsync(stream, &cfactor_buffer_cpu);
  cudaStreamSynchronize(stream);
  for (u32 y = 0; y < cfactor_buffer_cpu.height(); ++ y) {
    for (u32 x = 0; x < cfactor_buffer_cpu.width(); ++ x) {
      deformation_file << cfactor_buffer_cpu(x, y) << std::endl;
    }
  }
  deformation_file.close();
  LOG(INFO) << "Wrote deformation calibration to: " << deformation_path;
  return true;
}


bool LoadCalibration(
    DirectBA* direct_ba,
    const string& import_base_path) {
  {
    std::string intrinsics_path = import_base_path + ".depth_intrinsics.txt";
    std::ifstream calibration_file(intrinsics_path, std::ios::in);
    if (!calibration_file) {
      LOG(ERROR) << "Cannot read file: " << intrinsics_path;
      return false;
    }
    float intrinsics[4];
    calibration_file >> intrinsics[0] >> intrinsics[1] >> intrinsics[2] >> intrinsics[3];
    calibration_file.close();
    
    intrinsics[2] += 0.5;
    intrinsics[3] += 0.5;
    direct_ba->depth_camera() = PinholeCamera4f(
        direct_ba->depth_camera().width(),
        direct_ba->depth_camera().height(),
        intrinsics);
  }
  
  {
    std::string intrinsics_path = import_base_path + ".color_intrinsics.txt";
    std::ifstream calibration_file(intrinsics_path, std::ios::in);
    if (!calibration_file) {
      LOG(ERROR) << "Cannot read file: " << intrinsics_path;
      return false;
    }
    float intrinsics[4];
    calibration_file >> intrinsics[0] >> intrinsics[1] >> intrinsics[2] >> intrinsics[3];
    calibration_file.close();
    
    intrinsics[2] += 0.5;
    intrinsics[3] += 0.5;
    direct_ba->color_camera() = PinholeCamera4f(
        direct_ba->color_camera().width(),
        direct_ba->color_camera().height(),
        intrinsics);
  }
  
  std::string deformation_path = import_base_path + ".deformation.txt";
  std::ifstream deformation_file(deformation_path, std::ios::in);
  if (!deformation_file) {
    LOG(ERROR) << "Cannot read file: " << deformation_path;
    return false;
  }
  int cfactor_buffer_width, cfactor_buffer_height;
  deformation_file >> cfactor_buffer_width >> cfactor_buffer_height;
  CUDABufferPtr<float> cfactor_buffer = direct_ba->cfactor_buffer();
  if (cfactor_buffer_width != cfactor_buffer->width() ||
      cfactor_buffer_height != cfactor_buffer->height()) {
    LOG(ERROR) << "cfactor buffer size mismatch in current configuration vs. imported deformation - need to implement rescaling";
    return false;
  }
  deformation_file >> direct_ba->a();
  Image<float> cfactor_buffer_cpu(cfactor_buffer->width(), cfactor_buffer->height());
  for (u32 y = 0; y < cfactor_buffer_cpu.height(); ++ y) {
    for (u32 x = 0; x < cfactor_buffer_cpu.width(); ++ x) {
      deformation_file >> cfactor_buffer_cpu(x, y);
    }
  }
  cfactor_buffer->UploadAsync(/*stream*/ 0, cfactor_buffer_cpu);
  deformation_file.close();
  
  return true;
}

bool SavePointCloudAsPLY(
    cudaStream_t stream,
    const DirectBA& direct_ba,
    const string& export_path) {
  Point3fC3u8NfCloud cloud;
  direct_ba.ExportToPointCloud(stream, &cloud);
  bool result = cloud.WriteAsPLY(export_path);
  LOG(INFO) << "Wrote point cloud to: " << export_path;
  return result;
}

}
