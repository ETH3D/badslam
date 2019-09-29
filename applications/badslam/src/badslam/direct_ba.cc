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

#ifdef WIN32
// Just to have M_PI_2 defined
// See https://docs.microsoft.com/en-us/cpp/c-runtime-library/math-constants?view=vs-2019
#define _USE_MATH_DEFINES // for C++
#include <cmath> 
#endif

#include "badslam/direct_ba.h"

#include <algorithm>

#include <libvis/camera_frustum.h>
#include <libvis/image_display.h>
#include <libvis/timing.h>

#include "badslam/bad_slam.h"
#include "badslam/convergence_analysis.h"
#include "badslam/util.cuh"
#include "badslam/loop_detector.h"
#include "badslam/pose_graph_optimizer.h"
#include "badslam/robust_weighting.cuh"
#include "badslam/util.h"


namespace vis {

constexpr bool kDebugVerifySurfelCount = false;


struct MergeKeyframeDistance {
  MergeKeyframeDistance(float distance, u32 prev_keyframe_id, u32 keyframe_id, u32 next_keyframe_id)
      : distance(distance),
        prev_keyframe_id(prev_keyframe_id),
        keyframe_id(keyframe_id),
        next_keyframe_id(next_keyframe_id) {}
  
  bool operator< (const MergeKeyframeDistance& other) const {
    return distance < other.distance;
  }
  
  float distance;
  u32 prev_keyframe_id;
  u32 keyframe_id;
  u32 next_keyframe_id;
};


DirectBA::DirectBA(
    int max_surfel_count,
    float raw_to_float_depth,
    float baseline_fx,
    int sparse_surfel_cell_size,
    float surfel_merge_dist_factor,
    int min_observation_count_while_bootstrapping_1,
    int min_observation_count_while_bootstrapping_2,
    int min_observation_count,
    const PinholeCamera4f& color_camera_initial_estimate,
    const PinholeCamera4f& depth_camera_initial_estimate,
    int pyramid_level_for_color,
    bool use_depth_residuals,
    bool use_descriptor_residuals,
    shared_ptr<BadSlamRenderWindow> render_window,
    const SE3f& global_T_anchor_frame)
    : color_camera_(color_camera_initial_estimate),
      pyramid_level_for_color_(pyramid_level_for_color),
      depth_camera_(depth_camera_initial_estimate),
      use_depth_residuals_(use_depth_residuals),
      use_descriptor_residuals_(use_descriptor_residuals),
      min_observation_count_while_bootstrapping_1_(min_observation_count_while_bootstrapping_1),
      min_observation_count_while_bootstrapping_2_(min_observation_count_while_bootstrapping_2),
      min_observation_count_(min_observation_count),
      surfel_merge_dist_factor_(surfel_merge_dist_factor),
      intrinsics_optimization_helper_buffers_(
          /*pixel_count*/ depth_camera_.width() * depth_camera_.height(),
          /*sparse_pixel_count*/ ((depth_camera_.width() - 1) / sparse_surfel_cell_size + 1) *
                                 ((depth_camera_.height() - 1) / sparse_surfel_cell_size + 1),
          /*a_rows*/ 4 + 1),
      render_window_(render_window),
      global_T_anchor_frame_(global_T_anchor_frame) {
  depth_params_.a = 0;
  cfactor_buffer_.reset(new CUDABuffer<float>(
      (depth_camera_.height() - 1) / sparse_surfel_cell_size + 1,
      (depth_camera_.width() - 1) / sparse_surfel_cell_size + 1));
  cfactor_buffer_->Clear(0, /*stream*/ 0);
  cudaDeviceSynchronize();
  
  depth_params_.cfactor_buffer = cfactor_buffer_->ToCUDA();
  depth_params_.raw_to_float_depth = raw_to_float_depth;
  depth_params_.baseline_fx = baseline_fx;
  depth_params_.sparse_surfel_cell_size = sparse_surfel_cell_size;
  
  surfels_size_ = 0;
  surfel_count_ = 0;
  surfels_.reset(new CUDABuffer<float>(kSurfelAttributeCount, max_surfel_count));
  active_surfels_.reset(new CUDABuffer<u8>(1, max_surfel_count));
  
  ba_iteration_count_ = 0;
  last_ba_iteration_count_ = -1;
  
  new_surfels_temp_storage_ = nullptr;
  new_surfels_temp_storage_bytes_ = 0;
  free_spots_temp_storage_ = nullptr;
  free_spots_temp_storage_bytes_ = 0;
  new_surfel_flag_vector_.reset(new CUDABuffer<u8>(1, depth_camera_.height() * depth_camera_.width()));
  new_surfel_indices_.reset(new CUDABuffer<u32>(1, depth_camera_.height() * depth_camera_.width()));
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels_[i].reset(new CUDABuffer<u32>(depth_camera_.height(), depth_camera_.width()));
  }
  
  if (gather_convergence_samples_) {
    convergence_samples_file_.open("/media/thomas/Daten/convergence_samples.txt", std::ios::out);
  }
  
  timings_stream_ = nullptr;
  
  cudaEventCreate(&ba_surfel_creation_pre_event_);
  cudaEventCreate(&ba_surfel_creation_post_event_);
  cudaEventCreate(&ba_surfel_activation_pre_event_);
  cudaEventCreate(&ba_surfel_activation_post_event_);
  cudaEventCreate(&ba_surfel_compaction_pre_event_);
  cudaEventCreate(&ba_surfel_compaction_post_event_);
  cudaEventCreate(&ba_geometry_optimization_pre_event_);
  cudaEventCreate(&ba_geometry_optimization_post_event_);
  cudaEventCreate(&ba_surfel_merge_pre_event_);
  cudaEventCreate(&ba_surfel_merge_post_event_);
  cudaEventCreate(&ba_pose_optimization_pre_event_);
  cudaEventCreate(&ba_pose_optimization_post_event_);
  cudaEventCreate(&ba_intrinsics_optimization_pre_event_);
  cudaEventCreate(&ba_intrinsics_optimization_post_event_);
  cudaEventCreate(&ba_final_surfel_deletion_and_radius_update_pre_event_);
  cudaEventCreate(&ba_final_surfel_deletion_and_radius_update_post_event_);
  cudaEventCreate(&ba_final_surfel_merge_pre_event_);
  cudaEventCreate(&ba_final_surfel_merge_post_event_);
  cudaEventCreate(&ba_pcg_pre_event_);
  cudaEventCreate(&ba_pcg_post_event_);
}

DirectBA::~DirectBA() {
  cudaEventDestroy(ba_surfel_creation_pre_event_);
  cudaEventDestroy(ba_surfel_creation_post_event_);
  cudaEventDestroy(ba_surfel_activation_pre_event_);
  cudaEventDestroy(ba_surfel_activation_post_event_);
  cudaEventDestroy(ba_surfel_compaction_pre_event_);
  cudaEventDestroy(ba_surfel_compaction_post_event_);
  cudaEventDestroy(ba_geometry_optimization_pre_event_);
  cudaEventDestroy(ba_geometry_optimization_post_event_);
  cudaEventDestroy(ba_surfel_merge_pre_event_);
  cudaEventDestroy(ba_surfel_merge_post_event_);
  cudaEventDestroy(ba_pose_optimization_pre_event_);
  cudaEventDestroy(ba_pose_optimization_post_event_);
  cudaEventDestroy(ba_intrinsics_optimization_pre_event_);
  cudaEventDestroy(ba_intrinsics_optimization_post_event_);
  cudaEventDestroy(ba_final_surfel_deletion_and_radius_update_pre_event_);
  cudaEventDestroy(ba_final_surfel_deletion_and_radius_update_post_event_);
  cudaEventDestroy(ba_final_surfel_merge_pre_event_);
  cudaEventDestroy(ba_final_surfel_merge_post_event_);
  cudaEventDestroy(ba_pcg_pre_event_);
  cudaEventDestroy(ba_pcg_post_event_);
  
  if (new_surfels_temp_storage_bytes_ > 0) {
    cudaFree(new_surfels_temp_storage_);
  }
  
  if (gather_convergence_samples_) {
    convergence_samples_file_.close();
  }
}

void DirectBA::AddKeyframe(
    const shared_ptr<Keyframe>& new_keyframe) {
  int id = static_cast<int>(keyframes_.size());
  new_keyframe->SetID(id);
  
  DetermineNewKeyframeCoVisibility(new_keyframe);
  
  keyframes_.push_back(new_keyframe);
}

void DirectBA::DeleteKeyframe(
    int keyframe_index,
    LoopDetector* loop_detector) {
  // TODO: Re-use the deleted keyframe's buffers for new keyframes, since
  //       CUDA memory allocation is very slow.
  shared_ptr<Keyframe> frame_to_delete = keyframes_[keyframe_index];
  for (u32 covis_keyframe_index : frame_to_delete->co_visibility_list()) {
    Keyframe* covis_frame = keyframes_[covis_keyframe_index].get();
    
    for (usize i = 0, end = covis_frame->co_visibility_list().size(); i < end; ++ i) {
      if (covis_frame->co_visibility_list()[i] == static_cast<int>(keyframe_index)) {
        covis_frame->co_visibility_list().erase(covis_frame->co_visibility_list().begin() + i);
        break;
      }
    }
  }
  
  keyframes_[keyframe_index].reset();
  
  if (loop_detector) {
    loop_detector->RemoveImage(keyframe_index);
  }
}

void DirectBA::DetermineNewKeyframeCoVisibility(const shared_ptr<Keyframe>& new_keyframe) {
  // Update the co-visibility lists and set the other frame to co-visible active.
  CameraFrustum new_frustum(depth_camera_, new_keyframe->min_depth(), new_keyframe->max_depth(), new_keyframe->global_T_frame());
  for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
    if (!keyframe) {
      continue;
    }
    CameraFrustum keyframe_frustum(depth_camera_, keyframe->min_depth(), keyframe->max_depth(), keyframe->global_T_frame());
    
    if (new_frustum.Intersects(&keyframe_frustum)) {
      new_keyframe->co_visibility_list().push_back(keyframe->id());
      keyframe->co_visibility_list().push_back(new_keyframe->id());
      
      if (keyframe->activation() == Keyframe::Activation::kInactive) {
        keyframe->SetActivation(Keyframe::Activation::kCovisibleActive);
      }
    }
  }
}

void DirectBA::MergeKeyframes(
    cudaStream_t /*stream*/,
    LoopDetector* loop_detector,
    usize approx_merge_count) {
  // TODO: Make parameters:
  constexpr float kMaxAngleDifference = 0.5f * M_PI_2;
  constexpr float kMaxEuclideanDistance = 0.3f;
  
  if (keyframes_.size() <= 1) {
    return;
  }
  
  vector<MergeKeyframeDistance> distances;
  distances.reserve(keyframes_.size() - 1);
  
  float prev_half_distance = 0;
  usize prev_keyframe_id = 0;
  
  for (usize keyframe_id = 0; keyframe_id < keyframes_.size() - 1; ++ keyframe_id) {
    const shared_ptr<Keyframe>& keyframe = keyframes_[keyframe_id];
    if (!keyframe) {
      continue;
    }
    const Keyframe* next_keyframe = nullptr;
    for (usize next_id = keyframe_id + 1; next_id < keyframes_.size(); ++ next_id) {
      if (keyframes_[next_id]) {
        next_keyframe = keyframes_[next_id].get();
        break;
      }
    }
    if (!next_keyframe) {
      break;
    }
    
    float angle_difference = acosf(keyframe->global_T_frame().rotationMatrix().block<3, 1>(0, 2).dot(
                                       next_keyframe->global_T_frame().rotationMatrix().block<3, 1>(0, 2)));
    if (angle_difference > kMaxAngleDifference) {
      continue;
    }
    
    float euclidean_distance = (keyframe->global_T_frame().translation() - next_keyframe->global_T_frame().translation()).norm();
    if (euclidean_distance > kMaxEuclideanDistance) {
      continue;
    }
    
    // Weighting: 90 degree angle difference count like half a meter distance
    float next_half_distance = euclidean_distance + (0.5f / M_PI_2) * angle_difference;
    // NOTE: Never delete the first keyframe (with index 0) since it is the
    //       anchor for the reconstruction.
    if (keyframe_id > 0) {
      distances.emplace_back(prev_half_distance + next_half_distance, prev_keyframe_id, keyframe_id, next_keyframe->id());
    }
    prev_half_distance = next_half_distance;
    prev_keyframe_id = keyframe_id;
    
    // TODO: Idea for additional criteria:
    //       Maybe try to compute whether the co-vis frames cover all of one of
    //       the frames' frustum (such that no geometry is lost in the merge)?
  }
  
  usize number_of_sorted_distances = std::min(approx_merge_count, distances.size());
  std::partial_sort(distances.begin(), distances.begin() + number_of_sorted_distances, distances.end());
  
  if (loop_detector) {
    loop_detector->LockDetectorMutex();
  }
  
  for (usize i = 0; i < number_of_sorted_distances; ++ i) {
    const MergeKeyframeDistance& merge = distances[i];
    if (!keyframes_[merge.prev_keyframe_id] || !keyframes_[merge.keyframe_id] || !keyframes_[merge.next_keyframe_id]) {
      // One of the keyframes has been deleted by a previous merge.
      // Since we only do an approximate number of merges, simply ignore this
      // merge entry (instead of updating the distance).
      continue;
    }
    
    // TODO: Actually merge the frame into the other (and possibly other
    //       frames with co-visibility). At the moment, the frame is simply
    //       deleted.
    DeleteKeyframe(merge.keyframe_id, loop_detector);
    
    LOG(ERROR) << "Deleted keyframe with ID " << merge.keyframe_id;
  }
  
  if (loop_detector) {
    loop_detector->UnlockDetectorMutex();
  }
}

void DirectBA::CreateSurfelsForKeyframe(
    cudaStream_t stream,
    bool filter_new_surfels,
    const shared_ptr<Keyframe>& keyframe) {
  CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels[i] = supporting_surfels_[i].get();
  }
  
  DetermineSupportingSurfelsCUDA(
      stream,
      depth_camera_,
      keyframe->frame_T_global_cuda(),
      depth_params_,
      keyframe->depth_buffer(),
      keyframe->normals_buffer(),
      surfels_size_,
      surfels_.get(),
      supporting_surfels);
  
  // Prepare relative transformations outside of the .cu file since doing it
  // within the file gave wrong results on my laptop (but it worked on my
  // desktop PC).
  // TODO: This can probably be reverted once it is ensured that all compiler
  //       versions and settings are equal
  vector<CUDAMatrix3x4> covis_T_frame(keyframe->co_visibility_list().size());
  for (usize i = 0; i < keyframe->co_visibility_list().size(); ++ i) {
    int co_visible_keyframe_index = keyframe->co_visibility_list()[i];
    const shared_ptr<Keyframe>& co_visible_keyframe = keyframes_[co_visible_keyframe_index];
    covis_T_frame[i] = CUDAMatrix3x4((co_visible_keyframe->frame_T_global() * keyframe->global_T_frame()).matrix3x4());
  }
  
  u32 new_surfel_count;
  CreateSurfelsForKeyframeCUDA(
      stream,
      depth_params_.sparse_surfel_cell_size,
      filter_new_surfels,
      GetMinObservationCount(),
      keyframe->id(),
      keyframes_,
      color_camera_,
      depth_camera_,
      CUDAMatrix3x4(keyframe->global_T_frame().matrix3x4()),
      keyframe->frame_T_global_cuda(),
      covis_T_frame,
      depth_params_,
      keyframe->depth_buffer(),
      keyframe->normals_buffer(),
      keyframe->radius_buffer(),
      keyframe->color_buffer(),
      keyframe->color_texture(),
      supporting_surfels,
      &new_surfels_temp_storage_,
      &new_surfels_temp_storage_bytes_,
      new_surfel_flag_vector_.get(),
      new_surfel_indices_.get(),
      surfels_size_,
      surfel_count_,
      &new_surfel_count,
      surfels_.get());
  
  Lock();
  surfels_size_ += new_surfel_count;
  surfel_count_ += new_surfel_count;
  Unlock();
}

void DirectBA::BundleAdjustment(
      cudaStream_t stream,
      bool optimize_depth_intrinsics,
      bool optimize_color_intrinsics,
      bool do_surfel_updates,
      bool optimize_poses,
      bool optimize_geometry,
      int min_iterations,
      int max_iterations,
      bool use_pcg,
      int active_keyframe_window_start,
      int active_keyframe_window_end,
      bool increase_ba_iteration_count,
      int* iterations_done,
      bool* converged,
      double time_limit,
      Timer* timer,
      int pcg_max_inner_iterations,
      int pcg_max_keyframes,
      std::function<bool (int)> progress_function) {
  if (optimize_depth_intrinsics && !use_depth_residuals_) {
    LOG(WARNING) << "optimize_depth_intrinsics set to true, but use_depth_residuals_ set to false. Depth intrinsics will not be optimized.";
    optimize_depth_intrinsics = false;
  }
  if (optimize_color_intrinsics && !use_descriptor_residuals_) {
    LOG(WARNING) << "optimize_color_intrinsics set to true, but use_descriptor_residuals_ set to false. Color intrinsics will not be optimized.";
    optimize_color_intrinsics = false;
  }
  
  if (use_pcg) {
    BundleAdjustmentPCG(
        stream, optimize_depth_intrinsics, optimize_color_intrinsics,
        do_surfel_updates, optimize_poses, optimize_geometry,
        min_iterations, max_iterations,
        pcg_max_inner_iterations, pcg_max_keyframes,
        active_keyframe_window_start, active_keyframe_window_end,
        increase_ba_iteration_count, iterations_done, converged,
        time_limit, timer, progress_function);
  } else {
    BundleAdjustmentAlternating(
        stream, optimize_depth_intrinsics, optimize_color_intrinsics,
        do_surfel_updates, optimize_poses, optimize_geometry,
        min_iterations, max_iterations,
        active_keyframe_window_start, active_keyframe_window_end,
        increase_ba_iteration_count, iterations_done, converged,
        time_limit, timer, progress_function);
  }
}

void DirectBA::AssignColors(
    cudaStream_t stream) {
  AssignColorsCUDA(stream, color_camera_, depth_camera_, depth_params_, keyframes_, surfels_size_, surfels_.get());
}

void DirectBA::ExportToPointCloud(
    cudaStream_t stream,
    Point3fC3u8NfCloud* cloud) const {
  cloud->Resize(surfel_count_);
  
  // Download surfel x and determine valid surfels.
  vector<bool> is_valid(surfels_size_);
  vector<float> buffer(surfels_size_);
  surfels_->DownloadPartAsync(kSurfelX * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
  cudaStreamSynchronize(stream);
  usize index = 0;
  for (usize i = 0; i < surfels_size_; ++ i) {
    if (std::isnan(buffer[i])) {
      is_valid[i] = false;
      continue;
    }
    
    if (index >= surfel_count_) {
      LOG(ERROR) << "surfel_count_ is not consistent with the actual number of valid surfels! Skipping the remaining surfels.";
      return;
    }
    
    is_valid[i] = true;
    cloud->at(index).position().x() = buffer[i];
    ++ index;
  }
  
  if (index != surfel_count_) {
    LOG(ERROR) << "surfel_count_ (" << surfel_count_ << ") is not consistent with the actual number of valid surfels (" << index << ")!";
  }
  
  // Download surfel y.
  surfels_->DownloadPartAsync(kSurfelY * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
  cudaStreamSynchronize(stream);
  index = 0;
  for (usize i = 0; i < surfels_size_; ++ i) {
    if (is_valid[i]) {
      cloud->at(index).position().y() = buffer[i];
      ++ index;
    }
  }
  
  // Download surfel z.
  surfels_->DownloadPartAsync(kSurfelZ * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
  cudaStreamSynchronize(stream);
  index = 0;
  for (usize i = 0; i < surfels_size_; ++ i) {
    if (is_valid[i]) {
      cloud->at(index).position().z() = buffer[i];
      ++ index;
    }
  }
  
  // Download surfel color.
  surfels_->DownloadPartAsync(kSurfelColor * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
  cudaStreamSynchronize(stream);
  index = 0;
  for (usize i = 0; i < surfels_size_; ++ i) {
    if (is_valid[i]) {
      const uchar4& color = reinterpret_cast<const uchar4&>(buffer[i]);
      cloud->at(index).color().x() = color.x;
      cloud->at(index).color().y() = color.y;
      cloud->at(index).color().z() = color.z;
      ++ index;
    }
  }
  
  // Download surfel normals.
  surfels_->DownloadPartAsync(kSurfelNormal * surfels_->ToCUDA().pitch(), surfels_size_ * sizeof(float), stream, buffer.data());
  cudaStreamSynchronize(stream);
  index = 0;
  for (usize i = 0; i < surfels_size_; ++ i) {
    if (is_valid[i]) {
      u32 value = *reinterpret_cast<const u32*>(&buffer[i]);
      float3 normal = make_float3(
          TenBitSignedToFloat(value >> 0),
          TenBitSignedToFloat(value >> 10),
          TenBitSignedToFloat(value >> 20));
      float factor = 1.0f / sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
      
      cloud->at(index).normal().x() = factor * normal.x;
      cloud->at(index).normal().y() = factor * normal.y;
      cloud->at(index).normal().z() = factor * normal.z;
      ++ index;
    }
  }
}

void DirectBA::DetermineCovisibleActiveKeyframes() {
  for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
    if (!keyframe) {
      continue;
    }
    
    if (keyframe->activation() == Keyframe::Activation::kActive) {
      for (int covisible_index : keyframe->co_visibility_list()) {
        shared_ptr<Keyframe>& other_keyframe = keyframes_[covisible_index];
        if (other_keyframe->activation() == Keyframe::Activation::kInactive) {
          other_keyframe->SetActivation(Keyframe::Activation::kCovisibleActive);
        }
      }
    }
  }
}

void DirectBA::PerformBASchemeEndTasks(
    cudaStream_t stream,
    bool do_surfel_updates) {
  u32 surfel_count = surfel_count_;
  u32 surfels_size = surfels_size_;
  
  CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels[i] = supporting_surfels_[i].get();
  }
  
  // Merge similar surfels using all keyframes which were active.
  if (do_surfel_updates) {
    cudaEventRecord(ba_final_surfel_merge_pre_event_, stream);
    for (shared_ptr<Keyframe>& keyframe : keyframes_) {
      if (!keyframe) {
        continue;
      }
      
      if (keyframe->last_active_in_ba_iteration() == ba_iteration_count_) {
        DetermineSupportingSurfelsAndMergeSurfelsCUDA(
            stream,
            surfel_merge_dist_factor_,
            depth_camera_,
            keyframe->frame_T_global_cuda(),
            depth_params_,
            keyframe->depth_buffer(),
            keyframe->normals_buffer(),
            surfels_size,
            surfels_.get(),
            supporting_surfels,
            &surfel_count,
            &deleted_count_buffer_);
      }
    }
    cudaEventRecord(ba_final_surfel_merge_post_event_, stream);
    
    if (kDebugVerifySurfelCount) {
      DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
    }
  }
  
  
  // Delete surfels which are outliers or not sufficiently observed, and compact surfels.
  // TODO: It would be good if this could be limited to active surfels, but this
  //       left a few outliers. Maybe the reason was that covis-frames can also
  //       move, but not all of their observed surfels are set to active. Thus,
  //       it is possible that an inactive surfel becomes unobserved. In this
  //       case, limiting this check to active surfels will overlook the surfel.
  cudaEventRecord(ba_final_surfel_deletion_and_radius_update_pre_event_, stream);
  DeleteSurfelsAndUpdateRadiiCUDA(stream, GetMinObservationCount(), depth_camera_, depth_params_, keyframes_, &surfel_count, surfels_size, surfels_.get(), &deleted_count_buffer_);
  if (kDebugVerifySurfelCount) {
    DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
  }
  CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count, &surfels_size, &surfels_->ToCUDA());
  cudaEventRecord(ba_final_surfel_deletion_and_radius_update_post_event_, stream);
  
  if (kDebugVerifySurfelCount) {
    DebugVerifySurfelCount(stream, surfel_count, surfels_size, *surfels_);
  }
  
  Lock();
  surfels_size_ = surfels_size;
  surfel_count_ = surfel_count;
  Unlock();
  
  
  //LOG(INFO) << "--> final surfel_count: " << surfel_count_;  // << "  (surfels_size: " << surfels_size_ << ")";
  
  
  // Store timings for events used outside the optimization loop.
  cudaEventSynchronize(ba_final_surfel_deletion_and_radius_update_post_event_);
  float elapsed_milliseconds;
  
  cudaEventElapsedTime(&elapsed_milliseconds, ba_final_surfel_deletion_and_radius_update_pre_event_, ba_final_surfel_deletion_and_radius_update_post_event_);
  Timing::addTime(Timing::getHandle("BA final surfel del. and radius upd."), 0.001 * elapsed_milliseconds);
  if (timings_stream_) {
    *timings_stream_ << "BA_final_surfel_deletion_and_radius_update " << elapsed_milliseconds << endl;
  }
  
  if (do_surfel_updates) {
    cudaEventElapsedTime(&elapsed_milliseconds, ba_final_surfel_merge_pre_event_, ba_final_surfel_merge_post_event_);
    Timing::addTime(Timing::getHandle("BA final surfel merge and compact"), 0.001 * elapsed_milliseconds);
    if (timings_stream_) {
      *timings_stream_ << "BA_final_surfel_merge_and_compaction " << elapsed_milliseconds << endl;
    }
  }
}

void DirectBA::UpdateBAVisualization(cudaStream_t stream) {
  if (!render_window_) {
    return;
  }
  
  unique_lock<mutex> render_mutex_lock(render_window_->render_mutex());
  
  Lock();
  
  AssignColors(stream);
  
  SE3f anchor_pose_correction;
  if (!keyframes_.empty()) {
    anchor_pose_correction =
        global_T_anchor_frame_ *
        keyframes_[0]->frame_T_global();
  }
  
  UpdateVisualizationBuffersCUDA(
      stream,
      render_window_->surfel_vertices(),
      surfels_size_,
      surfels_->ToCUDA(),
      visualize_normals_,
      visualize_descriptors_,
      visualize_radii_);
  render_window_->UpdateVisualizationCloudCUDA(surfels_size_);
  
  render_window_->SetPoseCorrectionNoLock(anchor_pose_correction);
  
  vector<Mat4f> keyframe_poses;
  vector<int> keyframe_ids;
  
  keyframe_poses.reserve(keyframes_.size());
  keyframe_ids.reserve(keyframes_.size());
  
  for (usize i = 0; i < keyframes_.size(); ++ i) {
    if (!keyframes_[i]) {
      continue;
    }
    keyframe_poses.push_back(keyframes_[i]->global_T_frame().matrix());
    keyframe_ids.push_back(keyframes_[i]->id());
  }
  
  render_window_->SetKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ids));
  
  Unlock();
  
  cudaStreamSynchronize(stream);
  
  render_mutex_lock.unlock();
  
  render_window_->RenderFrame();
}

void DirectBA::UpdateKeyframeCoVisibility(const shared_ptr<Keyframe>& keyframe) {
  // Erase this keyframe from the other frames' covisibility lists.
  for (u32 covis_keyframe_index : keyframe->co_visibility_list()) {
    Keyframe* covis_frame = keyframes_[covis_keyframe_index].get();
    
    for (usize i = 0, end = covis_frame->co_visibility_list().size(); i < end; ++ i) {
      if (covis_frame->co_visibility_list()[i] == keyframe->id()) {
        covis_frame->co_visibility_list().erase(covis_frame->co_visibility_list().begin() + i);
        break;
      }
    }
  }
  
  keyframe->co_visibility_list().clear();
  
  // Find the current set of covisible frames.
  CameraFrustum frustum(depth_camera_, keyframe->min_depth(), keyframe->max_depth(), keyframe->global_T_frame());
  for (const shared_ptr<Keyframe>& other_keyframe : keyframes_) {
    if (!other_keyframe) {
      continue;
    }
    CameraFrustum other_frustum(depth_camera_, other_keyframe->min_depth(), other_keyframe->max_depth(), other_keyframe->global_T_frame());
    
    if (frustum.Intersects(&other_frustum)) {
      keyframe->co_visibility_list().push_back(other_keyframe->id());
      other_keyframe->co_visibility_list().push_back(keyframe->id());
    }
  }
}

}
