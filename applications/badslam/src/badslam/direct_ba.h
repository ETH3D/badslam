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


#pragma once

#include <atomic>
#include <functional>
#include <fstream>
#include <mutex>
#include <thread>

#include <cuda_runtime.h>
#include <libvis/libvis.h>
#include <libvis/point_cloud.h>
#include <libvis/sophus.h>

#include "badslam/kernels.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"

// #define DEBUG_LOCKING

namespace vis {

class BadSlam;
class BadSlamRenderWindow;
template <typename T> class CUDABuffer;
class LoopDetector;
class OpenGLContext;
template <typename ColorT, typename DepthT> class RGBDVideo;
class Timer;

// Direct bundle adjustment class, forming the SLAM back-end. Stores the scene
// model consisting of keyframes and surfels. May perform various operations on
// the scene. Is agnostic to the fact that the initial input is a video in BAD
// SLAM. The DirectBA class could be used as library API for the pure direct
// bundle adjustment (without any SLAM front-end functionality). For SLAM
// functionality, see the BadSlam class.
class DirectBA {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // Constructs a DirectBA object, setting various configuration options
  // (see bad_slam_config.h for descriptions). The render window may be null.
  // global_T_anchor_frame controls the coordinate frame of the visualization
  // and is ignored if the render window is null.
  DirectBA(
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
      const SE3f& global_T_anchor_frame);
  
  // Destructor.
  ~DirectBA();
  
  // Adds a new keyframe to the model. In parallel-BA mode, the keyframe is
  // first queued and later added by the BA thread.
  void AddKeyframe(
      const shared_ptr<Keyframe>& new_keyframe);
  
  // Deletes a keyframe.
  void DeleteKeyframe(
      int keyframe_index,
      LoopDetector* loop_detector);
  
  // Merges keyframes to free up GPU memory.
  // NOTE: This leads to nullptr entries in the keyframes vector.
  // TODO: This is currently misnamed, since it does not merge keyframes but simply delete them.
  void MergeKeyframes(
      cudaStream_t stream,
      LoopDetector* loop_detector,
      usize approx_merge_count = 10);
  
  // Creates new surfels for the depth pixels of the given keyframe which do not
  // correspond to existing surfels. If filter_new_surfels is true, applies
  // outlier filtering to discard new surfels which are considered outliers.
  void CreateSurfelsForKeyframe(
      cudaStream_t stream,
      bool filter_new_surfels,
      const shared_ptr<Keyframe>& keyframe);
  
  // Optimizes an RGB-D frame's pose by maximizing consistency with the surfel
  // model.
  // NOTE: The implementation of this function is in direct_ba_alternating.cc.
  void EstimateFramePose(
      cudaStream_t stream,
      const SE3f& global_T_frame_initial_estimate,
      const CUDABuffer<u16>& depth_buffer,
      const CUDABuffer<u16>& normals_buffer,
      const cudaTextureObject_t color_texture,
      SE3f* out_global_T_frame_estimate,
      bool called_within_ba);
  
  // Runs bundle adjustment using an alternating optimization scheme. For
  // standard behavior, set active_keyframe_window_start to 0,
  // active_keyframe_window_end to the highest valid keyframe index,
  // increase_ba_iteration_count to true, and leave all optional parameters at
  // their defaults. With use_pcg == true, the function will use a
  // preconditioned conjugate gradients (PCG) based solver for the Gauss-Newton
  // update equation instead of an alternating optimization scheme. This means
  // that a single optimization iteration will take more time, but it might
  // converge faster overall.
  // NOTE: The implementation to this function is in separate files due to its
  //       length. For use_pcg == false, it is in direct_ba_alternating.cc, and
  //       for use_pcg == true, it is in direct_ba_pcg.cc.
  void BundleAdjustment(
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
      int* iterations_done = nullptr,
      bool* converged = nullptr,
      double time_limit = 0,
      Timer* timer = nullptr,
      int pcg_max_inner_iterations = 30,
      int pcg_max_keyframes = 2500,
      std::function<bool (int)> progress_function = nullptr);
  
  // Using the current keyframe poses, projects all surfels into all keyframes
  // to assign colors to the surfels. This is purely for visualization purposes
  // and is not used for bundle adjustment.
  void AssignColors(cudaStream_t stream);
  
  // Makes BA run in a separate thread. The BadSlam object has to be given here
  // since the thread will call back functions on it to have the poses of
  // non-keyframes updated after BA iterations finish.
  void SetRunParallel(BadSlam* bad_slam, OpenGLContext* opengl_context);
  
  // Exports the surfel cloud (from the GPU) to a PointCloud object (on the CPU).
  void ExportToPointCloud(
      cudaStream_t stream,
      Point3fC3u8NfCloud* cloud) const;
  
  // Updates the parts of the visualization which should be updated after a BA
  // iteration finishes.
  void UpdateBAVisualization(cudaStream_t stream);
  
  // Updates the co-visibility list of the given keyframe.
  void UpdateKeyframeCoVisibility(const shared_ptr<Keyframe>& keyframe);
  
  
  // Locks the ba_thread_mutex_ which is used with parallel BA.
  // If using parallel BA, this mutex must be locked for accessing:
  // - keyframes()
  // - color_camera()
  // - depth_camera()
  // - depth_params()
  // - a()
  // - ba_iteration_count_
  // - TODO: for which other accesses as well?
  inline void Lock() const {
    ba_thread_mutex_.lock();
  }
  
  // Unlocks the ba_thread_mutex_ which is used with parallel BA.
  inline void Unlock() const {
    ba_thread_mutex_.unlock();
  }
  
  // Returns ba_thread_mutex_ (for use with a lock).
  inline std::mutex& Mutex() const {
    return ba_thread_mutex_;
  }
  
  
  // Returns the minimum observation count that is required to construct /
  // maintain a surfel. It depends on the number of keyframes that exist since
  // we need to have a bootstrapping phase at the beginning. Otherwise, if we
  // directly started with e.g. min_observation_count == 2, with only 2
  // keyframes there might be very few surfels which means that pose estimation
  // using those surfels might easily result in a noisy, bad pose, which might
  // destroy the reconstruction. The bootstrapping requires less observations at
  // the beginning, which helps to get a good number of observations first
  // before enforcing a high min_observation_count.
  inline int GetMinObservationCount() const {
    return (keyframes_.size() < 10) ?
              ((keyframes_.size() < 5) ?
                  min_observation_count_while_bootstrapping_1_ :
                  min_observation_count_while_bootstrapping_2_) :
              min_observation_count_;
  }
  
  // --- Accessors ---
  
  inline const vector<shared_ptr<Keyframe>>& keyframes() const {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return keyframes_;
  }
  inline vector<shared_ptr<Keyframe>>* keyframes_mutable() {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return &keyframes_;
  }
  
  inline PinholeCamera4f color_camera() const {
    lock_guard<mutex> lock(ba_thread_mutex_);
    return color_camera_;
  }
  inline PinholeCamera4f color_camera_no_lock() const {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return color_camera_;
  }
  inline void SetColorCamera(const PinholeCamera4f& camera) {
    lock_guard<mutex> lock(ba_thread_mutex_);
    color_camera_ = camera;
  }
  
  inline int pyramid_level_for_color() const { return pyramid_level_for_color_; }
  inline void SetPyramidLevelForColor(int level) { pyramid_level_for_color_ = level; }
  
  inline PinholeCamera4f depth_camera() const {
    lock_guard<mutex> lock(ba_thread_mutex_);
    return depth_camera_;
  }
  inline PinholeCamera4f depth_camera_no_lock() const {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return depth_camera_;
  }
  inline void SetDepthCamera(const PinholeCamera4f& camera) {
    lock_guard<mutex> lock(ba_thread_mutex_);
    depth_camera_ = camera;
  }
  
  inline DepthParameters depth_params() const {
    lock_guard<mutex> lock(ba_thread_mutex_);
    return depth_params_;
  }
  inline DepthParameters depth_params_no_lock() const {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return depth_params_;
  }
  inline void SetDepthParams(const DepthParameters& params) {
    lock_guard<mutex> lock(ba_thread_mutex_);
    depth_params_ = params;
  }
  
  inline float& a() {
#ifdef DEBUG_LOCKING
    CHECK(!ba_thread_mutex_.try_lock());
#endif
    return depth_params_.a;
  }
  inline float a() const {
    return depth_params_.a;
  }
  
  inline CUDABufferPtr<float> cfactor_buffer() {
    return cfactor_buffer_;
  }
  inline CUDABufferConstPtr<float> cfactor_buffer() const {
    return cfactor_buffer_;
  }
  inline void SetCFactorBuffer(const CUDABufferPtr<float>& cfactor_buffer) {
    cfactor_buffer_ = cfactor_buffer;
    depth_params_.cfactor_buffer = cfactor_buffer_->ToCUDA();
  }
  
  inline void IncreaseBAIterationCount() {
    lock_guard<mutex> lock(ba_thread_mutex_);
    ++ ba_iteration_count_;
  }
  
  inline bool use_depth_residuals() const {
    return use_depth_residuals_;
  }
  inline void SetUseDepthResiduals(bool use_depth_residuals) {
    use_depth_residuals_ = use_depth_residuals;
  }
  
  inline bool use_descriptor_residuals() const {
    return use_descriptor_residuals_;
  }
  inline void SetUseDescriptorResiduals(bool use_descriptor_residuals) {
    use_descriptor_residuals_ = use_descriptor_residuals;
  }
  
  inline int sparse_surfel_cell_size() const {
    return depth_params_.sparse_surfel_cell_size;
  }
  inline void SetSparsificationSideFactor(int sparse_surfel_cell_size) {
    depth_params_.sparse_surfel_cell_size = sparse_surfel_cell_size;
  }
  
  inline int min_observation_count_while_bootstrapping_1() const {
    return min_observation_count_while_bootstrapping_1_;
  }
  inline void SetMinObservationCountWhileBootstrapping1(int count) {
    min_observation_count_while_bootstrapping_1_ = count;
  }
  
  inline int min_observation_count_while_bootstrapping_2() const {
    return min_observation_count_while_bootstrapping_2_;
  }
  inline void SetMinObservationCountWhileBootstrapping2(int count) {
    min_observation_count_while_bootstrapping_2_ = count;
  }
  
  inline int min_observation_count() const {
    return min_observation_count_;
  }
  inline void SetMinObservationCount(int count) {
    min_observation_count_ = count;
  }
  
  /// Sets a function that is called by the BA thread each time after the
  /// intrinsics and depth deformation are updated. Can be used to visualize the
  /// intrinsics, for example.
  inline void SetIntrinsicsUpdatedCallback(const std::function<void()>& callback) {
    intrinsics_updated_callback_ = callback;
  }
  
  inline u32 surfel_count() const { lock_guard<mutex> lock(ba_thread_mutex_); return surfel_count_; }
  inline u32 surfels_size() const { lock_guard<mutex> lock(ba_thread_mutex_); return surfels_size_; }
  inline void SetSurfelCount(u32 surfel_count, u32 surfels_size) { surfel_count_ = surfel_count; surfels_size_ = surfels_size; }
  
  inline CUDABufferConstPtr<float> surfels() const { return surfels_; }
  inline CUDABufferPtr<float> surfels() { return surfels_; }
  
  inline int ba_iteration_count() const { return ba_iteration_count_; }
  inline void SetBAIterationCount(int count) { ba_iteration_count_ = count; }
  
  inline int last_ba_iteration_count() const { return last_ba_iteration_count_; }
  inline void SetLastBAIterationCount(int count) { last_ba_iteration_count_ = count; }
  
  inline float surfel_merge_dist_factor() const { return surfel_merge_dist_factor_; }
  inline void SetSurfelMergeDistFactor(float factor) { surfel_merge_dist_factor_ = factor; }
  
  inline void SetSaveTimings(std::ofstream* stream) { timings_stream_ = stream; }
  
  inline void SetVisualization(bool visualize_normals, bool visualize_descriptors, bool visualize_radii) {
    visualize_normals_ = visualize_normals;
    visualize_descriptors_ = visualize_descriptors;
    visualize_radii_ = visualize_radii;
  }
  
 private:
  // NOTE: The implementation of this function is in direct_ba_alternating.cc.
  void BundleAdjustmentAlternating(
      cudaStream_t stream,
      bool optimize_depth_intrinsics,
      bool optimize_color_intrinsics,
      bool do_surfel_updates,
      bool optimize_poses,
      bool optimize_geometry,
      int min_iterations,
      int max_iterations,
      int active_keyframe_window_start,
      int active_keyframe_window_end,
      bool increase_ba_iteration_count,
      int* num_iterations_done,
      bool* converged,
      double time_limit,
      Timer* timer,
      std::function<bool (int)> progress_function);
  
  // NOTE: The implementation of this function is in direct_ba_pcg.cc.
  void BundleAdjustmentPCG(
      cudaStream_t stream,
      bool optimize_depth_intrinsics,
      bool optimize_color_intrinsics,
      bool do_surfel_updates,
      bool optimize_poses,
      bool optimize_geometry,
      int min_iterations,
      int max_iterations,
      int max_inner_iterations,
      int max_keyframe_count,
      int active_keyframe_window_start,
      int active_keyframe_window_end,
      bool increase_ba_iteration_count,
      int* num_iterations_done,
      bool* converged,
      double time_limit,
      Timer* timer,
      std::function<bool (int)> progress_function);
  
  void DetermineCovisibleActiveKeyframes();
  
  void DetermineNewKeyframeCoVisibility(const shared_ptr<Keyframe>& new_keyframe);
  
  void PerformBASchemeEndTasks(
    cudaStream_t stream,
    bool do_surfel_updates);
  
  
  // GPU buffers for PCG-based bundle adjustment implementation.
  CUDABufferPtr<PCGScalar> pcg_r_;
  CUDABufferPtr<PCGScalar> pcg_M_;
  CUDABufferPtr<PCGScalar> pcg_delta_;
  CUDABufferPtr<PCGScalar> pcg_g_;
  CUDABufferPtr<PCGScalar> pcg_p_;
  CUDABufferPtr<PCGScalar> pcg_alpha_n_;
  CUDABufferPtr<PCGScalar> pcg_alpha_d_;
  CUDABufferPtr<PCGScalar> pcg_beta_n_;
  
  // Camera intrinsic parameters.
  PinholeCamera4f color_camera_;
  int pyramid_level_for_color_;
  PinholeCamera4f depth_camera_;
  CUDABufferPtr<float> cfactor_buffer_;
  // NOTE: Intrinsics also include the "a" (alpha) member in depth_params_.
  
  DepthParameters depth_params_;
  
  // Vector of all keyframes. Indexed by: [keyframe_id].
  // The ordering is such that keyframes with larger ID correspond to frames
  // that are later in the video.
  vector<shared_ptr<Keyframe>> keyframes_;
  
  // Number of valid surfels.
  u32 surfel_count_;
  
  // Number of valid plus invalid surfels (indicates the highest surfel index a
  // CUDA kernel must run on to run on all surfels).
  u32 surfels_size_;
  
  // GPU buffer with all surfels.
  CUDABufferPtr<float> surfels_;
  
  // GPU buffer with a u8 for each surfel, having bit 1 set if the surfel is
  // currently active, and bit 2 set if the surfel has been active at some
  // point during the current BA iteration block.
  CUDABufferPtr<u8> active_surfels_;
  
  // Helper buffer for surfel deletion.
  CUDABufferPtr<u32> deleted_count_buffer_;
  
  // Number of BA iteration blocks performed so far. Used for determining which
  // keyframes were active yet in the current iteration.
  int ba_iteration_count_;
  int last_ba_iteration_count_;
  
  // Settings.
  bool use_depth_residuals_;
  bool use_descriptor_residuals_;
  
  int min_observation_count_while_bootstrapping_1_;
  int min_observation_count_while_bootstrapping_2_;
  int min_observation_count_;
  
  float surfel_merge_dist_factor_;
  
  // Temporary CUDA buffers.
  void* new_surfels_temp_storage_;
  usize new_surfels_temp_storage_bytes_;
  void* free_spots_temp_storage_;
  usize free_spots_temp_storage_bytes_;
  CUDABufferPtr<u8> new_surfel_flag_vector_;
  CUDABufferPtr<u32> new_surfel_indices_;
  CUDABufferPtr<u32> supporting_surfels_[kMergeBufferCount];
  
  PoseEstimationHelperBuffers pose_estimation_helper_buffers_;
  IntrinsicsOptimizationHelperBuffers intrinsics_optimization_helper_buffers_;
  
  // Parallelism.
  mutable mutex ba_thread_mutex_;
  
  // For timing.
  cudaEvent_t ba_surfel_creation_pre_event_;
  cudaEvent_t ba_surfel_creation_post_event_;
  cudaEvent_t ba_surfel_activation_pre_event_;
  cudaEvent_t ba_surfel_activation_post_event_;
  cudaEvent_t ba_surfel_compaction_pre_event_;
  cudaEvent_t ba_surfel_compaction_post_event_;
  cudaEvent_t ba_geometry_optimization_pre_event_;
  cudaEvent_t ba_geometry_optimization_post_event_;
  cudaEvent_t ba_surfel_merge_pre_event_;
  cudaEvent_t ba_surfel_merge_post_event_;
  cudaEvent_t ba_pose_optimization_pre_event_;
  cudaEvent_t ba_pose_optimization_post_event_;
  cudaEvent_t ba_intrinsics_optimization_pre_event_;
  cudaEvent_t ba_intrinsics_optimization_post_event_;
  cudaEvent_t ba_final_surfel_deletion_and_radius_update_pre_event_;
  cudaEvent_t ba_final_surfel_deletion_and_radius_update_post_event_;
  cudaEvent_t ba_final_surfel_merge_pre_event_;
  cudaEvent_t ba_final_surfel_merge_post_event_;
  cudaEvent_t ba_pcg_pre_event_;
  cudaEvent_t ba_pcg_post_event_;
  
  std::ofstream* timings_stream_;
  
  // For convergence samples gathering.
  bool gather_convergence_samples_ = false;  // NOTE: This must be activated by setting it to true here to use it.
  std::ofstream convergence_samples_file_;
  
  // For visualization.
  shared_ptr<BadSlamRenderWindow> render_window_;
  SE3f global_T_anchor_frame_;
  cudaGraphicsResource_t surfel_vertices_;
  
  bool visualize_normals_ = false;
  bool visualize_descriptors_ = false;
  bool visualize_radii_ = false;
  
  std::function<void()> intrinsics_updated_callback_;
};

}
