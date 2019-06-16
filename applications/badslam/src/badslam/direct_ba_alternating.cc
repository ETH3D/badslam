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


#include "badslam/direct_ba.h"

#include <libvis/timing.h>

#include "badslam/convergence_analysis.h"
#include "badslam/render_window.h"
#include "badslam/util.cuh"

namespace vis {

constexpr bool kDebugVerifySurfelCount = false;

void DirectBA::EstimateFramePose(cudaStream_t stream,
                                 const SE3f& global_T_frame_initial_estimate,
                                 const CUDABuffer<u16>& depth_buffer,
                                 const CUDABuffer<u16>& normals_buffer,
                                 const cudaTextureObject_t color_texture,
                                 SE3f* out_global_T_frame_estimate,
                                 bool called_within_ba) {
  static int call_counter = 0;
  ++ call_counter;
  
  // Set kDebug to true to activate some debug outputs:
  bool kDebug = false; // bool kDebug = call_counter >= 500 && !called_within_ba;
  (void) called_within_ba;
  
  // Set this to true to debug apparent wrong pose estimates:
  constexpr bool kDebugDivergence = false;
  
repeat_pose_estimation:;
  
  SE3f global_T_frame_estimate = global_T_frame_initial_estimate;
  
  shared_ptr<Point3fC3u8Cloud> debug_frame_cloud;
  if (kDebug) {
    // Show point cloud.
//     Image<u16> depth_buffer_calibrated(depth_buffer.width(), depth_buffer.height());
    Image<u16> depth_buffer_cpu(depth_buffer.width(), depth_buffer.height());
    depth_buffer.DownloadAsync(stream, &depth_buffer_cpu);
    cudaStreamSynchronize(stream);
    
    Image<float> cfactor_buffer_cpu(cfactor_buffer_->width(), cfactor_buffer_->height());
    cfactor_buffer_->DownloadAsync(stream, &cfactor_buffer_cpu);
    cudaStreamSynchronize(stream);
    
    usize point_count = 0;
    for (u32 y = 0; y < depth_buffer_cpu.height(); ++ y) {
      const u16* ptr = depth_buffer_cpu.row(y);
      const u16* end = ptr + depth_buffer_cpu.width();
      while (ptr < end) {
        if (!(*ptr & kInvalidDepthBit)) {
          ++ point_count;
        }
        ++ ptr;
      }
    }
    
    debug_frame_cloud.reset(new Point3fC3u8Cloud(point_count));
    usize point_index = 0;
    for (int y = 0; y < depth_buffer.height(); ++ y) {
      for (int x = 0; x < depth_buffer.width(); ++ x) {
        if (kInvalidDepthBit & depth_buffer_cpu(x, y)) {
//           depth_buffer_calibrated(x, y) = numeric_limits<u16>::max();
          continue;
        }
        float depth = RawToCalibratedDepth(
            depth_params_.a,
            cfactor_buffer_cpu(x / depth_params_.sparse_surfel_cell_size,
                               y / depth_params_.sparse_surfel_cell_size),
            depth_params_.raw_to_float_depth,
            depth_buffer_cpu(x, y));
//         depth_buffer_calibrated(x, y) = depth / depth_params_.raw_to_float_depth;
        Point3fC3u8& point = debug_frame_cloud->at(point_index);
        point.position() = depth * depth_camera_.UnprojectFromPixelCenterConv(Vec2f(x, y));
        point.color() = Vec3u8(255, 80, 80);
        ++ point_index;
      }
    }
    
    LOG(INFO) << "Debug: initial estimate for camera position: " << global_T_frame_estimate.translation().transpose();
    if (render_window_) {
      render_window_->SetCurrentFramePose(global_T_frame_estimate.matrix());  // TODO: Display an additional frustum here instead of mis-using the current camera pose frustum.
      
      render_window_->SetFramePointCloud(
          debug_frame_cloud,
          global_T_frame_estimate);
      
      render_window_->RenderFrame();
    }
    std::getchar();
  }
  
  if (gather_convergence_samples_) {
    convergence_samples_file_ << "EstimateFramePose()" << std::endl;
  }
  
  // Coefficients for update equation: H * x = b
  Eigen::Matrix<float, 6, 6> H;
  Eigen::Matrix<float, 6, 1> b;
  
  const int kMaxIterations = gather_convergence_samples_ ? 100 : 30;
  bool converged = false;
  int iteration;
  for (iteration = 0; iteration < kMaxIterations; ++ iteration) {
    if (kDebug) {
      LOG(INFO) << "Debug: iteration " << iteration;
    }
    
    u32 residual_count = 0;
    float residual_sum = 0;
    SE3f frame_T_global_estimate = global_T_frame_estimate.inverse();
    
    // Accumulate update equation coefficients (H, b) from cost term Jacobians.
    // NOTE: We handle Eigen objects outside of code compiled by the CUDA
    //       compiler only, since otherwise there were wrong results on my
    //       laptop.
    // TODO: Can probably be dropped, since this was likely due to a compiler
    //       version or setting mismatch
    if (surfels_size_ == 0) {
      H.setZero();
      b.setZero();
    } else {
      float H_temp[6 * (6 + 1) / 2];
      AccumulatePoseEstimationCoeffsCUDA(
          stream,
          use_depth_residuals_,
          use_descriptor_residuals_,
          color_camera_,
          depth_camera_,
          depth_params_,
          depth_buffer,
          normals_buffer,
          color_texture,
          CUDAMatrix3x4(frame_T_global_estimate.matrix3x4()),
          surfels_size_,
          *surfels_,
          kDebug || gather_convergence_samples_,
          &residual_count,
          &residual_sum,
          H_temp,
          b.data(),
          &pose_estimation_helper_buffers_);
      
      int index = 0;
      for (int row = 0; row < 6; ++ row) {
        for (int col = row; col < 6; ++ col) {
          H(row, col) = H_temp[index];
          ++ index;
        }
      }
    }
    
    if (kDebug) {
      for (int row = 0; row < 6; ++ row) {
        for (int col = row + 1; col < 6; ++ col) {
          H(col, row) = H(row, col);
        }
      }
      
      LOG(INFO) << "Debug: surfel_count = " << surfel_count_;
      LOG(INFO) << "Debug: residual_sum = " << residual_sum;
      LOG(INFO) << "Debug: residual_count = " << residual_count;
      LOG(INFO) << "Debug: H = " << std::endl << H;
      LOG(INFO) << "Debug: b = " << std::endl << b;
      
      // Condition number of H using Frobenius norm. In octave / Matlab:
      // cond(H, "fro") = norm(H, "fro") * norm(inv(H), "fro")
      // If this is very high, the pose is probably not well constrained.
      // However, noise can make it appear well-conditioned when it should not be!
      // NOTE: H needs to be fully set symetrically for this to work.
//       float cond_H_fro = H.norm() * H.inverse().norm();
//       LOG(INFO) << "Debug: cond(H, \"fro\") = " << cond_H_fro;
    }
    
    // Solve for the update x
    // NOTE: Not sure if using double is helpful here
    Eigen::Matrix<float, 6, 1> x = H.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(b.cast<double>()).cast<float>();
    
    if (kDebug) {
      LOG(INFO) << "Debug: x = " << std::endl << x;
    }
    
    // Apply the (negative) update -x.
    constexpr float kDamping = 1.f;
    global_T_frame_estimate = global_T_frame_estimate * SE3f::exp(-kDamping * x);
    
    if (kDebug) {
      LOG(INFO) << "Debug: camera position: " << global_T_frame_estimate.translation().transpose();
      if (render_window_) {
        render_window_->SetCurrentFramePose(global_T_frame_estimate.matrix());  // TODO: Display an additional frustum here instead of mis-using the current camera pose frustum.
        
        render_window_->SetFramePointCloud(
            debug_frame_cloud,
            global_T_frame_estimate);
        
        render_window_->RenderFrame();
      }
      std::getchar();
    }
    
    // Check for convergence
    converged = IsScale1PoseEstimationConverged(x);
    if (!gather_convergence_samples_ && converged) {
      if (kDebug) {
        LOG(INFO) << "Debug: Assuming convergence.";
      }
      converged = true;
      ++ iteration;
      break;
    } else if (gather_convergence_samples_) {
      convergence_samples_file_ << "iteration " << iteration << std::endl;
      convergence_samples_file_ << "x " << x.transpose() << std::endl;
      convergence_samples_file_ << "residual_sum " << residual_sum << std::endl;
    }
  }
  
  if (!converged) {
    static int not_converged_count = 0;
    ++ not_converged_count;
    LOG(WARNING) << "Pose estimation not converged (not_converged_count: " << not_converged_count << ", call_counter: " << call_counter << ")";
  }
  
//   static float average_iteration_count = 0;
//   average_iteration_count = ((call_counter - 1) * average_iteration_count + iteration) / (1.0f * call_counter);
//   if (call_counter % 200 == 0) {
//     LOG(INFO) << "Average pose optimization iteration count: " << average_iteration_count;
//   }
  
  // Debug check for divergence
  constexpr float kDebugDivergenceCheckThresholdDistance = 0.3f;
  if (kDebugDivergence &&
      (global_T_frame_initial_estimate.translation() - global_T_frame_estimate.translation()).squaredNorm() >=
       kDebugDivergenceCheckThresholdDistance * kDebugDivergenceCheckThresholdDistance) {
    LOG(ERROR) << "Pose estimation divergence detected (movement from initial estimate is larger than the threshold of " << kDebugDivergenceCheckThresholdDistance << " meters).";
    LOG(ERROR) << "(Use a backtrace to see in which part it occurred.)";
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
  
  if (kDebug && render_window_) {
    render_window_->UnsetFramePointCloud();
  }
  
  *out_global_T_frame_estimate = global_T_frame_estimate;
}

void DirectBA::BundleAdjustmentAlternating(
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
    std::function<bool (int)> progress_function) {
  if (converged) {
    *converged = false;
  }
  if (num_iterations_done) {
    *num_iterations_done = 0;
  }
  
  Lock();
  int fixed_ba_iteration_count = ba_iteration_count_;
  Unlock();
  
  if (!increase_ba_iteration_count &&
      fixed_ba_iteration_count != last_ba_iteration_count_) {
    last_ba_iteration_count_ = fixed_ba_iteration_count;
    PerformBASchemeEndTasks(
        stream,
        do_surfel_updates);
  }
  
  CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels[i] = supporting_surfels_[i].get();
  }
  
  vector<u32> keyframes_with_new_surfels;
  keyframes_with_new_surfels.reserve(keyframes_.size());
  
  
  bool fixed_active_keyframe_set =
      active_keyframe_window_start > 0 || active_keyframe_window_end > 0;
  
  if (active_keyframe_window_start != 0 || active_keyframe_window_end != keyframes_.size() - 1) {
    LOG(WARNING) << "Currently, only using all keyframes in every optimization iteration will work properly. Deactivated keyframes will not be used for surfel descriptor optimization, potentially leaving some surfel descriptors in a bad state.";
  }
  
  // Initialize surfel active states.
  cudaMemsetAsync(active_surfels_->ToCUDA().address(), 0, surfels_size_ * sizeof(u8), stream);
  
  if (kDebugVerifySurfelCount) {
    DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
  }
  
  // Perform BA iterations.
  for (int iteration = 0; iteration < max_iterations; ++ iteration) {
    if (progress_function && !progress_function(iteration)) {
      break;
    }
    if (num_iterations_done) {
      ++ *num_iterations_done;
    }
    
    // Keyframe activation in case of fixed window
    if (fixed_active_keyframe_set) {
      Lock();
      
      for (u32 keyframe_index = 0; keyframe_index < keyframes_.size(); ++ keyframe_index) {
        if (!keyframes_[keyframe_index]) {
          continue;
        }
        
        if (keyframe_index >= static_cast<u32>(active_keyframe_window_start) && keyframe_index <= static_cast<u32>(active_keyframe_window_end)) {
          keyframes_[keyframe_index]->SetActivation(Keyframe::Activation::kActive);
        } else {
          keyframes_[keyframe_index]->SetActivation(Keyframe::Activation::kInactive);
        }
      }
      
      DetermineCovisibleActiveKeyframes();
      
      Unlock();
    }
    
    // Debug print?
    constexpr bool kPrintKeyframeActivationStates = false;
    if (kPrintKeyframeActivationStates) {
      int debug_active_count = 0;
      int debug_covisible_active_count = 0;
      int debug_inactive_count = 0;
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        if (keyframe->activation() == Keyframe::Activation::kActive) {
          ++ debug_active_count;
        } else if (keyframe->activation() == Keyframe::Activation::kCovisibleActive) {
          ++ debug_covisible_active_count;
        } else if (keyframe->activation() == Keyframe::Activation::kInactive) {
          ++ debug_inactive_count;
        }
      }
      
      LOG(INFO) << "[iteration " << iteration << "] active: " << debug_active_count << ", covis-active: " << debug_covisible_active_count << ", inactive: " << debug_inactive_count;
    }
    
    
    // --- SURFEL CREATION ---
    keyframes_with_new_surfels.clear();
    
    CHECK_EQ(surfels_size_, surfel_count_);
    usize old_surfels_size = surfels_size_;
    
    if (optimize_geometry && do_surfel_updates) {
      Lock();
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        if (keyframe->activation() == Keyframe::Activation::kActive &&
            keyframe->last_active_in_ba_iteration() != fixed_ba_iteration_count) {
          keyframe->SetLastActiveInBAIteration(fixed_ba_iteration_count);
          
          // This keyframe has become active the first time within this BA
          // iteration block.
          keyframes_with_new_surfels.push_back(keyframe->id());
        } else if (keyframe->activation() == Keyframe::Activation::kCovisibleActive &&
                  keyframe->last_covis_in_ba_iteration() != fixed_ba_iteration_count) {
          keyframe->SetLastCovisInBAIteration(fixed_ba_iteration_count);
        }
      }
      Unlock();
      
      cudaEventRecord(ba_surfel_creation_pre_event_, stream);
      for (u32 keyframe_id : keyframes_with_new_surfels) {
        // TODO: Would it be better for performance to group all keyframes
        //       together that become active in an iteration?
        CreateSurfelsForKeyframe(stream, /* filter_new_surfels */ true, keyframes_[keyframe_id]);
      }
      cudaEventRecord(ba_surfel_creation_post_event_, stream);
    }
    
    
    // --- SURFEL ACTIVATION ---
    cudaEventRecord(ba_surfel_activation_pre_event_, stream);
    
    // Set new surfels to active | have_been_active.
    if (optimize_geometry &&
        surfels_size_ > old_surfels_size) {
      cudaMemsetAsync(active_surfels_->ToCUDA().address() + old_surfels_size,
                      kSurfelActiveFlag,
                      (surfels_size_ - old_surfels_size) * sizeof(u8),
                      stream);
    }
    
    // Update activation state of old surfels.
    if (active_keyframe_window_start != 0 || active_keyframe_window_end != keyframes_.size() - 1) {
      cudaMemsetAsync(active_surfels_->ToCUDA().address(), kSurfelActiveFlag, old_surfels_size * sizeof(u8), stream);
    } else {
      UpdateSurfelActivationCUDA(
          stream,
          depth_camera_,
          depth_params_,
          keyframes_,
          old_surfels_size,
          surfels_.get(),
          active_surfels_.get());
    }
    
    cudaEventRecord(ba_surfel_activation_post_event_, stream);
    
    if (kDebugVerifySurfelCount) {
      DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
    }
    
    
    // --- GEOMETRY OPTIMIZATION ---
    if (optimize_geometry) {
      cudaEventRecord(ba_geometry_optimization_pre_event_, stream);
      OptimizeGeometryIterationCUDA(
          stream,
          use_depth_residuals_,
          use_descriptor_residuals_,
          color_camera_,
          depth_camera_,
          depth_params_,
          keyframes_,
          surfels_size_,
          *surfels_,
          *active_surfels_);
      cudaEventRecord(ba_geometry_optimization_post_event_, stream);
      
      if (kDebugVerifySurfelCount) {
        DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
      }
    }
    
    
    // --- SURFEL MERGE ---
    // For keyframes for which new surfels were created at the start of the
    // iteration (a subset of the active keyframes).
    if (do_surfel_updates) {
      u32 surfel_count = surfel_count_;
      cudaEventRecord(ba_surfel_merge_pre_event_, stream);
      for (int keyframe_id : keyframes_with_new_surfels) {
        const shared_ptr<Keyframe>& keyframe = keyframes_[keyframe_id];
        if (!keyframe) {
          continue;
        }
        
        // TODO: Run this on the active surfels only if faster, should still be correct
        DetermineSupportingSurfelsAndMergeSurfelsCUDA(
            stream,
            surfel_merge_dist_factor_,
            depth_camera_,
            keyframe->frame_T_global_cuda(),
            depth_params_,
            keyframe->depth_buffer(),
            keyframe->normals_buffer(),
            surfels_size_,
            surfels_.get(),
            supporting_surfels,
            &surfel_count,
            &deleted_count_buffer_);
      }
      cudaEventRecord(ba_surfel_merge_post_event_, stream);
      Lock();
      surfel_count_ = surfel_count;
      Unlock();
      
      if (kDebugVerifySurfelCount) {
        DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
      }
      
      cudaEventRecord(ba_surfel_compaction_pre_event_, stream);
      if (!keyframes_with_new_surfels.empty()) {
        // Compact the surfels list to increase performance of subsequent kernel calls.
        // TODO: Only run on the new surfels if possible
        
        u32 surfels_size = surfels_size_;
        CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count_, &surfels_size, &surfels_->ToCUDA(), &active_surfels_->ToCUDA());
        Lock();
        surfels_size_ = surfels_size;
        Unlock();
      }
      cudaEventRecord(ba_surfel_compaction_post_event_, stream);
      
      if (kDebugVerifySurfelCount) {
        DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
      }
    }
    
    
    // --- POSE OPTIMIZATION ---
    usize num_converged = 0;
    if (optimize_poses) {
      cudaEventRecord(ba_pose_optimization_pre_event_, stream);
      for (const shared_ptr<Keyframe>& keyframe : keyframes_) {
        // Only estimate pose for active and covisible-active keyframes.
        if (!keyframe || keyframe->activation() == Keyframe::Activation::kInactive) {
          ++ num_converged;
          continue;
        }
        
        SE3f global_T_frame_estimate;
        EstimateFramePose(stream,
                          keyframe->global_T_frame(),
                          keyframe->depth_buffer(),
                          keyframe->normals_buffer(),
                          keyframe->color_texture(),
                          &global_T_frame_estimate,
                          true);
        SE3f pose_difference = keyframe->frame_T_global() * global_T_frame_estimate;
        bool frame_moved = !IsScale1PoseEstimationConverged(pose_difference.log());
        
        Lock();
        keyframe->set_global_T_frame(global_T_frame_estimate);
        
        if (frame_moved) {
          keyframe->SetActivation(Keyframe::Activation::kActive);
        } else {
          keyframe->SetActivation(Keyframe::Activation::kInactive);
          ++ num_converged;
        }
        Unlock();
      }
      cudaEventRecord(ba_pose_optimization_post_event_, stream);
    }
    
    if (kDebugVerifySurfelCount) {
      DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
    }
    
    
    // --- INTRINSICS OPTIMIZATION ---
    bool optimize_intrinsics =
        optimize_depth_intrinsics || optimize_color_intrinsics;
    if (optimize_intrinsics) {
      cudaEventRecord(ba_intrinsics_optimization_pre_event_, stream);
      PinholeCamera4f out_color_camera;
      PinholeCamera4f out_depth_camera;
      float out_a = depth_params_.a;
      
      OptimizeIntrinsicsCUDA(
          stream,
          optimize_depth_intrinsics,
          optimize_color_intrinsics,
          keyframes_,
          color_camera_,
          depth_camera_,
          depth_params_,
          surfels_size_,
          *surfels_,
          &out_color_camera,
          &out_depth_camera,
          &out_a,
          &cfactor_buffer_,
          &intrinsics_optimization_helper_buffers_);
      
      if (surfels_size_ > 0) {
        Lock();
        if (optimize_color_intrinsics) {
          color_camera_ = out_color_camera;
        }
        if (optimize_depth_intrinsics) {
          depth_camera_ = out_depth_camera;
          depth_params_.a = out_a;
        }
        Unlock();
      }
      
      cudaEventRecord(ba_intrinsics_optimization_post_event_, stream);
      
      if (intrinsics_updated_callback_) {
        intrinsics_updated_callback_();
      }
    }
    
    
    // --- TIMING ---
    if (timings_stream_) {
      *timings_stream_ << "BA_count " << fixed_ba_iteration_count << " inner_iteration " << iteration << " keyframe_count " << keyframes_.size()
                       << " surfel_count " << surfel_count_ << endl;
    }
    
    // Store timings for events used within this loop.
    cudaEventSynchronize(ba_intrinsics_optimization_post_event_);
    float elapsed_milliseconds;
    
    if (optimize_geometry && do_surfel_updates) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_creation_pre_event_, ba_surfel_creation_post_event_);
      Timing::addTime(Timing::getHandle("BA surfel creation"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_surfel_creation " << elapsed_milliseconds << endl;
      }
    }
    
    cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_activation_pre_event_, ba_surfel_activation_post_event_);
    Timing::addTime(Timing::getHandle("BA surfel activation"), 0.001 * elapsed_milliseconds);
    if (timings_stream_) {
      *timings_stream_ << "BA_surfel_activation " << elapsed_milliseconds << endl;
    }
    
    if (optimize_geometry) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_geometry_optimization_pre_event_, ba_geometry_optimization_post_event_);
      Timing::addTime(Timing::getHandle("BA geometry optimization"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_geometry_optimization " << elapsed_milliseconds << endl;
      }
    }
    
    if (do_surfel_updates) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_merge_pre_event_, ba_surfel_merge_post_event_);
      Timing::addTime(Timing::getHandle("BA initial surfel merge"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_initial_surfel_merge " << elapsed_milliseconds << endl;
      }
      
      cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_compaction_pre_event_, ba_surfel_compaction_post_event_);
      Timing::addTime(Timing::getHandle("BA surfel compaction"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_surfel_compaction " << elapsed_milliseconds << endl;
      }
    }
    
    if (optimize_poses) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_pose_optimization_pre_event_, ba_pose_optimization_post_event_);
      Timing::addTime(Timing::getHandle("BA pose optimization"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_pose_optimization " << elapsed_milliseconds << endl;
      }
    }
    
    if (optimize_intrinsics) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_intrinsics_optimization_pre_event_, ba_intrinsics_optimization_post_event_);
      Timing::addTime(Timing::getHandle("BA intrinsics optimization"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_intrinsics_optimization " << elapsed_milliseconds << endl;
      }
    }
    
    
    // --- CONVERGENCE ---
    if (iteration >= min_iterations - 1 &&
        (num_converged == keyframes_.size() || !optimize_poses)) {
      // All frames are inactive. Early exit.
//       LOG(INFO) << "Early global BA exit after " << (iteration + 1) << " iterations";
      if (converged) {
        *converged = true;
      }
      break;
    }
    
    // Test for timeout if a timer is given
    if (timer) {
      double elapsed_time = timer->GetTimeSinceStart();
      if (elapsed_time > time_limit) {
        break;
      }
    }
    
    // Partial convergence: keyframes have been set to kActive or kInactive
    // depending on whether they moved in the last pose estimation iteration.
    // Use the covisibility lists to determine which kInactive frames must be
    // changed to kCovisibleActive.
    Lock();
    DetermineCovisibleActiveKeyframes();
    Unlock();
  }
  
  if (kDebugVerifySurfelCount) {
    DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
  }
  
  
  if (increase_ba_iteration_count) {
    PerformBASchemeEndTasks(
        stream,
        do_surfel_updates);
    
    ++ ba_iteration_count_;
    
//     if (ba_iteration_count_ % 10 == 0) {
//       LOG(INFO) << Timing::print(kSortByTotal);
//     }
  }
  
  UpdateBAVisualization(stream);
}

}
