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
#include "badslam/kernels.cuh"
#include "badslam/util.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

constexpr bool kDebugVerifySurfelCount = false;

void DirectBA::BundleAdjustmentPCG(
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
    bool increase_ba_iteration_count,  // TODO: This parameter has side-effects that are not reflected in its name --> rename!
    int* num_iterations_done,
    bool* converged,
    double time_limit,
    Timer* timer,
    std::function<bool (int)> progress_function) {
  // This implementation of a PCG-based Gauss-Newton solver is mainly based on
  // Fig. 6 in the "Opt" paper:
  // 
  // Opt: A Domain Specific Language for Non-linear Least Squares Optimization
  // in Graphics and Imaging, by:
  // Zachary Devito, Michael Mara, Michael Zollhöfer, Gilbert Bernstein,
  // Jonathan Ragan-Kelley, Christian Theobalt, Pat Hanrahan, Matthew Fisher,
  // and Matthias Niessner.
  // 
  // Beware: at the time of writing, in their arxiv version, there is a mistake
  // in the last line of the algorithm: it should say "beta_n_k" instead of
  // "beta_k".
  // 
  // The algorithm is also stated here on Wikipedia, where the only difference
  // seems to be that A * x0 is subtracted from b at the start, and convergence
  // testing is mentioned as well:
  // https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_preconditioned_conjugate_gradient_method
  // 
  // The principle is as follows. In principle we apply Gauss-Newton, but instead
  // of solving for the update x exactly, we solve for it approximately. The
  // update equation for Gauss-Newton is:
  //   H * x = -b,
  // with H being the Gauss-Newton approximation to the Hessian, and x (corresponds
  // to pcg_delta) the desired delta vector containing the updates to the state variables.
  // We can write this equation based on its components, with r being the number
  // of residuals, and n being the number of unknowns in the state:
  // - F: [rx1] vector containing the residual values
  // - J: [rxn] matrix, where each row contains the Jacobian of the corresponding
  //      residual wrt. all unknowns in the state (where usually most values will
  //      be zero).
  // Since H = J^T J and -b = -J^T F, the equation looks as follows then:
  //   J^T J x = -J^T F
  // Since we use iteratively re-weighted least squares, we add a weight matrix W:
  //   J^T W J x = -J^T W F
  // Finally, we have to ensure that H is positive definite (instead of only
  // semi-positive definite), so we add a small value "lambda" to its diagonal
  // (this is the value "kDiagEpsilon" in kernel_pcg.cu), as it would also be
  // done in Levenberg-Marquardt:
  //   (J^T W J + lambda I) x = -J^T W F
  // 
  // Approximately determining x in this equation is the problem that the
  // preconditioned conjugate gradient (PCG) approach solves. The advantage is
  // that we don't need to compute the huge matrix H, instead it is sufficient
  // to be able to multiply vectors with it, which can be done while only knowing
  // small parts of J and W at a time.
  // 
  // The "trick" to implement the parts of the algorithm efficiently is
  // to split up the sums conveniently. Each CUDA thread then only needs to
  // compute one residual and its weight and Jacobian. The addition of "lambda I x"
  // to "J^T W J x" can be conveniently delayed to later in the algorithm when
  // the result of that computation is used.
  // 
  // The pcg_r variable takes the role of the residual of the PCG process, showing
  // how well the current estimate for x reaches b when multiplied with H. Its
  // norm can be used for convergence testing.
  // 
  // The "2"s in the specification of the algorithm in the Opt paper can be
  // skipped, since they cancel themselves out.
  
  constexpr bool kDebug = false;  // set this to true to get some debugging output
  
  if ((active_keyframe_window_start != -1 || active_keyframe_window_end != -1) &&
      (active_keyframe_window_start != 0 || active_keyframe_window_end != keyframes().size() - 1)) {
    LOG(WARNING) << "The PCG-based solver implementation does not support an active window! These parameters will be ignored.";
  }
  
  if (num_iterations_done) {
    *num_iterations_done = 0;
  }
  if (converged) {
    *converged = false;
  }
  
  // This optimization implementation currently does not support having deleted keyframe in the keyframe list, so check for this.
  // TODO: Make this work. The null keyframe entries should be ignored and the remaining ones should be indexed sequentially.
  //       See the "This will not work if keyframes are removed from the list." comments below.
  for (auto& keyframe : keyframes_) {
    if (!keyframe) {
      LOG(ERROR) << "The PCG-based solver implementation does not support having deleted keyframes yet! Aborting.";
      return;
    }
  }
  
  if (kDebug) {
    LOG(INFO) << "Debug: PCG iteration with:";
    LOG(INFO) << "optimize_depth_intrinsics: " << optimize_depth_intrinsics;
    LOG(INFO) << "optimize_color_intrinsics: " << optimize_color_intrinsics;
    LOG(INFO) << "do_surfel_updates: " << do_surfel_updates;
    LOG(INFO) << "optimize_poses: " << optimize_poses;
    LOG(INFO) << "optimize_geometry: " << optimize_geometry;
    LOG(INFO) << "use_depth_residuals: " << use_depth_residuals_;
    LOG(INFO) << "use_descriptor_residuals: " << use_descriptor_residuals_;
  }
  
  
  if (!increase_ba_iteration_count &&
      ba_iteration_count_ != last_ba_iteration_count_) {
    last_ba_iteration_count_ = ba_iteration_count_;
    PerformBASchemeEndTasks(stream, do_surfel_updates);
  }
  
  
  CUDABuffer<u32>* supporting_surfels[kMergeBufferCount];
  for (int i = 0; i < kMergeBufferCount; ++ i) {
    supporting_surfels[i] = supporting_surfels_[i].get();
  }
  
  vector<u32> keyframes_with_new_surfels;
  keyframes_with_new_surfels.reserve(keyframes_.size());
  
  // Loop over optimization iterations
  for (int iteration = 0; iteration < max_iterations; ++ iteration) {
    if (progress_function && !progress_function(iteration)) {
      break;
    }
    if (num_iterations_done) {
      ++ *num_iterations_done;
    }
    
    // Surfel creation?
    keyframes_with_new_surfels.clear();
    
    if (optimize_geometry && do_surfel_updates) {
      cudaEventRecord(ba_surfel_creation_pre_event_, stream);
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        if (keyframe->activation() == Keyframe::Activation::kActive &&
            keyframe->last_active_in_ba_iteration() != ba_iteration_count_) {
          keyframe->SetLastActiveInBAIteration(ba_iteration_count_);
          
          // This keyframe has become active the first time within this BA
          // iteration block.
          // TODO: Would it be better for performance to group all keyframes
          //       together that become active in an iteration?
          CreateSurfelsForKeyframe(stream, /* filter_new_surfels */ true, keyframe);
          keyframes_with_new_surfels.push_back(keyframe->id());
        } else if (keyframe->activation() == Keyframe::Activation::kCovisibleActive &&
                   keyframe->last_covis_in_ba_iteration() != ba_iteration_count_) {
          keyframe->SetLastCovisInBAIteration(ba_iteration_count_);
        }
      }
      cudaEventRecord(ba_surfel_creation_post_event_, stream);
    }
    
    // Set all surfels to active. TODO: Do this implicitly instead of requiring this memset.
    cudaMemsetAsync(active_surfels_->ToCUDA().address(),
                    kSurfelActiveFlag,
                    surfels_size_ * sizeof(u8),
                    stream);
    
    // Surfel normal update step
    if (optimize_geometry) {
      // NOTE: Slightly mis-using this event here since it is only used to measure normal updates
      cudaEventRecord(ba_geometry_optimization_pre_event_, stream);
      UpdateSurfelNormalsCUDA(
          stream,
          depth_camera_,
          depth_params_,
          keyframes_,
          surfels_size_,
          *surfels_,
          *active_surfels_);
      cudaEventRecord(ba_geometry_optimization_post_event_, stream);
    }
    
    // Perform a single optimization step using PCG.
    CHECK_LE(keyframes_.size(), max_keyframe_count);
    
    const u32 max_keyframes_unknown_count = 6 * (max_keyframe_count - 1);
    const u32 keyframes_unknown_count = optimize_poses ? (6 * (keyframes_.size() - 1)) : 0;  // TODO: This will not work if keyframes are removed from the list.
    CHECK_LE(keyframes_unknown_count, max_keyframes_unknown_count);
    
    const u32 max_surfels_unknown_count = 3 * surfels_->width();
    const u32 surfels_unknown_count = optimize_geometry ? ((use_descriptor_residuals_ ? 3 : 1) * surfels_size_) : 0;
    CHECK_LE(surfels_unknown_count, max_surfels_unknown_count);
    
    const u32 max_depth_intrinsics_unknown_count = 4 + 1 + cfactor_buffer_->width() * cfactor_buffer_->height();
    const u32 depth_intrinsics_unknown_count = optimize_depth_intrinsics ? max_depth_intrinsics_unknown_count : 0;
    CHECK_LE(depth_intrinsics_unknown_count, max_depth_intrinsics_unknown_count);
    
    const u32 max_color_intrinsics_unknown_count = 4;
    const u32 color_intrinsics_unknown_count = optimize_color_intrinsics ? max_color_intrinsics_unknown_count : 0;
    CHECK_LE(color_intrinsics_unknown_count, max_color_intrinsics_unknown_count);
    
    const u32 max_unknown_count =
        max_surfels_unknown_count +
        max_keyframes_unknown_count +
        max_depth_intrinsics_unknown_count +
        max_color_intrinsics_unknown_count;
    
    // Allocate buffers if not done yet
    if (!pcg_r_ || pcg_r_->width() < max_unknown_count) {
      if (pcg_r_) {
        LOG(WARNING) << "Re-allocating the PCG GPU buffers.";
      }
      
      pcg_r_.reset(new CUDABuffer<PCGScalar>(1, max_unknown_count));
      pcg_M_.reset(new CUDABuffer<PCGScalar>(1, max_unknown_count));
      pcg_delta_.reset(new CUDABuffer<PCGScalar>(1, max_unknown_count));
      pcg_g_.reset(new CUDABuffer<PCGScalar>(1, max_unknown_count));
      pcg_p_.reset(new CUDABuffer<PCGScalar>(1, max_unknown_count));
      pcg_alpha_n_.reset(new CUDABuffer<PCGScalar>(1, 1));
      pcg_alpha_d_.reset(new CUDABuffer<PCGScalar>(1, 1));
      pcg_beta_n_.reset(new CUDABuffer<PCGScalar>(1, 1));
    }
    
    // Unknown ordering:
    // - 6 unknowns for each keyframe, subtract 6 for gauge fixing (trivial gauge, fixing a certain keyframe pose)
    // - 1 or 3 for each surfel (one or all of: position offset along normal, descriptor part 1, descriptor part 2, in that order)
    // - 4 + 1 + cfactor buffer pixel count for depth intrinsics (fx, fy, cx, cy, a, cfactors)
    // - 4 for color intrinsics (fx, fy, cx, cy)
    u32 current_unknown_index = 0;
    constexpr u32 kInvalidUnknownIndex = numeric_limits<u32>::max();
    
    u32 keyframe_unknown_start_index = kInvalidUnknownIndex;
    (void) keyframe_unknown_start_index;  // TODO: Code further down assumes that this is always zero
    if (optimize_poses) {
      keyframe_unknown_start_index = current_unknown_index;
      current_unknown_index += keyframes_unknown_count;
    }
    
    u32 surfel_unknown_start_index = kInvalidUnknownIndex;
    if (optimize_geometry) {
      surfel_unknown_start_index = current_unknown_index;
      current_unknown_index += surfels_unknown_count;
    }
    
    u32 depth_intrinsics_unknown_start_index = kInvalidUnknownIndex;
    u32 a_unknown_index = kInvalidUnknownIndex;
    if (optimize_depth_intrinsics) {
      depth_intrinsics_unknown_start_index = current_unknown_index;
      current_unknown_index += depth_intrinsics_unknown_count;
      
      a_unknown_index = depth_intrinsics_unknown_start_index + 4;
    }
    
    u32 color_intrinsics_unknown_start_index = kInvalidUnknownIndex;
    if (optimize_color_intrinsics) {
      color_intrinsics_unknown_start_index = current_unknown_index;
      current_unknown_index += color_intrinsics_unknown_count;
    }
    
    u32 unknown_count = current_unknown_index;
    CHECK_LE(unknown_count, max_unknown_count);
    
    if (kDebug) {
      LOG(INFO) << "Debug: max_unknown_count: " << max_unknown_count << ", unknown_count: " << unknown_count;
    }
    
    cudaEventRecord(ba_pcg_pre_event_, stream);
    
    cudaMemsetAsync(pcg_r_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
    cudaMemsetAsync(pcg_M_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
    
    // Decide for a keyframe to fix its pose to fix the gauge freedom.
    // In theory it should be irrelevant which one we choose, in practice I
    // had the impression in similar cases (haven't checked for this PCG
    // implementation) that the optimization worked worse on that keyframe then.
    // Maybe some numerical issues appear there? In any case, this is the
    // reason why the keyframe to fix is chosen randomly here in every
    // iteration. This way, different keyframes will be affected, but only for
    // one iteration each, so the issue should not have any significant effect
    // then overall.
    // TODO: This will not work if keyframes are removed from the list.
    const int kFixGaugeWithKeyframeID = rand() % keyframes_.size();
    auto get_kf_pose_unknown_index = [&](int keyframe_id) {
      if (keyframe_id == kFixGaugeWithKeyframeID) {
        return kInvalidUnknownIndex;
      } else if (keyframe_id < kFixGaugeWithKeyframeID) {
        return static_cast<u32>(6 * keyframe_id);
      } else {  // if keyframe_id > kFixGaugeWithKeyframeID
        return static_cast<u32>(6 * (keyframe_id - 1));
      }
    };
    
    // PCG Init
    for (shared_ptr<Keyframe>& keyframe : keyframes_) {
      if (!keyframe) {
        continue;
      }
      
      PCGInitCUDA(
          stream,
          CreateSurfelProjectionParameters(depth_camera_, depth_params_, surfels_size_, *surfels_, keyframe.get()),
          CreateDepthToColorPixelCorner(depth_camera_, color_camera_),
          CreatePixelCenterUnprojector(depth_camera_),
          CreatePixelCornerProjector(color_camera_),
          keyframe->color_texture(),
          get_kf_pose_unknown_index(keyframe->id()),
          surfel_unknown_start_index,
          (keyframe->id() == kFixGaugeWithKeyframeID) ? false : optimize_poses,
          optimize_geometry,
          use_depth_residuals_,
          use_descriptor_residuals_,
          optimize_depth_intrinsics,
          optimize_color_intrinsics,
          depth_intrinsics_unknown_start_index,
          color_intrinsics_unknown_start_index,
          &pcg_r_->ToCUDA(),
          &pcg_M_->ToCUDA(),
          surfels_size_);
    }
    
    PCGInit2CUDA(
        stream,
        unknown_count,
        a_unknown_index,
        depth_params_.a,
        pcg_r_->ToCUDA(),
        pcg_M_->ToCUDA(),
        &pcg_delta_->ToCUDA(),
        &pcg_g_->ToCUDA(),
        &pcg_p_->ToCUDA(),
        &pcg_alpha_n_->ToCUDA());
    
    PCGScalar prev_r_norm = numeric_limits<PCGScalar>::infinity();
    int num_iterations_without_improvement = 0;
    
    for (int step = 0; step < max_inner_iterations; ++ step) {
      cudaMemsetAsync(pcg_alpha_d_->ToCUDA().address(), 0, 1 * sizeof(PCGScalar), stream);
      
      if (step > 0) {
        // Set pcg_alpha_n_ to pcg_beta_n_ by swapping the pointers (since we
        // don't need to preserve pcg_beta_n_).
        // NOTE: This is wrong in the Opt paper, it says "beta" only instead of
        //       "beta_n" which is something different.
        std::swap(pcg_alpha_n_, pcg_beta_n_);
        
        // This is cleared by PCGInit2CUDA() for the first iteration
        cudaMemsetAsync(pcg_g_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
      }
      
      // PCG Step 1
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        
        PCGStep1CUDA(
            stream,
            unknown_count,
            CreateSurfelProjectionParameters(depth_camera_, depth_params_, surfels_size_, *surfels_, keyframe.get()),
            CreateDepthToColorPixelCorner(depth_camera_, color_camera_),
            CreatePixelCenterUnprojector(depth_camera_),
            CreatePixelCornerProjector(color_camera_),
            keyframe->color_texture(),
            get_kf_pose_unknown_index(keyframe->id()),
            surfel_unknown_start_index,
            (keyframe->id() == kFixGaugeWithKeyframeID) ? false : optimize_poses,
            optimize_geometry,
            use_depth_residuals_,
            use_descriptor_residuals_,
            optimize_depth_intrinsics,
            optimize_color_intrinsics,
            depth_intrinsics_unknown_start_index,
            a_unknown_index,
            color_intrinsics_unknown_start_index,
            &pcg_p_->ToCUDA(),
            &pcg_g_->ToCUDA(),
            &pcg_alpha_d_->ToCUDA(),
            surfels_size_);
      }
      
      PCGStep2CUDA(
          stream,
          unknown_count,
          a_unknown_index,
          pcg_r_->ToCUDA(),
          pcg_M_->ToCUDA(),
          &pcg_delta_->ToCUDA(),
          &pcg_g_->ToCUDA(),
          &pcg_p_->ToCUDA(),
          &pcg_alpha_n_->ToCUDA(),
          &pcg_alpha_d_->ToCUDA(),
          &pcg_beta_n_->ToCUDA());
      
      // Check for convergence
      PCGScalar r_norm;
      pcg_beta_n_->DownloadAsync(stream, &r_norm);
      cudaStreamSynchronize(stream);
      r_norm = sqrt(r_norm);
      if (kDebug) {
        LOG(INFO) << "r_norm: " << r_norm << "; advancement: " << (prev_r_norm - r_norm);
      }
      if (r_norm < prev_r_norm - 1e-3) {  // TODO: Make this threshold a parameter
        num_iterations_without_improvement = 0;
      } else {
        ++ num_iterations_without_improvement;
        if (num_iterations_without_improvement >= 3) {
          break;
        }
      }
      prev_r_norm = r_norm;
      
      // This (and some computations from step 2) is not necessary in the last
      // iteration since the result is already computed in pcg_delta.
      // NOTE: For best speed, could make a special version of step 2 (templated)
      //       which excludes the unnecessary operations. Probably not very relevant though.
      if (step < max_inner_iterations - 1) {
        PCGStep3CUDA(
            stream,
            unknown_count,
            &pcg_g_->ToCUDA(),
            &pcg_p_->ToCUDA(),
            &pcg_alpha_n_->ToCUDA(),
            &pcg_beta_n_->ToCUDA());
      }
    }
    
    cudaEventRecord(ba_pcg_post_event_, stream);
    
    // TEST: Debug-verify the result. Show that J^T * J * pcg_delta == r0.
    constexpr bool kDebugVerifyResult = false;
    if (kDebugVerifyResult) {
      // Step 1: Compute r0 in pcg_r_
      cudaMemsetAsync(pcg_r_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
      cudaMemsetAsync(pcg_M_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        bool fix_kf_pose = keyframe->id() == kFixGaugeWithKeyframeID;
        PCGInitCUDA(
            stream,
            CreateSurfelProjectionParameters(depth_camera_, depth_params_, surfels_size_, *surfels_, keyframe.get()),
            CreateDepthToColorPixelCorner(depth_camera_, color_camera_),
            CreatePixelCenterUnprojector(depth_camera_),
            CreatePixelCornerProjector(color_camera_),
            keyframe->color_texture(),
            get_kf_pose_unknown_index(keyframe->id()),
            surfel_unknown_start_index,
            fix_kf_pose ? false : optimize_poses,
            optimize_geometry,
            use_depth_residuals_,
            use_descriptor_residuals_,
            optimize_depth_intrinsics,
            optimize_color_intrinsics,
            depth_intrinsics_unknown_start_index,
            color_intrinsics_unknown_start_index,
            &pcg_r_->ToCUDA(),
            &pcg_M_->ToCUDA(),
            surfels_size_);
      }
      
      // Step 2: Compute J^T * J * pcg_delta in pcg_g_
      cudaMemsetAsync(pcg_g_->ToCUDA().address(), 0, unknown_count * sizeof(PCGScalar), stream);
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        
        PCGStep1CUDA(
            stream,
            unknown_count,
            CreateSurfelProjectionParameters(depth_camera_, depth_params_, surfels_size_, *surfels_, keyframe.get()),
            CreateDepthToColorPixelCorner(depth_camera_, color_camera_),
            CreatePixelCenterUnprojector(depth_camera_),
            CreatePixelCornerProjector(color_camera_),
            keyframe->color_texture(),
            get_kf_pose_unknown_index(keyframe->id()),
            surfel_unknown_start_index,
            optimize_poses,
            optimize_geometry,
            use_depth_residuals_,
            use_descriptor_residuals_,
            optimize_depth_intrinsics,
            optimize_color_intrinsics,
            depth_intrinsics_unknown_start_index,
            a_unknown_index,
            color_intrinsics_unknown_start_index,
            &pcg_delta_->ToCUDA(),  // NOTE: In contrast to the optimization, we are passing delta here instead of p
            &pcg_g_->ToCUDA(),
            &pcg_alpha_d_->ToCUDA(),
            surfels_size_);
      }
      
      // Step 3: Subtract the two results (paying attention to the implicitly-added
      //         kDiagEpsilon on the diagonal of J^T J) and verify that the resulting
      //         vector is near-zero.
      PCGDebugVerifyResultCUDA(
          stream,
          unknown_count,
          a_unknown_index,
          &pcg_r_->ToCUDA(),
          &pcg_g_->ToCUDA(),
          &pcg_delta_->ToCUDA());
    }
    
    // Update the variables from pcg_delta.
    // Keyframe poses:
    usize num_converged = 0;
    if (optimize_poses) {
      vector<PCGScalar> pcg_delta_cpu(6 * (keyframes_.size() - 1));
      pcg_delta_->DownloadPartAsync(0, (6 * (keyframes_.size() - 1)) * sizeof(PCGScalar), stream, pcg_delta_cpu.data());
      cudaStreamSynchronize(stream);
      for (shared_ptr<Keyframe>& keyframe : keyframes_) {
        if (!keyframe) {
          continue;
        }
        if (keyframe->id() == kFixGaugeWithKeyframeID) {
          ++ num_converged;
          continue;
        }
        
        u32 kf_pose_unknown_index = get_kf_pose_unknown_index(keyframe->id());
        SE3f delta = SE3f::exp(Eigen::Matrix<PCGScalar, 6, 1>::Map(&pcg_delta_cpu[kf_pose_unknown_index]).cast<float>());
        keyframe->set_global_T_frame(keyframe->global_T_frame() * delta);
        
        if (IsScale1PoseEstimationConverged(delta.log())) {
          ++ num_converged;
        }
      }
    }
    
    // Surfel positions:
    if (optimize_geometry) {
      UpdateSurfelsFromPCGDeltaCUDA(
          stream,
          surfels_size_,
          &surfels_->ToCUDA(),
          use_descriptor_residuals_,
          surfel_unknown_start_index,
          pcg_delta_->ToCUDA());
    }
    
    // Depth intrinsics:
    if (optimize_depth_intrinsics) {
      // Intrinsics and global distortion parameters (fx, fy, cx, cy, a):
      PCGScalar buffer[5];
      pcg_delta_->DownloadPartAsync(
          depth_intrinsics_unknown_start_index * sizeof(PCGScalar),
          5 * sizeof(PCGScalar),
          stream, buffer);
      cudaStreamSynchronize(stream);
      
      double old_depth_fx_inv = 1. / depth_camera_.parameters()[0];
      double old_depth_fy_inv = 1. / depth_camera_.parameters()[1];
      const double old_depth_cx_pixel_center = depth_camera_.parameters()[2] - 0.5;
      const double old_depth_cy_pixel_center = depth_camera_.parameters()[3] - 0.5;
      double old_depth_cx_inv = -old_depth_cx_pixel_center * old_depth_fx_inv;
      double old_depth_cy_inv = -old_depth_cy_pixel_center * old_depth_fy_inv;
      
      double new_depth_fx = 1. / (old_depth_fx_inv + buffer[0]);
      double new_depth_fy = 1. / (old_depth_fy_inv + buffer[1]);
      double new_depth_cx = -(new_depth_fx * (old_depth_cx_inv + buffer[2])) + 0.5;
      double new_depth_cy = -(new_depth_fy * (old_depth_cy_inv + buffer[3])) + 0.5;
      float new_depth_camera_parameters[4] = {
          static_cast<float>(new_depth_fx),
          static_cast<float>(new_depth_fy),
          static_cast<float>(new_depth_cx),
          static_cast<float>(new_depth_cy)};
      depth_camera_ = PinholeCamera4f(depth_camera_.width(), depth_camera_.height(), new_depth_camera_parameters);
      
      depth_params_.a += buffer[4];
      
      // cfactor_buffer
      UpdateCFactorsFromPCGDeltaCUDA(
          stream,
          &depth_params_.cfactor_buffer,
          depth_intrinsics_unknown_start_index + 5,
          pcg_delta_->ToCUDA());
    }
    
    // Color intrinsics:
    if (optimize_color_intrinsics) {
      PCGScalar buffer[4];
      pcg_delta_->DownloadPartAsync(
          color_intrinsics_unknown_start_index * sizeof(PCGScalar),
          4 * sizeof(PCGScalar),
          stream, buffer);
      cudaStreamSynchronize(stream);
      
      float new_color_camera_parameters[4] = {
          static_cast<float>(color_camera_.parameters()[0] + buffer[0]),
          static_cast<float>(color_camera_.parameters()[1] + buffer[1]),
          static_cast<float>(color_camera_.parameters()[2] + buffer[2]),
          static_cast<float>(color_camera_.parameters()[3] + buffer[3])};
      color_camera_ = PinholeCamera4f(color_camera_.width(), color_camera_.height(), new_color_camera_parameters);
    }
    
    if ((optimize_depth_intrinsics || optimize_color_intrinsics) && intrinsics_updated_callback_) {
      intrinsics_updated_callback_();
    }
    
    // --- SURFEL MERGE ---
    // For keyframes for which new surfels were created at the start of the
    // iteration (a subset of the active keyframes).
    if (do_surfel_updates) {
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
            &surfel_count_,
            &deleted_count_buffer_);
      }
      cudaEventRecord(ba_surfel_merge_post_event_, stream);
      
      if (kDebugVerifySurfelCount) {
        DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
      }
      
      cudaEventRecord(ba_surfel_compaction_pre_event_, stream);
      if (!keyframes_with_new_surfels.empty()) {
        // Compact the surfels list to increase performance of subsequent kernel calls.
        // TODO: Only run on the new surfels if possible
        CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count_, &surfels_size_, &surfels_->ToCUDA(), &active_surfels_->ToCUDA());
      }
      cudaEventRecord(ba_surfel_compaction_post_event_, stream);
      
      if (kDebugVerifySurfelCount) {
        DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
      }
    }
    
    // Timing
    if (timings_stream_) {
      *timings_stream_ << "BA_count " << ba_iteration_count_ << " inner_iteration " << iteration << " keyframe_count " << keyframes_.size()
                       << " surfel_count " << surfel_count_ << endl;
    }
    
    // Store timings for events used within this loop.
    if (do_surfel_updates) {
      cudaEventSynchronize(ba_surfel_compaction_post_event_);
    } else {
      cudaEventSynchronize(ba_pcg_post_event_);
    }
    float elapsed_milliseconds;
    
    if (optimize_geometry && do_surfel_updates) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_surfel_creation_pre_event_, ba_surfel_creation_post_event_);
      Timing::addTime(Timing::getHandle("BA surfel creation"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_surfel_creation " << elapsed_milliseconds << endl;
      }
    }
    
    if (optimize_geometry) {
      cudaEventElapsedTime(&elapsed_milliseconds, ba_geometry_optimization_pre_event_, ba_geometry_optimization_post_event_);
      Timing::addTime(Timing::getHandle("BA normals update"), 0.001 * elapsed_milliseconds);
      if (timings_stream_) {
        *timings_stream_ << "BA_normals_update " << elapsed_milliseconds << endl;
      }
    }
    
    cudaEventElapsedTime(&elapsed_milliseconds, ba_pcg_pre_event_, ba_pcg_post_event_);
    Timing::addTime(Timing::getHandle("BA PCG step"), 0.001 * elapsed_milliseconds);
    if (timings_stream_) {
      *timings_stream_ << "BA_PCG_step " << elapsed_milliseconds << endl;
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
    
    // Test for convergence
    // TODO: This criterion does not work if for example optimizing for intrinsics only
    if (iteration >= min_iterations - 1 &&
        (num_converged == keyframes_.size() || !optimize_poses)) {
      // All frames are inactive. Early exit.
      LOG(INFO) << "Debug: early convergence in PCG-based BA at iteration: " << iteration;
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
  }  // end loop over optimization iterations
  
  
  if (increase_ba_iteration_count) {
    PerformBASchemeEndTasks(
        stream,
        do_surfel_updates);
    
    ++ ba_iteration_count_;
    
    if (ba_iteration_count_ % 10 == 0) {
      LOG(INFO) << Timing::print(kSortByTotal);
    }
  } else if (do_surfel_updates) {
    // Surfel merging for newly activated keyframes
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
          &surfel_count_,
          &deleted_count_buffer_);
    }
    cudaEventRecord(ba_surfel_merge_post_event_, stream);
    
    if (kDebugVerifySurfelCount) {
      DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
    }
    
    cudaEventRecord(ba_surfel_compaction_pre_event_, stream);
    if (!keyframes_with_new_surfels.empty()) {
      // Compact the surfels list to increase performance of subsequent kernel calls.
      // TODO: Only run on the new surfels if possible
      CompactSurfelsCUDA(stream, &free_spots_temp_storage_, &free_spots_temp_storage_bytes_, surfel_count_, &surfels_size_, &surfels_->ToCUDA(), &active_surfels_->ToCUDA());
    }
    cudaEventRecord(ba_surfel_compaction_post_event_, stream);
    
    if (kDebugVerifySurfelCount) {
      DebugVerifySurfelCount(stream, surfel_count_, surfels_size_, *surfels_);
    }
  }
  
  UpdateBAVisualization(stream);
}

}
