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


// Parts of this file are based on the demo code of DLoopDetector, which has
// the following license:
// 
// DLoopDetector: loop detector for monocular image sequences
// 
// Copyright (c) 2015 Dorian Galvez-Lopez. http://doriangalvez.com
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. The original author of the work must be notified of any 
//    redistribution of source code or in binary form.
// 4. Neither the name of copyright holders nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
// 
// If you use it in an academic work, please cite:
// 
//   @ARTICLE{GalvezTRO12,
//     author={G\'alvez-L\'opez, Dorian and Tard\'os, J. D.}, 
//     journal={IEEE Transactions on Robotics},
//     title={Bags of Binary Words for Fast Place Recognition in Image Sequences},
//     year={2012},
//     month={October},
//     volume={28},
//     number={5},
//     pages={1188--1197},
//     doi={10.1109/TRO.2012.2197158},
//     ISSN={1552-3098}
// }


#include "badslam/loop_detector.h"

#include <opengv/point_cloud/methods.hpp>
#include <opengv/point_cloud/PointCloudAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>

#include "badslam/cuda_image_processing.cuh"
#include "badslam/pairwise_frame_tracking.h"
#include "badslam/pose_graph_optimizer.h"
#include "badslam/surfel_projection.h"
#include "badslam/trajectory_deformation.h"
#include "badslam/util.cuh"
#include "badslam/util.h"

namespace vis {

BriefExtractor::BriefExtractor(const std::string &pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // Loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw string("Could not open file ") + pattern_file;
  }
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator() (
    const cv::Mat &im, 
    vector<cv::KeyPoint> &keys,
    vector<DVision::BRIEF::bitset> &descriptors) const {
  // Extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  
  // Compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}


// -----------------------------------------------------------------------------


LoopDetector:: LoopDetector(
    const string& vocabulary_path,
    const string& pattern_path,
    int image_width,
    int image_height,
    float raw_to_float_depth,
    int depth_image_width,
    int depth_image_height,
    int num_scales,
    float image_frequency,
    bool parallel_loop_detection)
    : pairwise_tracking_buffers_(depth_image_width,
                                 depth_image_height,
                                 num_scales) {
  raw_to_float_depth_ = raw_to_float_depth;
  
  // Set loop detector parameters
  typename TDetector::Parameters params(image_width, image_height, image_frequency);
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  params.use_nss = true;  // use normalized similarity score instead of raw score
  params.alpha = 0.15;  // nss threshold
  params.k = 1;  // a loop must be consistent with 1 previous matches
  // NOTE: The case DLoopDetector::GEOM_EXHAUSTIVE was modified in DLoopDetector
  //       to return the keypoint matches; if using another case, it has to be
  //       modified as well.
  params.geom_check = DLoopDetector::GEOM_EXHAUSTIVE;  // use direct index for geometrical checking.
  params.di_levels = 2;  // use two direct index levels
  
  // Load the vocabulary to use
  LOG(INFO) << "Loop detector: Loading vocabulary (from " << vocabulary_path << ") ...";
  TVocabulary voc(vocabulary_path);  // throws exception if file not found
  
  // Initiate loop detector with the vocabulary 
  detector_.reset(new TDetector(voc, params));
  
  // Optionally allocate memory for the expected number of images
  detector_->allocate(2500);
  
  extractor_.reset(new TExtractor(pattern_path));
  
  if (parallel_loop_detection) {
    // Start a separate thread for loop detection.
    quit_requested_ = false;
    quit_done_ = false;
    detection_thread_.reset(new thread(std::bind(&LoopDetector::DetectionThreadMain, this)));
  }
}

LoopDetector::~LoopDetector() {
  if (detection_thread_) {
    // Signal to the loop detection thread that it should exit
    unique_lock<mutex> lock(detection_thread_mutex_);
    quit_requested_ = true;
    lock.unlock();
    zero_images_condition_.notify_all();
    
    // Wait for the thread to exit
    unique_lock<mutex> quit_lock(quit_mutex_);
    while (!quit_done_) {
      quit_condition_.wait(quit_lock);
    }
    quit_lock.unlock();
    
    detection_thread_->join();
  }
}

bool LoopDetector::AddImage(
    cudaStream_t stream,
    u32 start_frame,
    u32 end_frame,
    RGBDVideo<Vec3u8, u16>* rgbd_video,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    const PinholeCamera4f& gray_camera,
    const cv::Mat_<u8>& gray_image,
    const shared_ptr<Image<u16>>& depth_image,
    const shared_ptr<BadSlamRenderWindow>& render_window,
    const Keyframe& current_keyframe,
    PairwiseFrameTrackingBuffers* /*pairwise_tracking_buffers*/,
    DirectBA* direct_ba) {
  // Set this to true to get some debug output
  constexpr bool kDebug = false;
  
  // TODO: Make configurable. Whether to use gradient magnitudes for direct tracking,
  //       or separate x/y gradient components.
  constexpr bool use_gradmag = false;
  
  // Verify assumptions on image sizes
  if (!gray_image.empty() && depth_image) {
    CHECK_EQ(gray_image.cols, depth_image->width());
    CHECK_EQ(gray_image.rows, depth_image->height());
  }
  
  if (depth_image) {
    CHECK_EQ(depth_image->width(), gray_camera.width());
    CHECK_EQ(depth_image->height(), gray_camera.height());
  }
  
  // Detect loops (or wait for / use the stored result in case of doing loop
  // detection in parallel)
  DLoopDetector::DetectionResult result;
  vector<cv::KeyPoint> keys;
  vector<cv::KeyPoint> old_keypoints;
  vector<cv::KeyPoint> cur_keypoints;
  if (detection_thread_) {
    unique_lock<mutex> lock(detection_result_mutex_);
    while (detection_results_.empty()) {
      detection_result_condition_.wait(lock);
    }
    
    result = detection_results_.front();
    keys = std::move(detected_keys_.front());
    old_keypoints = std::move(detected_old_keypoints_.front());
    cur_keypoints = std::move(detected_cur_keypoints_.front());
    
    detection_results_.erase(detection_results_.begin());
    detected_keys_.erase(detected_keys_.begin());
    detected_old_keypoints_.erase(detected_old_keypoints_.begin());
    detected_cur_keypoints_.erase(detected_cur_keypoints_.begin());
    
    lock.unlock();
    
    if (!result.detection()) {
      return false;
    }
  } else {
    if (!DetectLoop(gray_image, depth_image, &result, &keys, &old_keypoints, &cur_keypoints)) {
      return false;
    }
  }
  
  usize num_matches = old_keypoints.size();
  
  static int loop_count = 0;
  ++ loop_count;
  LOG(WARNING) << "- Loop found with image " << result.match << "! (" << loop_count << " loops found so far)";
  
  // Try to estimate the rough relative pose using the matched feature points.
  // First, convert the keypoints to 3D points in their keyframe's local
  // coordinate system.
  // TODO: No depth deformation calibration or bilinear filtering is used here!
  opengv::points_t old_points;
  old_points.reserve(num_matches);
  opengv::points_t cur_points;
  cur_points.reserve(num_matches);
  
  for (usize i = 0, size = num_matches; i < size; ++ i) {
    if (old_keypoints[i].response == 0 ||
        cur_keypoints[i].response == 0) {
      // If the depth for one or both of the keypoints is missing, do not use this match.
    } else {
      old_points.emplace_back((old_keypoints[i].response * gray_camera.UnprojectFromPixelCornerConv(Vec2f(old_keypoints[i].pt.x, old_keypoints[i].pt.y))).cast<double>());
      cur_points.emplace_back((cur_keypoints[i].response * gray_camera.UnprojectFromPixelCornerConv(Vec2f(cur_keypoints[i].pt.x, cur_keypoints[i].pt.y))).cast<double>());
    }
  }
  
  // Alignment using 3D-3D matches
  opengv::point_cloud::PointCloudAdapter adapter(old_points, cur_points);
  
  // Create a RANSAC object
  opengv::sac::Ransac<opengv::sac_problems::point_cloud::PointCloudSacProblem> ransac;
  
  // Create the sample consensus problem
  std::shared_ptr<opengv::sac_problems::point_cloud::PointCloudSacProblem>
      relposeproblem_ptr(new opengv::sac_problems::point_cloud::PointCloudSacProblem(adapter));
  
  // Run RANSAC
  constexpr float kRANSACThreshold = 0.06f;  // TODO: Make parameter; tune?
  constexpr float kRANSACMaxIterations = 500;  // TODO: Make parameter; tune?
  constexpr int kRANSACMinInliers = 10;  // TODO: Make parameter; tune?
  
  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = kRANSACThreshold;
  ransac.max_iterations_ = kRANSACMaxIterations;
  ransac.computeModel(/*debug_verbosity_level*/ 0);
  
  // Verify the result
  if (ransac.inliers_.size() < kRANSACMinInliers) {
    LOG(INFO) << "--> Rejecting loop closure since RANSAC on relative pose with 3D-3D matches did not find enough inliers (" << ransac.inliers_.size() << " < " << kRANSACMinInliers << ")";
    
    // NOTE: Alternative for 2D-2D correspondences. Could be tried if the point cloud version does not work (due to missing / wrong depth).
    //     // Nister's 5-point algorithm
    //     // create the central relative adapter
    //     relative_pose::CentralRelativeAdapter adapter(
    //         bearingVectors1, bearingVectors2 );
    //     
    //     // create a RANSAC object
    //     sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem> ransac;
    //     
    //     // create a CentralRelativePoseSacProblem
    //     // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
    //     std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
    //         relposeproblem_ptr(
    //         new sac_problems::relative_pose::CentralRelativePoseSacProblem(
    //         adapter,
    //         sac_problems::relative_pose::CentralRelativePoseSacProblem::NISTER ) );
    //     
    //     // run ransac
    //     ransac.sac_model_ = relposeproblem_ptr;
    //     ransac.threshold_ = threshold;
    //     ransac.max_iterations_ = maxIterations;
    //     ransac.computeModel();
    //     
    //     // get the result
    //     transformation_t best_transformation = ransac.model_coefficients_;
    
    if (kDebug) {
      std::getchar();
    }
    return false;
  }
  
  LOG(INFO) << "- loop closure 3D-3D inlier count: " << ransac.inliers_.size();
  
  opengv::transformation_t best_transformation = ransac.model_coefficients_;
  SE3f old_T_cur_initial =
      SE3f(best_transformation.block<3, 3>(0, 0).cast<float>(),
           best_transformation.block<3, 1>(0, 3).cast<float>());
  
  // DEBUG: Show the estimated old pose constructed by concatenating the current pose and the relative pose estimate.
  constexpr bool kDebugInitialPoseEstimate = false;
  if (kDebugInitialPoseEstimate && render_window) {
    SE3f old_T_global = old_T_cur_initial * current_keyframe.frame_T_global();
    render_window->SetCurrentFramePose(old_T_global.inverse().matrix());
    
    shared_ptr<Point3fC3u8Cloud> debug_frame_cloud(new Point3fC3u8Cloud(cur_points.size() + old_points.size()));
    
    const SE3f& global_T_current_kf = current_keyframe.global_T_frame();
    for (usize i = 0; i < cur_points.size(); ++ i) {
      Point3fC3u8& point = debug_frame_cloud->at(i);
      point.position() = global_T_current_kf * cur_points[i].cast<float>();  // NOTE: slow transformation
      point.color() = Vec3u8(255, 80, 80);
    }
    
    SE3f global_T_old_kf_estimate = old_T_global.inverse();
    for (usize i = 0; i < old_points.size(); ++ i) {
      Point3fC3u8& point = debug_frame_cloud->at(cur_points.size() + i);
      point.position() = global_T_old_kf_estimate * old_points[i].cast<float>();  // NOTE: slow transformation
      point.color() = Vec3u8(80, 80, 255);
    }
    
    render_window->SetFramePointCloud(
        debug_frame_cloud,
        SE3f());
    
    render_window->RenderFrame();
    LOG(INFO) << "Showing estimated old frame pose in loop closure";
    std::getchar();
    
    render_window->UnsetFramePointCloud();
  }
  
  // Refine the relative pose estimate using direct frame-to-frame alignment.
  // First, convert the raw u16 depths of the current frame to calibrated float
  // depths and transform the color image to depth intrinsics (and image size)
  // such that the code from the multi-res odometry tracking can be re-used
  // which expects these inputs.
  if (!calibrated_depth_) {
    CreatePairwiseTrackingInputBuffersAndTextures(
        current_keyframe.depth_buffer().width(),
        current_keyframe.depth_buffer().height(),
        current_keyframe.color_buffer().width(),
        current_keyframe.color_buffer().height(),
        &calibrated_depth_,
        &calibrated_gradmag_,
        &base_kf_gradmag_,
        &tracked_gradmag_,
        &calibrated_gradmag_texture_,
        &base_kf_gradmag_texture_,
        &tracked_gradmag_texture_);
  }
  
  if (use_gradmag) {
    ComputeSobelGradientMagnitudeCUDA(
        stream,
        current_keyframe.color_texture(),
        &base_kf_gradmag_->ToCUDA());
  } else {
    ComputeBrightnessCUDA(
        stream,
        current_keyframe.color_texture(),
        &base_kf_gradmag_->ToCUDA());
  }
  
  CalibrateDepthAndTransformColorToDepthCUDA(
      stream,
      CreateDepthToColorPixelCorner(depth_camera, color_camera),
      depth_params,
      current_keyframe.depth_buffer().ToCUDA(),
      base_kf_gradmag_texture_,
      &calibrated_depth_->ToCUDA(),
      &calibrated_gradmag_->ToCUDA());
  
  // Refine relative pose estimate using the matched old keyframe, as well as
  // using its previous and next keyframe (individually), to get three estimates
  // of the refined pose.
  SE3f old_T_cur_refined[3];
  SE3f cur_T_old_refined[3];
  Keyframe* old_keyframes[3];
  
  // Get matched keyframe
  Keyframe* matched_keyframe = direct_ba->keyframes()[result.match].get();
  if (!matched_keyframe) {
    LOG(FATAL) << "Loop detection returned a deleted keyframe, this should not happen!";
  }
  old_keyframes[0] = matched_keyframe;
  // matched_T_this[0] = identity;
  
  // Get next keyframe
  // NOTE: Since the loop detector never returns a recent keyframe,
  //       we do not need to check the case that the 'next keyframe' could be
  //       the current one.
  usize next_keyframe_index;
  old_keyframes[1] = nullptr;
  for (usize i = result.match + 1; i < direct_ba->keyframes().size(); ++ i) {
    if (direct_ba->keyframes()[i]) {
      old_keyframes[1] = direct_ba->keyframes()[i].get();
      next_keyframe_index = i;
      break;
    }
  }
  if (!old_keyframes[1]) {
    LOG(INFO) << "--> Rejecting loop closure since no 'next keyframe' found for verification.";
    if (kDebug) {
      std::getchar();
    }
    return false;
  }
  
  // Get previous keyframe
  old_keyframes[2] = nullptr;
  for (int i = static_cast<int>(result.match) - 1; i >= 0; -- i) {
    if (direct_ba->keyframes()[i]) {
      old_keyframes[2] = direct_ba->keyframes()[i].get();
      break;
    }
  }
  if (!old_keyframes[2]) {
    // The matched keyframe was the first one. Use another later keyframe instead of a previous one for verification.
    for (usize i = next_keyframe_index + 1; i < direct_ba->keyframes().size(); ++ i) {
      if (direct_ba->keyframes()[i]) {
        old_keyframes[2] = direct_ba->keyframes()[i].get();
        break;
      }
    }
    
    if (!old_keyframes[2]) {
      LOG(INFO) << "--> Rejecting loop closure since no second keyframe found for verification.";
      if (kDebug) {
        std::getchar();
      }
      return false;
    }
  }
  
  SE3f cur_T_tracked[3];
  for (int i = 0; i < 3; ++ i) {
    SE3f matched_T_this = (i == 0) ? SE3f() : (old_keyframes[0]->frame_T_global() * old_keyframes[i]->global_T_frame());
    
    if (use_gradmag) {
      ComputeSobelGradientMagnitudeCUDA(
          stream,
          old_keyframes[i]->color_texture(),
          &tracked_gradmag_->ToCUDA());
    } else {
      ComputeBrightnessCUDA(
          stream,
          old_keyframes[i]->color_texture(),
          &tracked_gradmag_->ToCUDA());
    }
    
    // base = current, tracked = matched / next / prev
    SE3f base_T_tracked_initial_estimate = old_T_cur_initial.inverse() * matched_T_this;
    TrackFramePairwise(
        &pairwise_tracking_buffers_,
        stream,
        direct_ba->color_camera(),
        direct_ba->depth_camera(),
        direct_ba->depth_params(),
        *direct_ba->cfactor_buffer(),
        &pose_estimation_helper_buffers_,
        render_window,
        /*kGatherConvergenceSamples ? &convergence_samples_file_ :*/ nullptr,
        direct_ba->use_depth_residuals(),
        direct_ba->use_descriptor_residuals(),
        /*use_pyramid_level_0*/ true,
        use_gradmag,
        /* tracked frame */
        old_keyframes[i]->depth_buffer(),
        old_keyframes[i]->normals_buffer(),
        tracked_gradmag_texture_,
        /* base frame */
        *calibrated_depth_,
        current_keyframe.normals_buffer(),
        *calibrated_gradmag_,
        calibrated_gradmag_texture_,
        /* input / output poses */
        current_keyframe.global_T_frame(),
        /*test_different_initial_estimates*/ false,
        base_T_tracked_initial_estimate,
        base_T_tracked_initial_estimate,
        &cur_T_tracked[i]);
    
    old_T_cur_refined[i] = matched_T_this * cur_T_tracked[i].inverse();
    cur_T_old_refined[i] = old_T_cur_refined[i].inverse();
  }
  
  // DEBUG: Display refined poses
  constexpr bool kDebugRefinedPoseEstimates = false;
  if (kDebugRefinedPoseEstimates && render_window) {
    LOG(INFO) << "Showing refined old frame pose estimates in loop closure";
    LOG(INFO) << "Press return to show next refined estimate, press q + return to exit this debug visualization";
    
    int i = 0;
    while (true) {
      LOG(INFO) << "Showing estimate " << (i % 3);
      
      SE3f old_T_global = old_T_cur_refined[i % 3] * current_keyframe.frame_T_global();
      render_window->SetCurrentFramePose(old_T_global.inverse().matrix());
      
      render_window->RenderFrame();
      
      int key = std::getchar();
      if (key == 'q') {
        std::getchar();  // remove the return following the q from the queue
        break;
      }
      
      ++ i;
    }
  }
  
  // Verify that the refined pose estimates are similar. Reject the loop closure
  // if they are not.
  constexpr float kMaxAngleDifference = M_PI / 180.f * 10.f;
  constexpr float kMaxEuclideanDistance = 0.02f;
  
  for (int i = 0; i < (3 - 1); ++ i) {
    for (int k = i + 1; k < 3; ++ k) {
      float rotational_distance = acosf(std::min(1.f, std::max(-1.f, cur_T_old_refined[i].rotationMatrix().block<3, 1>(0, 2).dot(
                                                                     cur_T_old_refined[k].rotationMatrix().block<3, 1>(0, 2)))));
      if (rotational_distance > kMaxAngleDifference) {
        LOG(INFO) << "--> Rejecting loop closure since rotational_distance (" << rotational_distance << ") > kMaxAngleDifference (" << kMaxAngleDifference << ")";
        if (kDebug) {
          std::getchar();
        }
        return false;
      }
      
      float translational_distance = (cur_T_old_refined[i].translation() - cur_T_old_refined[k].translation()).norm();
      if (translational_distance > kMaxEuclideanDistance) {
        LOG(INFO) << "--> Rejecting loop closure since translational_distance (" << translational_distance << ") > kMaxEuclideanDistance (" << kMaxEuclideanDistance << ")";
        if (kDebug) {
          std::getchar();
        }
        return false;
      }
      
      LOG(INFO) << "- rotational_distance (" << i << ", " << k << "): " << rotational_distance;
      LOG(INFO) << "- translational_distance (" << i << ", " << k << "): " << translational_distance;
    }
  }
  
  // Average the refined relative pose estimates.
  // NOTE: A slower but potentially better alternative would be to repeat the
  //       pose estimation while using all 3 frames simultaneously.
  Sophus::SE3f cur_T_old_averaged = AveragePose(3, cur_T_old_refined);
  
  // DEBUG: Display averaged pose
  constexpr bool kDebugAveragedPoseEstimate = false;
  if (kDebugAveragedPoseEstimate && render_window) {
    LOG(INFO) << "Showing averaged old frame pose estimate in loop closure";
    
    SE3f old_T_global = cur_T_old_averaged.inverse() * current_keyframe.frame_T_global();
    render_window->SetCurrentFramePose(old_T_global.inverse().matrix());
    
    render_window->RenderFrame();
    
    std::getchar();
  }
  
  // Decide whether performing a loop closure is necessary, or whether the
  // estimated and actual relative pose are close enough to let the BA do it.
  // Check how many pixels some sub-selected depth points (the matched keypoints) would
  // move (measured in the pyramid level used for color images) if the keyframe
  // was moved. If they would move more than (half?) a pixel, it may be outside the
  // convergence region for BA, so a loop closure must be performed.
  SE3f cur_T_global_estimate = cur_T_old_averaged * matched_keyframe->frame_T_global();
  const SE3f& global_T_cur_actual = current_keyframe.global_T_frame();
  
  SE3f cur_estimate_TR_cur_actual =
      cur_T_global_estimate * global_T_cur_actual;
  Vec3f cur_estimate_T_cur_actual = cur_estimate_TR_cur_actual.translation();
  Mat3f cur_estimate_R_cur_actual = cur_estimate_TR_cur_actual.rotationMatrix();
  
  float distance_sum = 0.f;
  int distance_count = 0;
  
  for (usize i = 0, size = cur_keypoints.size(); i < size; ++ i) {
    // cur_keypoints[i]->pt gives the position in the current pose (however, potentially with different intrinsics than we are interested in)
    // cur_points[i] is the corresponding unprojected local 3D point
    Vec3f cur_point_at_estimate = cur_estimate_R_cur_actual * cur_points[i].cast<float>() + cur_estimate_T_cur_actual;
    
    Vec2f proj_estimate;
    if (color_camera.ProjectToPixelCornerConvIfVisible(cur_point_at_estimate, 0.f, &proj_estimate)) {
      Vec2f proj_current;
      if (color_camera.ProjectToPixelCornerConvIfVisible(cur_points[i].cast<float>(), 0.f, &proj_current)) {
        distance_sum += (proj_estimate - proj_current).norm();
        ++ distance_count;
      }
    }
  }
  
  constexpr float kAveragePixelDistanceThreshold = 1.0f;  // TODO: Make parameter
  
  float average_pixel_distance = distance_sum / distance_count;
  if (distance_count >= 5 && average_pixel_distance <= kAveragePixelDistanceThreshold) {
    LOG(INFO) << "--> Ignoring loop closure since the average pixel distance expected from performing it ("
              << average_pixel_distance << ") is less than kAveragePixelDistanceThreshold (" << kAveragePixelDistanceThreshold << ").";
    if (kDebug) {
      std::getchar();
    }
    return false;
  }
  
  LOG(INFO) << "- average_pixel_distance: " << average_pixel_distance;
  
  // Close the loop using pose-graph optimization.
  direct_ba->Lock();
  PoseGraphOptimizer optimizer(direct_ba, /*add_current_state_odometry_constraints*/ true);
  direct_ba->Unlock();
  
  // Add loop closure constraint
  // TODO: Also consider any potential previous loop closure constraints?
  // SE3f transformation = current_keyframe.frame_T_global() * matched_keyframe->global_T_frame();
  optimizer.AddEdge(current_keyframe.id(), matched_keyframe->id(), cur_T_old_averaged);
  
  // Optimize the graph
  optimizer.Optimize();
  
  // Apply the trajectory change to keyframes and non-keyframes
  direct_ba->Lock();
  
  vector<SE3f> original_keyframe_T_global;
  RememberKeyframePoses(direct_ba, &original_keyframe_T_global);
  
  for (usize i = 0; i < direct_ba->keyframes().size(); ++ i) {
    Keyframe* keyframe = direct_ba->keyframes()[i].get();
    if (!keyframe) {
      continue;
    }
    
    keyframe->set_global_T_frame(optimizer.GetGlobalTFrame(keyframe->id()));
  }
  
  ExtrapolateAndInterpolateKeyframePoseChanges(
      start_frame,
        end_frame,
      direct_ba,
      original_keyframe_T_global,
      rgbd_video);
  
  direct_ba->Unlock();
  
  if (kDebug) {
    std::getchar();
  }
  return true;
}

void LoopDetector::RemoveImage(int id) {
  detector_->removeImage(id);
}

void LoopDetector::QueueForLoopDetection(
    const cv::Mat_<u8>& image,
    const shared_ptr<Image<u16>>& depth_image) {
  detection_thread_mutex_.lock();
  
  parallel_image_queue_.push_back(image);
  parallel_depth_image_queue_.push_back(depth_image);
  
  detection_thread_mutex_.unlock();
  zero_images_condition_.notify_all();
}

bool LoopDetector::DetectLoop(
    const cv::Mat_<u8>& image,
    const shared_ptr<Image<u16>>& depth_image,
    DLoopDetector::DetectionResult* result,
    vector<cv::KeyPoint>* keys,
    vector<cv::KeyPoint>* old_keypoints,
    vector<cv::KeyPoint>* cur_keypoints) {
  // Enable to get debug output
  constexpr bool kDebug = false;
  
  vector<TDescriptor> descriptors;
  
  // Extract features
  (*extractor_)(image, *keys, descriptors);
  
  // Amend the extracted features with their depth.
  // HACK: Storing the depth in the "response" field of cv::KeyPoint.
  for (usize i = 0, size = keys->size(); i < size; ++ i) {
    // TODO: This (and UnprojectFromPixelCornerConv() used later) assumes that
    //       the OpenCV keypoints use the "pixel corner"
    //       origin convention, which I am not sure about.
    // TODO: This should ideally take the potential difference in the color and
    //       depth camera intrinsics into account.
    int x = std::max<int>(0, std::min<int>(depth_image->width() - 1, (*keys)[i].pt.x));
    int y = std::max<int>(0, std::min<int>(depth_image->height() - 1, (*keys)[i].pt.y));
    
    float depth = raw_to_float_depth_ * (*depth_image)(x, y);
    (*keys)[i].response = depth;
  }
  
  // Add the image to the collection and check if a loop is detected.
  // DLoopDetector was modified to return the potentially matching keypoints in
  // old_keypoints and cur_keypoints for the case of params.geom_check ==
  // DLoopDetector::GEOM_DI || params.geom_check == DLoopDetector::GEOM_EXHAUSTIVE.
  LockDetectorMutex();
  detector_->detectLoop(*keys, descriptors, *result, old_keypoints, cur_keypoints);
  UnlockDetectorMutex();
  
  if (kDebug) {
    // LOG(INFO) << "Loop detection: " << (result->detection() ? "true" : "false");
    
    string loop_status;
    switch (result->status) {
    case DLoopDetector::LOOP_DETECTED:              loop_status = "LOOP_DETECTED: A loop was detected";                                   break;
    case DLoopDetector::CLOSE_MATCHES_ONLY:         loop_status = "CLOSE_MATCHES_ONLY: All the matches are very recent";                  break;
    case DLoopDetector::NO_DB_RESULTS:              loop_status = "NO_DB_RESULTS: No matches against the database";                       break;
    case DLoopDetector::LOW_NSS_FACTOR:             loop_status = "LOW_NSS_FACTOR: Score of current image against previous one too low";  break;
    case DLoopDetector::LOW_SCORES:                 loop_status = "LOW_SCORES: Scores (or NS Scores) were below the alpha threshold";     break;
    case DLoopDetector::NO_GROUPS:                  loop_status = "NO_GROUPS: Not enough matches to create groups";                       break;
    case DLoopDetector::NO_TEMPORAL_CONSISTENCY:    loop_status = "NO_TEMPORAL_CONSISTENCY: Not enough temporary consistent matches (k)"; break;
    case DLoopDetector::NO_GEOMETRICAL_CONSISTENCY: loop_status = "NO_GEOMETRICAL_CONSISTENCY: The geometrical consistency failed";       break;
    }
    LOG(INFO) << "Loop status: " << loop_status;
  }
  
  if (!result->detection()) {
    return false;
  }
  
  CHECK_GT(old_keypoints->size(), 0u);
  CHECK_EQ(old_keypoints->size(), cur_keypoints->size());
  return true;
}

void LoopDetector::DetectionThreadMain() {
  while (true) {
    unique_lock<mutex> lock(detection_thread_mutex_);
    
    while (parallel_image_queue_.empty() && !quit_requested_) {
      zero_images_condition_.wait(lock);
    }
    if (quit_requested_) {
      break;
    }
    
    // Pop item from parallel_image_queue_ and parallel_depth_image_queue_
    cv::Mat_<u8> image = parallel_image_queue_.front();
    parallel_image_queue_.erase(parallel_image_queue_.begin());
    shared_ptr<Image<u16>> depth_image = parallel_depth_image_queue_.front();
    parallel_depth_image_queue_.erase(parallel_depth_image_queue_.begin());
    
    lock.unlock();
    
    // Detect loops
    DLoopDetector::DetectionResult result;
    vector<cv::KeyPoint> keys;
    vector<cv::KeyPoint> old_keypoints;
    vector<cv::KeyPoint> cur_keypoints;
    DetectLoop(image, depth_image, &result, &keys, &old_keypoints, &cur_keypoints);
    
    detection_result_mutex_.lock();
    detection_results_.push_back(result);
    detected_keys_.push_back(keys);
    detected_old_keypoints_.push_back(old_keypoints);
    detected_cur_keypoints_.push_back(cur_keypoints);
    detection_result_mutex_.unlock();
    detection_result_condition_.notify_all();
  }
  
  unique_lock<mutex> quit_lock(quit_mutex_);
  quit_done_ = true;
  quit_lock.unlock();
  quit_condition_.notify_all();
}

}
