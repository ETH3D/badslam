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


#pragma once

#include <cuda_runtime.h>

#include <DBoW2/DBoW2.h>
#include <DLoopDetector/DLoopDetector.h>
#include <DVision/DVision.h>
#include <DVision/BRIEF.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "badslam/kernels.h"
#include "badslam/direct_ba.h"
#include "badslam/pairwise_frame_tracking.h"
#include "badslam/render_window.h"
#include "badslam/keyframe.h"

namespace vis {

struct PairwiseFrameTrackingBuffers;


/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  virtual ~FeatureExtractor() = default;
  
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(
      const cv::Mat& im,
      vector<cv::KeyPoint>& keys,
      vector<TDescriptor>& descriptors) const = 0;
};

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  virtual ~BriefExtractor() = default;
  
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(
      const cv::Mat& im, 
      vector<cv::KeyPoint>& keys,
      vector<DVision::BRIEF::bitset>& descriptors) const;
  
  /**
   * Creates the brief extractor with the given pattern file
   */
  BriefExtractor(const std::string& pattern_file);

private:
  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};

// Detects loops using DLoopDetector. If a loop is detected, verifies it using
// direct pose estimation and distorts the estimated trajectory to close the
// loop.
class LoopDetector {
 public:
  typedef BriefVocabulary TVocabulary;
  typedef BriefLoopDetector TDetector;
  typedef FBrief::TDescriptor TDescriptor;
  typedef BriefExtractor TExtractor;
  
  // Constructor. The paths to the vocabulary .voc and pattern .yml file must be
  // given, as well as the dimensions of the images that will be used for loop
  // closure detection. If parallel_loop_detection is true, a separate thread
  // will be used for loop detection (but not loop closing). In this case, new
  // images have to be queued with QueueForLoopDetection() first before calling
  // AddImage(). Otherwise (i.e., in the sequential case), only AddImage() must
  // be called.
  LoopDetector(
      const string& vocabulary_path,
      const string& pattern_path,
      int image_width,
      int image_height,
      float raw_to_float_depth,
      int depth_image_width,
      int depth_image_height,
      int num_scales,
      float image_frequency,
      bool parallel_loop_detection);
  
  // If parallel loop detection is enabled, waits for the loop detection thread
  // to exit.
  ~LoopDetector();
  
  // Adds an image to the loop detector. Detects loops (or retrieves the
  // detection result in the parallel-detection case), verifies the loop, and
  // closes the loop if verification was successful. Returns true if a loop
  // closure was performed, false otherwise. depth_image can be null and
  // gray_image can be invalid if the image was added before with
  // QueueForLoopDetection().
  bool AddImage(
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
      const shared_ptr<BadSlamRenderWindow>& render_window, // TODO: for debugging only
      const Keyframe& current_keyframe,
      PairwiseFrameTrackingBuffers* pairwise_tracking_buffers,
      DirectBA* dense_ba);
  
  // Removes an image from the loop detector (such that further loop detection
  // requests will not consider it anymore). If loop detection runs in parallel,
  // then the detector mutex must be locked when RemoveImage() is called.
  void RemoveImage(int id);
  
  // Queues an image for loop detection in the case of running loop detection in
  // parallel. Later, AddImage() has to be called for the same image. The order
  // in which the images are passed to QueueForLoopDetection() and AddImage()
  // must be the same (it is used to associate loop detection results with added
  // images).
  void QueueForLoopDetection(
      const cv::Mat_<u8>& image,
      const shared_ptr<Image<u16>>& depth_image);
  
  inline void LockDetectorMutex() { detector_mutex_.lock(); }
  inline void UnlockDetectorMutex() { detector_mutex_.unlock(); }
  
 private:
  // TODO: This function is misnamed (just as in the underlying library) since
  //       it also adds the image to the collection.
  bool DetectLoop(
      const cv::Mat_<u8>& image,
      const shared_ptr<Image<u16>>& depth_image,
      DLoopDetector::DetectionResult* result,
      vector<cv::KeyPoint>* keys,
      vector<cv::KeyPoint>* old_keypoints,
      vector<cv::KeyPoint>* cur_keypoints);
  
  // Main function of the thread which runs loop detection in parallel. This
  // is only used if loop detection is configured to run in parallel.
  void DetectionThreadMain();
  
  
  CUDABufferPtr<float> calibrated_depth_;
  CUDABufferPtr<uchar> calibrated_gradmag_;
  CUDABufferPtr<uchar> base_kf_gradmag_;
  CUDABufferPtr<uchar> tracked_gradmag_;
  cudaTextureObject_t calibrated_gradmag_texture_;
  cudaTextureObject_t base_kf_gradmag_texture_;
  cudaTextureObject_t tracked_gradmag_texture_;
  
  PairwiseFrameTrackingBuffers pairwise_tracking_buffers_;
  PoseEstimationHelperBuffers pose_estimation_helper_buffers_;
  
  std::mutex detector_mutex_;
  unique_ptr<TDetector> detector_;
  unique_ptr<TExtractor> extractor_;
  
  // Parallel loop detection.
  std::atomic<bool> quit_requested_;
  std::atomic<bool> quit_done_;
  std::mutex quit_mutex_;
  condition_variable quit_condition_;
  unique_ptr<thread> detection_thread_;
  std::mutex detection_thread_mutex_;
  condition_variable zero_images_condition_;
  
  vector<cv::Mat_<u8>> parallel_image_queue_;
  vector<shared_ptr<Image<u16>>> parallel_depth_image_queue_;
  
  std::mutex detection_result_mutex_;
  std::condition_variable detection_result_condition_;
  vector<DLoopDetector::DetectionResult> detection_results_;
  // NOTE on the three members below: Vectors of vectors with dynamic allocation
  //      of the outer vectors are not a good idea performance-wise. I don't
  //      think it matters here, but in principle they should be avoided.
  vector<vector<cv::KeyPoint>> detected_keys_;
  vector<vector<cv::KeyPoint>> detected_old_keypoints_;
  vector<vector<cv::KeyPoint>> detected_cur_keypoints_;
  
  float raw_to_float_depth_;
};

}
