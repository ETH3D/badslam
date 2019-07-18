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

#ifdef HAVE_STRUCTURE
#include <ST/CaptureSession.h>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>

#include "badslam/bad_slam_config.h"

namespace vis {

#ifdef HAVE_STRUCTURE

struct SessionDelegate;

// Manages a thread which retrieves input RGB-D frames from a Structure Core
// depth camera and stores them in an RGBDVideo.
class StructureInputThread {
 friend struct SessionDelegate;
 public:
  ~StructureInputThread();
  
  // Initializes the input streams, waits for a short while to let auto-exposure
  // and auto white balance adapt, then starts the input thread.
  // @param rgbd_video Pointer to the RGBDVideo where the recorded frames will
  //                   be stored.
  // @param depth_scaling Output parameter in which the depth scaling will be returned:
  //                      recorded_depth = depth_scaling * depth_in_meters
  void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling, const BadSlamConfig& config);
  
  // Retrieves the next input frame and stores it in the RGBDVideo given to the
  // constructor. Blocks while no new input frame is available.
  void GetNextFrame();
  
 private:
  void ThreadMain();
  
  std::mutex queue_mutex;
  std::condition_variable new_frame_condition_;
  vector<shared_ptr<Image<u16>>> depth_image_queue;
  vector<shared_ptr<Image<Vec3u8>>> color_image_queue;
  
  atomic<bool> have_intrinsics_;
  std::mutex have_intrinsics_mutex_;
  std::condition_variable have_intrinsics_condition_;
  ST::Intrinsics color_intrinsics_;
  ST::Intrinsics depth_intrinsics_;
  PinholeCamera4f target_camera;
  Image<Vec2f> color_undistortion_map_;
  Image<Vec3f> depth_unprojection_map_;
  atomic<bool> have_target_intrinsics_;
  
  bool one_shot_calibration_;
  bool stream_depth_only_;
  float depth_difference_threshold_;
  
  RGBDVideo<Vec3u8, u16>* rgbd_video_;
  
  shared_ptr<SessionDelegate> delegate_;
  shared_ptr<ST::CaptureSession> session_;
};

#else

// Dummy version of StructureInputThread which replaces the actual version in
// case the program is compiled without librealsense2. Asserts if any of its
// functions are called.
class StructureInputThread {
 public:
  inline void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling, const BadSlamConfig& config) {
    (void) rgbd_video;
    (void) depth_scaling;
    (void) config;
    LOG(FATAL) << "Structure input requested, but the program was compiled without Structure support.";
  }
  
  inline void GetNextFrame() {
    LOG(FATAL) << "Structure input requested, but the program was compiled without Structure support.";
  }
};

#endif
}
