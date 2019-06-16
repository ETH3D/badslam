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

#ifdef BAD_SLAM_HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>

namespace vis {

#ifdef BAD_SLAM_HAVE_REALSENSE

// Manages a thread which retrieves input RGB-D frames from an Intel RealSense
// depth camera and stores them in an RGBDVideo.
class RealSenseInputThread {
 public:
  ~RealSenseInputThread();
  
  // Initializes the input streams, waits for a short while to let auto-exposure
  // and auto white balance adapt, then starts the input thread.
  // @param rgbd_video Pointer to the RGBDVideo where the recorded frames will
  //                   be stored.
  // @param depth_scaling Output parameter in which the depth scaling will be returned:
  //                      recorded_depth = depth_scaling * depth_in_meters
  void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling);
  
  // Retrieves the next input frame and stores it in the RGBDVideo given to the
  // constructor. Blocks while no new input frame is available.
  void GetNextFrame();
  
 private:
  void ThreadMain();
  
  std::mutex queue_mutex;
  std::condition_variable new_frame_condition_;
  vector<shared_ptr<Image<u16>>> depth_image_queue;
  vector<shared_ptr<Image<Vec3u8>>> color_image_queue;
  
  atomic<bool> exit_;
  
  // The pipeline should not be allocated unless it is actually used. When that
  // object is allocated, it seems to create several threads already and also
  // continuously creates new short-lived threads.
  unique_ptr<rs2::pipeline> pipe;
  shared_ptr<rs2::align> align_;
  
  RGBDVideo<Vec3u8, u16>* rgbd_video_;
  
  unique_ptr<thread> thread_;
};

#else

// Dummy version of RealSenseInputThread which replaces the actual version in
// case the program is compiled without librealsense2. Asserts if any of its
// functions are called.
class RealSenseInputThread {
 public:
  inline void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling) {
    (void) rgbd_video;
    (void) depth_scaling;
    LOG(FATAL) << "RealSense input requested, but the program was compiled without RealSense support.";
  }
  
  inline void GetNextFrame() {
    LOG(FATAL) << "RealSense input requested, but the program was compiled without RealSense support.";
  }
};

#endif

}
