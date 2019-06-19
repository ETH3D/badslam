// Copyright 2019 ETH Zürich, Silvano Galliani, Thomas Schöps
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

#ifdef HAVE_K4A
#include <k4a/k4a.h>
#include <k4arecord/playback.h>

// call k4a_device_close on every failed CHECK
#define K4A_CHECK(x)                                                                                               \
    {                                                                                                                  \
        auto retval = (x);                                                                                             \
        if (retval)                                                                                                    \
        {                                                                                                              \
            k4a_device_close(device);                                                                                  \
            LOG(ERROR) << "Runtime error: " << #x << " returned " << retval ;                                          \
        }                                                                                                              \
    }
#endif

#ifdef LIBVIS_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#include <atomic>
#include <mutex>
#include <thread>

#include <libvis/eigen.h>
#include <libvis/image.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>


namespace vis {

#ifdef HAVE_K4A

// Manages a thread which retrieves input RGB-D frames from a Microsoft Azure Kinect
// depth camera and stores them in an RGBDVideo.
class K4AInputThread {
 public:
  ~K4AInputThread();
  
  // Initializes the input streams, waits for a short while to let auto-exposure
  // and auto white balance adapt, then starts the input thread.
  // @param rgbd_video Pointer to the RGBDVideo where the recorded frames will
  //                   be stored.
  // @param depth_scaling Output parameter in which the depth scaling will be returned:
  //                      recorded_depth = depth_scaling * depth_in_meters
  void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling, int fps, int resolution, int _factor, int use_depth, string mode, int exposure);
  
  // Retrieves the next input frame and stores it in the RGBDVideo given to the
  // constructor. Blocks while no new input frame is available.
  void GetNextFrame();
  
 private:
  void ThreadMain();
  void init_memory();
  void init_undistortion_map();
  bool decode_image_opencv(
      const k4a_image_t & color_image,
      k4a_image_t * uncompressed_color_image, 
      cv::Mat & decodedImage);

  bool transform_depth_to_color(const k4a_transformation_t & transformation_handle, const k4a_image_t & depth_image, const k4a_image_t & color_image, k4a_image_t * transformed_image);

  bool undistort_depth_and_rgb(k4a_calibration_intrinsic_parameters_t & intrinsics, const cv::Mat & cv_color, const cv::Mat & cv_depth, cv::Mat & undistorted_color, cv::Mat & undistorted_depth, const float factor);
  uint32_t k4a_convert_fps_to_uint(k4a_fps_t fps);

  k4a_fps_t k4a_convert_uint_to_fps(int fps);

  k4a_depth_mode_t k4a_convert_string_to_mode(std::string strmode);

  k4a_color_resolution_t k4a_convert_uint_to_resolution(int fps);
  
  std::mutex queue_mutex;
  std::condition_variable new_frame_condition_;
  vector<shared_ptr<Image<u16>>> depth_image_queue;
  vector<shared_ptr<Image<Vec3u8>>> color_image_queue;
  k4a_device_t device{ NULL };
  k4a_capture_t capture;
  k4a_calibration_t calibration;
  k4a_transformation_t transformation;
  k4a_image_t transformed_depth_image;
  k4a_image_t uncompressed_color_image;
  k4a_device_configuration_t config{ K4A_DEVICE_CONFIG_INIT_DISABLE_ALL };
  cv::Mat camera_matrix;
  cv::Mat new_camera_matrix;
  int width;
  int height;

  float factor{ 1.0 }; // scaling factor
  bool use_depth{ false };
  string mode;
  cv::Mat cv_undistorted_color;
  cv::Mat cv_undistorted_depth;
  cv::Mat cv_undistorted_color_noalpha;
  cv::Mat cv_depth_downscaled;
  cv::Mat cv_color_downscaled;
  cv::Mat map1;
  cv::Mat map2;
  
  atomic<bool> exit_;
  
  // The pipeline should not be allocated unless it is actually used. When that
  // object is allocated, it seems to create several threads already and also
  // continuously creates new short-lived threads.
  //unique_ptr<rs2::pipeline> pipe;
  //shared_ptr<rs2::align> align_;
  
  RGBDVideo<Vec3u8, u16>* rgbd_video_;
  
  unique_ptr<thread> thread_;
};

#else

// Dummy version of K4AInputThread which replaces the actual version in
// case the program is compiled without librealsense2. Asserts if any of its
// functions are called.
class K4AInputThread {
 public:
  void Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling, int fps, int resolution, int _factor, int use_depth, string mode, int exposure) {
    (void) rgbd_video;
    (void) depth_scaling;
    (void) fps;
    (void) resolution;
    (void) _factor;
    (void) use_depth;
    (void) mode;
    (void) exposure;
    LOG(FATAL) << "Azure Kinect input requested, but the program was compiled without Azure Kinect support.";
  }
  
  inline void GetNextFrame() {
    LOG(FATAL) << "Azure Kinect input requested, but the program was compiled without Azure Kinect support.";
  }
};

#endif

}
