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


#include "badslam/input_realsense.h"

#ifdef BAD_SLAM_HAVE_REALSENSE

namespace vis {

RealSenseInputThread::~RealSenseInputThread() {
  exit_ = true;
  if (thread_) {
    thread_->join();
  }
}

void RealSenseInputThread::Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling) {
  rgbd_video_ = rgbd_video;
  
  pipe.reset(new rs2::pipeline());
  
  // // Start streaming with default recommended configuration
  // rs2::pipeline_profile profile = pipe.start();
  
  // Create a configuration for configuring the pipeline with a non default profile
  rs2::config cfg;
  
  // Add desired streams to configuration
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
  
  // Instruct pipeline to start streaming with the requested configuration
  rs2::pipeline_profile profile = pipe->start(cfg);
  
  // Set the white balance and exposure to auto to get good values
  for (rs2::sensor& sensor : profile.get_device().query_sensors()) {
    if (sensor.get_stream_profiles()[0].stream_type() == RS2_STREAM_COLOR) {
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, true);
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, true);
    }
  }
  
  // Determine the depth scale
  float depth_scale = numeric_limits<float>::quiet_NaN();
  for (rs2::sensor& sensor : profile.get_device().query_sensors()) {
    // Check if the sensor is a depth sensor
    if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>()) {
      depth_scale = dpt.get_depth_scale();
    }
  }
  if (std::isnan(depth_scale)) {
    LOG(FATAL) << "Cannot determine the depth scale";
  }
  *depth_scaling = 1.0 / depth_scale;
  
  // Find streams
  rs2_stream color_stream = RS2_STREAM_ANY;
  rs2_intrinsics color_intrinsics;
  
  rs2_stream depth_stream = RS2_STREAM_ANY;
  rs2_intrinsics depth_intrinsics;
  
  for (rs2::stream_profile sp : profile.get_streams()) {
    rs2_stream profile_stream = sp.stream_type();
    if (color_stream == RS2_STREAM_ANY && profile_stream == RS2_STREAM_COLOR) {
      color_stream = profile_stream;
      color_intrinsics = sp.as<rs2::video_stream_profile>().get_intrinsics();
    }
    if (depth_stream == RS2_STREAM_ANY && profile_stream == RS2_STREAM_DEPTH) {
      depth_stream = profile_stream;
      depth_intrinsics = sp.as<rs2::video_stream_profile>().get_intrinsics();
    }
  }
  if (color_stream == RS2_STREAM_ANY) {
    LOG(FATAL) << "Cannot find a color stream";
  }
  if (depth_stream == RS2_STREAM_ANY) {
    LOG(FATAL) << "Cannot find a depth stream";
  }
  
  // Prepare depth reprojection to the color stream
  align_.reset(new rs2::align(color_stream));
  
  // Set color camera
  // TODO: Ignoring the color stream's distortion. In principle, we should undistort the
  //       frames ourselves (and possibly create undistorted rs2::video_frames such
  //       that we can still use rs2::align to reproject the depth images to the
  //       color frame?). In practice, the distortion coefficients for my camera were all
  //       zero, so we can simply ignore them in that case.
  float color_parameters[4];
  color_parameters[0] = color_intrinsics.fx;
  color_parameters[1] = color_intrinsics.fy;
  color_parameters[2] = color_intrinsics.ppx + 0.5f;
  color_parameters[3] = color_intrinsics.ppy + 0.5f;
  if (color_intrinsics.coeffs[0] != 0 ||
      color_intrinsics.coeffs[1] != 0 ||
      color_intrinsics.coeffs[2] != 0 ||
      color_intrinsics.coeffs[3] != 0 ||
      color_intrinsics.coeffs[4] != 0) {
    LOG(ERROR) << "Ignoring the color stream's distortion, but at least one of the distortion coefficients is non-zero!";
    LOG(ERROR) << "Model: " << color_intrinsics.model << ". Coefficients: "
               << color_intrinsics.coeffs[0] << ", "
               << color_intrinsics.coeffs[1] << ", "
               << color_intrinsics.coeffs[2] << ", "
               << color_intrinsics.coeffs[3] << ", "
               << color_intrinsics.coeffs[4] << "";
  }
  rgbd_video->color_camera_mutable()->reset(new PinholeCamera4f(
      color_intrinsics.width, color_intrinsics.height, color_parameters));
  
  // Set depth camera to be the same as the color camera
  rgbd_video->depth_camera_mutable()->reset(new PinholeCamera4f(
      color_intrinsics.width, color_intrinsics.height, color_parameters));
  
  // Wait a short time to let the auto-exposure and white balance find a good initial setting
  for (int i = 0; i < 30; ++ i) {
    pipe->wait_for_frames();
  }
  
  // Set the white balance and exposure to fixed for more consistent coloring of the reconstruction
  for (rs2::sensor& sensor : profile.get_device().query_sensors()) {
    if (sensor.get_stream_profiles()[0].stream_type() == RS2_STREAM_COLOR) {
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, false);
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, false);
    }
  }
  
  // Wait some more to let the new settings take effect
  for (int i = 0; i < 15; ++ i) {
    pipe->wait_for_frames();
  }
  
  // Start thread
  exit_ = false;
  thread_.reset(new thread(std::bind(&RealSenseInputThread::ThreadMain, this)));
}

void RealSenseInputThread::GetNextFrame() {
  // Wait for the next frame
  unique_lock<mutex> lock(queue_mutex);
  while (depth_image_queue.empty()) {
    new_frame_condition_.wait(lock);
  }
  
  shared_ptr<Image<u16>> depth_image = depth_image_queue.front();
  depth_image_queue.erase(depth_image_queue.begin());
  shared_ptr<Image<Vec3u8>> color_image = color_image_queue.front();
  color_image_queue.erase(color_image_queue.begin());
  
  lock.unlock();
  
  // Add the frame to the RGBDVideo object
  rgbd_video_->depth_frames_mutable()->push_back(
      ImageFramePtr<u16, SE3f>(new ImageFrame<u16, SE3f>(depth_image)));
  rgbd_video_->color_frames_mutable()->push_back(
      ImageFramePtr<Vec3u8, SE3f>(new ImageFrame<Vec3u8, SE3f>(color_image)));
}

void RealSenseInputThread::ThreadMain() {
  while (true) {
    if (exit_) {
      break;
    }
    
    // Wait for a new frame from the camera
    rs2::frameset frameset = pipe->wait_for_frames();
    rs2::depth_frame depth = frameset.get_depth_frame();
    rs2::video_frame color = frameset.get_color_frame();
    
    // Reproject the depth image into the color frame
    auto processed = align_->process(frameset);
    // rs2::video_frame other_frame = processed.first(align_to);
    rs2::depth_frame aligned_depth = processed.get_depth_frame();
    
    // Add the frame to the queue
    unique_lock<mutex> lock(queue_mutex);
    
    shared_ptr<Image<u16>> depth_image(new Image<u16>(aligned_depth.get_width(), aligned_depth.get_height()));
    depth_image->SetTo(reinterpret_cast<const u16*>(aligned_depth.get_data()), aligned_depth.get_stride_in_bytes());
    depth_image_queue.push_back(depth_image);
    
    shared_ptr<Image<Vec3u8>> color_image(new Image<Vec3u8>(color.get_width(), color.get_height()));
    color_image->SetTo(reinterpret_cast<const Vec3u8*>(color.get_data()), color.get_stride_in_bytes());
    color_image_queue.push_back(color_image);
    
    lock.unlock();
    new_frame_condition_.notify_all();
  }
}

}

#endif  // BAD_SLAM_HAVE_REALSENSE
