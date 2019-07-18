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


#include "badslam/input_structure.h"

#include <chrono>

#include <libvis/mesh.h>
#include <libvis/mesh_opengl.h>
#include <libvis/opengl_context.h>
#include <libvis/renderer.h>

#include "badslam/undistortion.h"

#ifdef HAVE_STRUCTURE

namespace vis {

constexpr int kDepthScaling = 2500;

bool isMono(const ST::ColorFrame &visFrame) {
  return visFrame.width() * visFrame.height() == visFrame.rgbSize();
}

struct SessionDelegate : ST::CaptureSessionDelegate {
  SessionDelegate(StructureInputThread* thread)
      : thread_(thread) {
    start_time_ = chrono::steady_clock::now();
  }
  
  ~SessionDelegate() {
    if (context) {
      // Temporarily make the OpenGL context current in this thread.
      OpenGLContext old_context;
      SwitchOpenGLContext(*context, &old_context);
      
      renderer.reset();
      gl_mesh.reset();
      
      SwitchOpenGLContext(old_context);
      
      // TODO: We leak the context here. This is because at the point this destructor
      //       is called, the thread which ran the Qt event loop is done with that
      //       and waits for the input thread to exit. So, if we use the Qt implementation
      //       for the OpenGL contexts, it would fail attempting to RunInQtThreadBlocking()
      //       during its destruction. This leak is not a real issue since the program
      //       is going to exit anyway, but it would be nicer to clean up properly.
      context.release();
    }
  }
  
  void captureSessionEventDidOccur(ST::CaptureSession* session, ST::CaptureSessionEventId event) override {
    LOG(INFO) << "Received capture session event " << (int)event << " (" << ST::CaptureSessionSample::toString(event) << ")";
    switch (event) {
      case ST::CaptureSessionEventId::Booting: break;
      case ST::CaptureSessionEventId::Ready:
        LOG(INFO) << "Starting streams...";
        LOG(INFO) << "Sensor Serial Number is " << session->sensorSerialNumber();
        session->startStreaming();
        break;
      case ST::CaptureSessionEventId::Disconnected:
      case ST::CaptureSessionEventId::Error:
        LOG(ERROR) << "Capture session error";
        break;
      default:
        LOG(INFO) << "  Event unhandled";
    }
  }
  
  void captureSessionDidOutputSample(ST::CaptureSession* /*session*/, const ST::CaptureSessionSample& sample) override {
    // Discard frames during the first 10 seconds if doing one-shot dynamic calibration.
    // TODO: Is there any way to figure out when calibration completed? Could it take longer than 10 seconds?
    if (thread_->one_shot_calibration_ &&
        (chrono::duration<double>(chrono::steady_clock::now() - start_time_).count() < 10)) {
      LOG(INFO) << "Discarding frame while waiting for calibration to complete ...";
      return;
    }
    
    if ((thread_->stream_depth_only_ && sample.type == ST::CaptureSessionSample::Type::DepthFrame) ||
        (!thread_->stream_depth_only_ && sample.type == ST::CaptureSessionSample::Type::SynchronizedFrames)) {
      if (!thread_->stream_depth_only_ && !sample.visibleFrame.isValid()) {
        LOG(ERROR) << "Visible frame is not valid";
        return;
      }
      if (!sample.depthFrame.isValid()) {
        LOG(ERROR) << "Depth frame is not valid";
        return;
      }
      
      // If this is the first sample, notify the main class about the intrinsics
      if (!thread_->have_intrinsics_) {
        if (!thread_->stream_depth_only_) {
          thread_->color_intrinsics_ = sample.visibleFrame.intrinsics();
        }
        thread_->depth_intrinsics_ = sample.depthFrame.intrinsics();
        
        thread_->have_intrinsics_ = true;
        thread_->have_intrinsics_condition_.notify_all();
      } else {
        if (!thread_->stream_depth_only_) {
          if (!(sample.visibleFrame.intrinsics() == thread_->color_intrinsics_)) {
            LOG(WARNING) << "Structure stream: Intrinsics of the visible frame changed while streaming!";
          }
        }
        if (!(sample.depthFrame.intrinsics() == thread_->depth_intrinsics_)) {
          LOG(WARNING) << "Structure stream: Intrinsics of the depth frame changed while streaming!";
        }
      }
      
      // If we haven't decided on the target intrinsics yet, drop the frame
      if (!thread_->have_target_intrinsics_) {
        return;
      }
      
      shared_ptr<Image<Vec3u8>> color_image(new Image<Vec3u8>());
      shared_ptr<Image<u16>> depth_image(new Image<u16>());
      
      if (thread_->stream_depth_only_) {
        // Create a dummy color image (TODO: This should ideally not be necessary)
        color_image->SetSize(thread_->target_camera.width(), thread_->target_camera.height());
        color_image->SetTo(Vec3u8(80, 80, 80));
        
        // Undistort the depth image if necessary
        if (thread_->depth_intrinsics_.k1k2k3p1p2AreZero()) {
          depth_image->SetSize(thread_->target_camera.width(), thread_->target_camera.height());
          const float* in_ptr = sample.depthFrame.depthInMillimeters();
          for (int i = 0; i < depth_image->pixel_count(); ++ i) {
            depth_image->data()[i] = isnan(in_ptr[i]) ? 0 : static_cast<u16>(in_ptr[i] + 0.5f);
          }
        } else {
          ReprojectDepthImage(sample.depthFrame, depth_image.get(), /*transform_to_color_frame*/ false);
        }
      } else {
        // Undistort the color image
        // TODO: Avoid the copy here, only wrap the data instead
        Image<Vec3u8> original_color(
            sample.visibleFrame.width(),
            sample.visibleFrame.height());
        if (!isMono(sample.visibleFrame)) {
          original_color.SetTo(reinterpret_cast<const Vec3u8*>(sample.visibleFrame.rgbData()));
        } else {
          for (int i = 0; i < original_color.pixel_count(); ++ i) {
            original_color.data()[i] = Vec3u8::Constant(sample.visibleFrame.yData()[i]);
          }
        }
        UndistortImage(original_color, color_image.get(), thread_->color_undistortion_map_);
        
        // Reproject the depth image into the color frame
        ReprojectDepthImage(sample.depthFrame, depth_image.get(), /*transform_to_color_frame*/ true);
      }
      
      // Add the frame to the queue
      unique_lock<mutex> lock(thread_->queue_mutex);
      // If too many frames are queued, drop the frame
      if (thread_->depth_image_queue.size() < 10) {
        thread_->depth_image_queue.push_back(depth_image);
        thread_->color_image_queue.push_back(color_image);
      }
      lock.unlock();
      thread_->new_frame_condition_.notify_all();
    }
  }
  
  void ReprojectDepthImage(
      const ST::DepthFrame& original_depth,
      Image<u16>* depth_image,
      bool transform_to_color_frame) {
    const auto& depth_intrinsics = thread_->depth_intrinsics_;
    const auto& unprojection_map = thread_->depth_unprojection_map_;
    
    if (!mesh.vertices()) {
      mesh.vertices_mutable()->reset(new PointCloud<Point3f>());
    }
    
    // Create vertices
    shared_ptr<PointCloud<Point3f>>& vertices = *mesh.vertices_mutable();
    vertices->Resize(depth_intrinsics.height * depth_intrinsics.width);
    for (u32 y = 0; y < depth_intrinsics.height; ++ y) {
      for (u32 x = 0; x < depth_intrinsics.width; ++ x) {
        float depth = (1.f / 1000.f) * original_depth.depthInMillimeters()[x + depth_intrinsics.width * y];  // is NaN if no depth value at that pixel
        vertices->at(x + depth_intrinsics.width * y).position() = depth * unprojection_map(x, y);
      }
    }
    
    // Create triangles
    vector<Triangle<u32>>* triangles = mesh.triangles_mutable();
    triangles->clear();
    for (u32 y = 0; y < depth_intrinsics.height - 1; ++ y) {
      for (u32 x = 0; x < depth_intrinsics.width - 1; ++ x) {
        float depth_00 = (1.f / 1000.f) * original_depth.depthInMillimeters()[x + depth_intrinsics.width * y];
        float depth_01 = (1.f / 1000.f) * original_depth.depthInMillimeters()[(x + 1) + depth_intrinsics.width * y];
        float depth_10 = (1.f / 1000.f) * original_depth.depthInMillimeters()[x + depth_intrinsics.width * (y + 1)];
        float depth_11 = (1.f / 1000.f) * original_depth.depthInMillimeters()[(x + 1) + depth_intrinsics.width * (y + 1)];
        
        float max_diff =
            std::max(fabs(depth_00 - depth_10),
                    std::max(fabs(depth_00 - depth_01),
                              std::max(fabs(depth_00 - depth_11),
                                      std::max(fabs(depth_01 - depth_10),
                                                std::max(fabs(depth_01 - depth_11),
                                                         fabs(depth_10 - depth_11))))));
        if (max_diff < thread_->depth_difference_threshold_) {
          triangles->emplace_back();
          Triangle<u32>* new_triangle = &triangles->back();
          new_triangle->index(0) = (x + 0) + depth_intrinsics.width * (y + 0);
          new_triangle->index(1) = (x + 1) + depth_intrinsics.width * (y + 0);
          new_triangle->index(2) = (x + 0) + depth_intrinsics.width * (y + 1);
          
          triangles->emplace_back();
          new_triangle = &triangles->back();
          new_triangle->index(0) = (x + 1) + depth_intrinsics.width * (y + 0);
          new_triangle->index(1) = (x + 1) + depth_intrinsics.width * (y + 1);
          new_triangle->index(2) = (x + 0) + depth_intrinsics.width * (y + 1);
        }
      }
    }
    
    // Render and transfer the result to the CPU.
    if (!gl_mesh) {
      context.reset(new OpenGLContext());
      if (!context->InitializeWindowless()) {
        LOG(FATAL) << "Cannot initialize windowless OpenGL context.";
      }
      SwitchOpenGLContext(*context);
      
      gl_mesh.reset(new Mesh3fOpenGL());
      program_storage.reset(new RendererProgramStorage());
      renderer.reset(new Renderer(
          /*render_color*/ false,
          /*render_depth*/ true,
          thread_->target_camera.width(), thread_->target_camera.height(),
          program_storage));
    }
    gl_mesh->TransferToGPU(mesh, GL_DYNAMIC_DRAW);
    
    SE3f transform;
    if (transform_to_color_frame) {
      const auto& P = original_depth.visibleCameraPoseInDepthCoordinateFrame();
      Matrix<float, 3, 4> M;
      // NOTE: Naming of the attributes in ST::Matrix4 is confusing, entries are named: mColumnRow.
      M << P.m00, P.m10, P.m20, P.m30,
           P.m01, P.m11, P.m21, P.m31,
           P.m02, P.m12, P.m22, P.m32;
      if (P.m03 != 0 || P.m13 != 0 || P.m23 != 0 || P.m33 != 1) {
        LOG(WARNING) << "Last row in transformation matrix is not (0, 0, 0, 1), this is not handled!";
        LOG(WARNING) << "Row values: " << P.m03 << " " << P.m13 << " " << P.m23 << " " << P.m33;
      }
      transform = SE3f(M.leftCols<3>(), M.rightCols<1>());
    }
    renderer->BeginRendering(transform, thread_->target_camera, /*min_depth*/ 0.05f, /*max_depth*/ 50.f);
    gl_mesh->Render(&renderer->shader_program());
    renderer->EndRendering();
    
    Image<float> depth_image_float(thread_->target_camera.width(), thread_->target_camera.height());
    // TODO: It would be much more efficient to keep the depth image on the GPU,
    //       rather than downloading it here and uploading it again later (once it has been passed to BAD SLAM).
    renderer->DownloadDepthResult(thread_->target_camera.width(), thread_->target_camera.height(), depth_image_float.data());
    
    // Convert depth to u16
    depth_image->SetSize(depth_image_float.size());
    for (u32 y = 0; y < depth_image_float.height(); ++ y) {
      for (u32 x = 0; x < depth_image_float.width(); ++ x) {
        (*depth_image)(x, y) = (depth_image_float(x, y) == 0) ? 0 : (kDepthScaling * depth_image_float(x, y) + 0.5f);
      }
    }
  }
  
  Mesh3f mesh;
  unique_ptr<OpenGLContext> context;
  unique_ptr<Mesh3fOpenGL> gl_mesh;
  unique_ptr<Renderer> renderer;
  RendererProgramStoragePtr program_storage;
  
  chrono::steady_clock::time_point start_time_;
  StructureInputThread* thread_;
};


StructureInputThread::~StructureInputThread() {
  if (session_) {
    session_->stopStreaming();
    session_.reset();  // make sure that the session is deleted before the delegate
  }
}

void StructureInputThread::Start(RGBDVideo<Vec3u8, u16>* rgbd_video, float* depth_scaling, const BadSlamConfig& config) {
  rgbd_video_ = rgbd_video;
  
  one_shot_calibration_ = config.structure_one_shot_dynamic_calibration;
  stream_depth_only_ = config.structure_depth_only;
  depth_difference_threshold_ = config.structure_depth_diff_threshold;
  
  ST::CaptureSessionSettings settings;
  settings.source = ST::CaptureSessionSourceId::StructureCore;
  settings.frameSyncEnabled = true;
  settings.applyExpensiveCorrection = config.structure_expensive_correction;
  settings.structureCore.depthEnabled = true;
  settings.structureCore.visibleEnabled = !stream_depth_only_;
  settings.structureCore.infraredEnabled = false;
  settings.structureCore.infraredAutoExposureEnabled = config.structure_infrared_auto_exposure;
  settings.structureCore.accelerometerEnabled = false;
  settings.structureCore.gyroscopeEnabled = false;
  
  if (config.structure_one_shot_dynamic_calibration) {
    settings.structureCore.dynamicCalibrationMode = ST::StructureCoreDynamicCalibrationMode::OneShotPersistent;
  } else {
    settings.structureCore.dynamicCalibrationMode = ST::StructureCoreDynamicCalibrationMode::Off;
  }
  
  if (config.structure_depth_resolution == string("320x240")) {
    settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::_320x240;
  } else if (config.structure_depth_resolution == string("640x480")) {
    settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::_640x480;
  } else if (config.structure_depth_resolution == string("1280x960")) {
    settings.structureCore.depthResolution = ST::StructureCoreDepthResolution::_1280x960;
  } else {
    LOG(FATAL) << "Unknown value for config.structure_depth_resolution: " << config.structure_depth_resolution;
  }
  
  if (config.structure_depth_range == string("VeryShort")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::VeryShort;
  } else if (config.structure_depth_range == string("Short")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Short;
  } else if (config.structure_depth_range == string("Medium")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Medium;
  } else if (config.structure_depth_range == string("Long")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Long;
  } else if (config.structure_depth_range == string("VeryLong")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::VeryLong;
  } else if (config.structure_depth_range == string("Hybrid")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Hybrid;
  } else if (config.structure_depth_range == string("BodyScanning")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::BodyScanning;
  } else if (config.structure_depth_range == string("Default")) {
    settings.structureCore.depthRangeMode = ST::StructureCoreDepthRangeMode::Default;
  } else {
    LOG(FATAL) << "Unknown value for config.structure_depth_range: " << config.structure_depth_resolution;
  }
  
  have_intrinsics_ = false;
  have_target_intrinsics_ = false;
  
  delegate_.reset(new SessionDelegate(this));
  session_.reset(new ST::CaptureSession());
  session_->setDelegate(delegate_.get());
  if (!session_->startMonitoring(settings)) {
    LOG(FATAL) << "Failed to initialize capture session";
  }
  
  // Get the intrinsics from the first streamed frame
  unique_lock<mutex> lock(have_intrinsics_mutex_);
  while (!have_intrinsics_) {
    have_intrinsics_condition_.wait(lock);
  }
  
  // Decide for undistorted intrinsics and store those in rgbd_video.
  // TODO: Assuming that the ST::Intrinsics are given in "pixel center" convention, is that correct?
  vector<double> color_parameters = {
      color_intrinsics_.k1, color_intrinsics_.k2, color_intrinsics_.k3,
      color_intrinsics_.p1, color_intrinsics_.p2,
      color_intrinsics_.fx, color_intrinsics_.fy,
      color_intrinsics_.cx, color_intrinsics_.cy};
  RadtanCamera9d color_camera(color_intrinsics_.width, color_intrinsics_.height, color_parameters.data());
  
  vector<double> depth_parameters = {
      depth_intrinsics_.k1, depth_intrinsics_.k2, depth_intrinsics_.k3,
      depth_intrinsics_.p1, depth_intrinsics_.p2,
      depth_intrinsics_.fx, depth_intrinsics_.fy,
      depth_intrinsics_.cx, depth_intrinsics_.cy};
  RadtanCamera9d depth_camera(depth_intrinsics_.width, depth_intrinsics_.height, depth_parameters.data());
  
  if (stream_depth_only_) {
    // If the distortion is zero, use the images directly.
    if (depth_intrinsics_.k1k2k3p1p2AreZero()) {
      vector<float> pinhole_parameters = {
          depth_intrinsics_.fx, depth_intrinsics_.fy,
          depth_intrinsics_.cx, depth_intrinsics_.cy};
      target_camera = PinholeCamera4f(depth_camera.width(), depth_camera.height(), pinhole_parameters.data());
      *depth_scaling = 1000;
    } else {
      DecideForUndistortedCamera(&depth_camera, &target_camera, /*avoid_invalid_pixels*/ false);
      *depth_scaling = kDepthScaling;
    }
  } else {
    DecideForUndistortedCamera(&color_camera, &target_camera, /*avoid_invalid_pixels*/ true);
    *depth_scaling = kDepthScaling;
  }
  rgbd_video->color_camera_mutable()->reset(target_camera.Scaled(1.0));  // small HACK: using Scaled(1.0) to duplicate camera
  rgbd_video->depth_camera_mutable()->reset(target_camera.Scaled(1.0));  // small HACK: using Scaled(1.0) to duplicate camera
  
  CreateUndistortionMap(
      &color_camera,
      target_camera,
      &color_undistortion_map_);
  
  depth_unprojection_map_.SetSize(depth_camera.width(), depth_camera.height());
  for (int y = 0; y < depth_camera.height(); ++ y) {
    for (int x = 0; x < depth_camera.width(); ++ x) {
      depth_unprojection_map_(x, y) = depth_camera.UnprojectFromPixelCenterConv(Vec2f(x, y)).cast<float>();
    }
  }
  
  have_target_intrinsics_ = true;
}

void StructureInputThread::GetNextFrame() {
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

}

#endif  // HAVE_STRUCTURE
