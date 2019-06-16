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


// Must be included before render_window.h to avoid errors
#include <QApplication>
#include <QClipboard>

#include "badslam/render_window.h"

#include <iomanip>

#include <GL/glew.h>
#ifndef WIN32
#include <GL/glx.h>
#endif
#include <libvis/image.h>
#include <libvis/cuda/cuda_util.h>
#include <libvis/render_window_qt_opengl.h>

#include "badslam/kernels.h"
#include "badslam/direct_ba.h"
#include "badslam/util.cuh"
#include "badslam/keyframe.h"

namespace vis {

struct TimerHelper : public QObject {
 Q_OBJECT
 public:
  TimerHelper(BadSlamRenderWindow* render_window)
      : render_window(render_window) {
    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(RenderFrame()));
  }
 
 public slots:
  void RenderFrame() {
    render_window->RenderFrame();
  }
  
 public:
  BadSlamRenderWindow* render_window;
  QTimer* timer;
};

BadSlamRenderWindow::BadSlamRenderWindow(
    float splat_half_extent_in_pixels,
    bool embedded_in_gui)
    : have_visualization_cloud_(false),
      embedded_in_gui_(embedded_in_gui),
      dense_ba_(nullptr) {
  width_ = 0;
  height_ = 0;
  
  last_drag_x_ = 0;
  last_drag_y_ = 0;
  
  background_color_ = Vec4f(1, 1, 1, 0);
  
  dragging_ = false;
  pressed_mouse_buttons_ = 0;
  m_pressed_ = false;
  
  // Set default view parameters.
  min_depth_ = 0.01f;
  max_depth_ = 50.0f;
  
  camera_free_orbit_theta_ = 0.5;
  camera_free_orbit_phi_ = -1.57;
  camera_free_orbit_radius_ = 6;
  camera_free_orbit_offset_ = Vec3f(0, 0, 0);
  
  use_camera_matrix_ = false;
  
  splat_half_extent_in_pixels_ = splat_half_extent_in_pixels;
  
  render_current_frame_frustum_ = true;
  render_estimated_trajectory_ = true;
  render_ground_truth_trajectory_ = true;
  render_keyframes_ = true;
  render_surfels_ = true;
  
  // up_direction_rotation_ = Mat3f::Identity();
  up_direction_rotation_ = AngleAxisf(M_PI, Vec3f(0, 0, 1)) * AngleAxisf(-M_PI / 2, Vec3f(1, 0, 0));
  
  pose_correction_matrix_ = Mat4f::Identity();
  
  visualization_cloud_size_ = numeric_limits<usize>::max();
  new_visualization_cloud_size_ = numeric_limits<usize>::max();
  
  init_max_point_count_ = 0;
  
  current_frame_pose_set_ = false;
  current_frame_cloud_set_ = false;
  
  camera_update_ = false;
  
  surfel_vertices_ = nullptr;
  
  tool_ = Tool::kSelectKeyframe;
}

BadSlamRenderWindow::~BadSlamRenderWindow() {}

void BadSlamRenderWindow::SetDirectBA(DirectBA* dense_ba) {
  dense_ba_ = dense_ba;
}

void BadSlamRenderWindow::Initialize() {
  GLenum glew_init_result = glewInit();
  CHECK_EQ(static_cast<int>(glew_init_result), GLEW_OK);
  glGetError();  // Ignore GL_INVALID_ENUM​ error caused by glew
  CHECK_OPENGL_NO_ERROR();
  
  CreateSplatProgram();
  CHECK_OPENGL_NO_ERROR();
  
  CreateConstantColorProgram();
  CHECK_OPENGL_NO_ERROR();
  
  // TODO: It would probably be preferable to handle this in a sane way instead
  //       of simply creating a global VAO at the beginning and then forgetting
  //       about it.
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
}

void BadSlamRenderWindow::Deinitialize() {
  if (surfel_vertices_) {
    cudaGraphicsUnregisterResource(surfel_vertices_);
  }
}

void BadSlamRenderWindow::Resize(int width, int height) {
  width_ = width;
  height_ = height;
}

void BadSlamRenderWindow::Render() {
  CHECK_OPENGL_NO_ERROR();
  
  unique_lock<mutex> lock(render_mutex_);
  
  if (use_follow_camera_) {
    // Update the camera based on global_T_current_frame_.
    // camera_matrix_ is used as camera_T_world_.
    SE3f follow_camera_tr_trajectory_camera;
    follow_camera_tr_trajectory_camera.translation() = Vec3f(0, 0.7 * 0.3, 0.7 * 1);
    follow_camera_tr_trajectory_camera.setRotationMatrix(AngleAxisf(12 * M_PI / 180.f, Vec3f(1, 0, 0)).matrix());
    
    SE3f current_frame_tr_world = SE3f(global_T_current_frame_).inverse();
    SE3f target_camera_tr_world = follow_camera_tr_trajectory_camera * current_frame_tr_world;
    
    chrono::steady_clock::time_point now = chrono::steady_clock::now();
    
    if (!follow_camera_initialized_) {
      timer_helper_.reset(new TimerHelper(this));
      timer_helper_->timer->start(33);
      
      // Set the follow-camera based on the current camera position.
      use_camera_matrix_ = true;
      camera_matrix_ = target_camera_tr_world.matrix();
      
      follow_camera_initialized_ = true;
    } else {
      // Update the follow-camera position.
      float factor_per_second = 0.15f;  // Factor applied on the difference between current and target pose per second
      float elapsed_seconds = 1e-9 * chrono::duration<double, nano>(now - last_render_time_).count();
      float factor = std::pow(factor_per_second, elapsed_seconds);
      
      const SE3f current_camera_tr_world = SE3f(camera_matrix_);
      camera_matrix_ = SE3f(current_camera_tr_world.unit_quaternion().slerp(1 - factor, target_camera_tr_world.unit_quaternion()),
                            factor * current_camera_tr_world.translation() + (1 - factor) * target_camera_tr_world.translation()).matrix();
    }
    
    last_render_time_ = now;
  } else if (use_rotating_camera_) {
    chrono::steady_clock::time_point now = chrono::steady_clock::now();
    
    if (!follow_camera_initialized_) {
      timer_helper_.reset(new TimerHelper(this));
      timer_helper_->timer->start(33);
      
      follow_camera_initialized_ = true;
    } else {
      // Update the follow-camera position.
      float radiant_per_second = -0.4f;  // Factor applied on the difference between current and target pose per second
      float elapsed_seconds = 1e-9 * chrono::duration<double, nano>(now - last_render_time_).count();
      float delta = radiant_per_second * elapsed_seconds;
      
      camera_free_orbit_phi_ += delta;
    }
    
    last_render_time_ = now;
  } else if (playback_frame_ >= 0) {
    // Determine camera pose from spline-based keyframe playback.
    usize first_keyframe_index = spline_frame_indices.size() - 1;
    for (usize i = 1; i < spline_frame_indices.size(); ++ i) {
      if (spline_frame_indices[i] >= playback_frame_) {
        first_keyframe_index = i - 1;
        break;
      }
    }
    usize prev_frame_index = spline_frame_indices[first_keyframe_index];
    usize next_frame_index = spline_frame_indices[first_keyframe_index + 1];
    float t = -1.f + first_keyframe_index + (playback_frame_ - prev_frame_index) * 1.0f / (next_frame_index - prev_frame_index);
    
    Vec3f camera_free_orbit_offset;
    camera_free_orbit_offset.x() = offset_x_spline->getPosition(t);
    camera_free_orbit_offset.y() = offset_y_spline->getPosition(t);
    camera_free_orbit_offset.z() = offset_z_spline->getPosition(t);
    float camera_free_orbit_radius = radius_spline->getPosition(t);
    float camera_free_orbit_theta = theta_spline->getPosition(t);
    float camera_free_orbit_phi = phi_spline->getPosition(t);
    
    SetViewParameters(camera_free_orbit_offset, camera_free_orbit_radius, camera_free_orbit_theta, camera_free_orbit_phi, max_depth_);
    
    if (render_playback_) {
      ostringstream screenshot_path;
      screenshot_path << "/media/thomas/Daten/RGBD-SLAM-Benchmark/oral_video/images/raw_frames/" << std::setw(5) << std::setfill('0') << playback_frame_ << ".png";
      screenshot_path_ = screenshot_path.str();
    }
    
    if (playback_frame_ >= spline_frame_indices[spline_frame_indices.size() - 2]) {
      // End playback.
      timer_helper_.reset();
      playback_frame_ = -1;
    } else {
      ++ playback_frame_;
      
      // Ensure that the next frame will get rendered
      timer_helper_.reset(new TimerHelper(this));
      timer_helper_->timer->start(33);
    }
  } else if (timer_helper_) {
    timer_helper_.reset();
    use_camera_matrix_ = false;
    follow_camera_initialized_ = false;
  }
  
  
  // ### Setup ###
  
  if (camera_update_) {
    camera_frustum_.Create(updated_camera_);
    camera_update_ = false;
  }
  
  // Setup the render_camera_.
  SetCamera();
  
  unique_lock<mutex> camera_mutex_lock(camera_mutex_);
  
  // Compute projection_matrix_ from the camera.
  ComputeProjectionMatrix();
  
  // Set the rendering bounds (viewport) according to the camera dimensions.
  SetupViewport();
  
  // Set the camera_T_world_ transformation according to an orbiting scheme.
  SetViewpoint();
  
  camera_mutex_lock.unlock();
  
  CHECK_OPENGL_NO_ERROR();
  
  // Compute the model-view-projection matrix.
  Mat4f model_matrix = Mat4f::Identity();
  Mat4f model_view_matrix = camera_T_world_.matrix() * model_matrix;
  model_view_projection_matrix_ = projection_matrix_ * model_view_matrix;
  
  // Set states for rendering.
  glClearColor(background_color_.x(), background_color_.y(), background_color_.z(), background_color_.w());  // background color
  // TODO: While it works on Ubuntu 14.04, both on Ubuntu 18.04 and Windows,
  //       for an unknown reason, multisampling does not work currently.
  //       Instead, when it was enabled, for some parts of the render window
  //       the window was semi-translucent, showing the windows behind it.
  glDisable(GL_MULTISAMPLE);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  glEnable(GL_BLEND);
  glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
  
  // Render.
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  //glShadeModel(GL_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  
  CHECK_OPENGL_NO_ERROR();
  
  
  // ### Rendering ###
  
  // Surfel rendering.
  if (render_surfels_) {
    RenderPointSplats();
    RenderCurrentFrameCloud();
  }
  
  // Camera frustum rendering.
  if (render_keyframes_) {
    RenderCameraFrustums();
  }
  if (render_current_frame_frustum_ && current_frame_pose_set_) {
    RenderCurrentFrameFrustum();
  }
  
  // Trajectory rendering.
  if (render_ground_truth_trajectory_) {
    RenderGroundTruthTrajectory();
  }
  if (render_estimated_trajectory_) {
    RenderEstimatedTrajectory();
  }
  
  
  // InitializeForCUDAInterop() body
  if (init_max_point_count_ > 0) {
    InitializeForCUDAInteropInRenderingThread();
  }
  
  
  // Take screenshot?
  unique_lock<mutex> screenshot_lock(screenshot_mutex_);
  if (!screenshot_path_.empty()) {
    SaveScreenshotImpl();
    
    screenshot_path_ = "";
    screenshot_lock.unlock();
    screenshot_condition_.notify_all();
  } else {
    screenshot_lock.unlock();
  }
};

void BadSlamRenderWindow::RenderPointSplats() {
  unique_lock<mutex> cloud_lock(visualization_cloud_mutex_);
  if (new_visualization_cloud_size_ != numeric_limits<usize>::max()) {
    // The vertex data is on the GPU already.
    have_visualization_cloud_ = true;
    visualization_cloud_size_ = new_visualization_cloud_size_;
    new_visualization_cloud_size_ = numeric_limits<usize>::max();
  }
  cloud_lock.unlock();
  
  // Render the visualization cloud if a cloud is available.
  if (have_visualization_cloud_) {
    splat_program_.UseProgram();
    splat_program_.SetUniformMatrix4f(
        splat_u_model_view_projection_matrix_location_,
        model_view_projection_matrix_ * pose_correction_matrix_);
    splat_program_.SetUniform1f(splat_u_point_size_x_location_, splat_half_extent_in_pixels_ / width_);
    splat_program_.SetUniform1f(splat_u_point_size_y_location_, splat_half_extent_in_pixels_ / height_);
    CHECK_OPENGL_NO_ERROR();
    
    if (visualization_cloud_size_ != numeric_limits<usize>::max()) {
      visualization_cloud_.SetAttributes(&splat_program_);
      // TODO: valgrind complains "Conditional jump or move depends on uninitialised value(s)" for the line below, not sure why
      glDrawArrays(GL_POINTS, 0, visualization_cloud_size_);
    } else {
      visualization_cloud_.Render(&splat_program_);
    }
    CHECK_OPENGL_NO_ERROR();
  }
}

void BadSlamRenderWindow::RenderCameraFrustums() {
  constant_color_program_.UseProgram();
  constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0.1f, 0.1f, 0.1f);
  
  glLineWidth(2);
  
  constexpr float kScaling = 0.1f;  // Equals the z (forward) extent of the frustum.
  Mat4f scaling_matrix = Mat4f::Identity();
  scaling_matrix(0, 0) = kScaling;
  scaling_matrix(1, 1) = kScaling;
  scaling_matrix(2, 2) = kScaling;
  
  Mat4f view_projection_matrix = projection_matrix_ * camera_T_world_.matrix() * pose_correction_matrix_;
  
  auto render_keyframe = [&](const Mat4f& global_T_keyframe) {
    Mat4f frustum_model_view_projection_matrix =
        view_projection_matrix * global_T_keyframe * scaling_matrix;  // TODO: precompute global_T_keyframe * scaling_matrix?
    constant_color_program_.SetUniformMatrix4f(
          constant_color_u_model_view_projection_matrix_location_,
          frustum_model_view_projection_matrix);
    
    camera_frustum_.Render(&constant_color_program_);
  };
  
  for (usize i = 0; i < global_T_keyframe_.size(); ++ i) {
    render_keyframe(global_T_keyframe_[i]);
  }
  for (usize i = 0; i < queued_global_T_keyframe_.size(); ++ i) {
    render_keyframe(queued_global_T_keyframe_[i]);
  }
  
  glLineWidth(1);
}

void BadSlamRenderWindow::RenderCurrentFrameFrustum() {
  constant_color_program_.UseProgram();
  constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0.1f, 0.1f, 1.0f);
  
  glLineWidth(2);
  
  constexpr float kScaling = 0.1f;  // Equals the z (forward) extent of the frustum.
  Mat4f scaling_matrix = Mat4f::Identity();
  scaling_matrix(0, 0) = kScaling;
  scaling_matrix(1, 1) = kScaling;
  scaling_matrix(2, 2) = kScaling;
  
  Mat4f view_projection_matrix = projection_matrix_ * camera_T_world_.matrix() * pose_correction_matrix_;
  
  Mat4f frustum_model_view_projection_matrix =
      view_projection_matrix * global_T_current_frame_ * scaling_matrix;
  constant_color_program_.SetUniformMatrix4f(
        constant_color_u_model_view_projection_matrix_location_,
        frustum_model_view_projection_matrix);
  
  camera_frustum_.Render(&constant_color_program_);
  
  glLineWidth(1);
}

void BadSlamRenderWindow::RenderCurrentFrameCloud() {
  if (current_frame_cloud_) {
    current_frame_cloud_set_ = true;
    
    current_frame_cloud_opengl_.TransferToGPU(*current_frame_cloud_, GL_DYNAMIC_DRAW);
    CHECK_OPENGL_NO_ERROR();
    
    current_frame_cloud_.reset();
  }
  
  // Render the cloud if a cloud is available.
  if (current_frame_cloud_set_) {
    model_view_projection_matrix_ = projection_matrix_ * camera_T_world_.matrix() * pose_correction_matrix_ * current_frame_cloud_global_T_frame_.matrix();
    
    splat_program_.UseProgram();
    splat_program_.SetUniformMatrix4f(
        splat_u_model_view_projection_matrix_location_,
        model_view_projection_matrix_);
    splat_program_.SetUniform1f(splat_u_point_size_x_location_, splat_half_extent_in_pixels_ / width_);
    splat_program_.SetUniform1f(splat_u_point_size_y_location_, splat_half_extent_in_pixels_ / height_);
    CHECK_OPENGL_NO_ERROR();
    
    current_frame_cloud_opengl_.Render(&splat_program_);
    CHECK_OPENGL_NO_ERROR();
  }
}

void BadSlamRenderWindow::RenderGroundTruthTrajectory() {
  if (!gt_trajectory_.empty()) {
    // Transfer vertices to the GPU.
    gt_trajectory_cloud_.TransferToGPU(sizeof(Vec3f), gt_trajectory_.size(), reinterpret_cast<float*>(gt_trajectory_.data()));
    gt_trajectory_.clear();
  }
  
  if (gt_trajectory_cloud_.buffer_allocated()) {
    constant_color_program_.UseProgram();
    constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0.1f, 0.9f, 0.1f);
    
    glLineWidth(2);
    
    // NOTE: Pose correction must not be applied here since this does not drift.
    constant_color_program_.SetUniformMatrix4f(
          constant_color_u_model_view_projection_matrix_location_,
          projection_matrix_ * camera_T_world_.matrix());
    
    gt_trajectory_cloud_.RenderAsLineStrip(&constant_color_program_);
    
    glLineWidth(1);
  }
}

void BadSlamRenderWindow::RenderEstimatedTrajectory() {
  if (!estimated_trajectory_.empty()) {
    // Transfer vertices to the GPU.
    estimated_trajectory_cloud_.TransferToGPU(sizeof(Vec3f), estimated_trajectory_.size(), reinterpret_cast<float*>(estimated_trajectory_.data()));  // TODO: , GL_DYNAMIC_DRAW));?
    estimated_trajectory_.clear();
  }
  
  if (estimated_trajectory_cloud_.buffer_allocated() &&
      estimated_trajectory_cloud_.cloud_size() >= 2) {
    constant_color_program_.UseProgram();
    constant_color_program_.SetUniform3f(constant_color_u_constant_color_location_, 0.9f, 0.1f, 0.1f);
    
    glLineWidth(2);
    
    constant_color_program_.SetUniformMatrix4f(
          constant_color_u_model_view_projection_matrix_location_,
          projection_matrix_ * camera_T_world_.matrix() * pose_correction_matrix_);
    
    estimated_trajectory_cloud_.RenderAsLineStrip(&constant_color_program_);
    
    glLineWidth(1);
  }
}

void BadSlamRenderWindow::MouseDown(MouseButton button, int x, int y) {
  pressed_mouse_buttons_ |= static_cast<int>(button);
  
  if (button == MouseButton::kLeft ||
      button == MouseButton::kMiddle) {
    last_drag_x_ = x;
    last_drag_y_ = y;
  }
  // TODO: Implement this functionality somewhere else (e.g., in the GUI keyframe dialog)
  //else if (button == MouseButton::kRight) {
//     if (embedded_in_gui_) {
//       return;
//     }
//     if (!dense_ba_) {
//       LOG(ERROR) << "SetDirectBA() must be called for the right-click functionality!";
//       return;
//     }
//     
//     constexpr bool kVisualizePointCloud = false;
//     constexpr bool kVisualizeColorProjection = true;
//     constexpr bool kVisualizeDescriptors = false;
//     
//     unique_lock<mutex> render_lock(render_mutex_);
//     int clicked_kf_index = DetermineClickedKeyframe(x, y);
//     
//     if (clicked_kf_index < 0) {
//       render_lock.unlock();
//       
//       LOG(INFO) << "Clicked no keyframe.";
//       
//       if (kVisualizePointCloud) {
//         UnsetFramePointCloud();
//       }
//       
//       RenderFrame();
//       return;
//     }
//     
//     Keyframe* keyframe = keyframe_ptrs_[clicked_kf_index];  // non-threadsafe access
//     
//     LOG(INFO) << "Clicked keyframe " << keyframe->id();
//     
//     shared_ptr<Point3fC3u8Cloud> current_frame_cloud;
//     
//     if (kVisualizeColorProjection) {
//       vector<shared_ptr<Keyframe>> single_keyframe_vector;
//       single_keyframe_vector.push_back(shared_ptr<Keyframe>(keyframe, [](Keyframe* /*k*/){}));  // HACK: Use empty deleter
//       AssignColorsCUDA(
//           /*stream*/ 0,
//           dense_ba_->color_camera(),
//           dense_ba_->depth_camera(),
//           dense_ba_->depth_params(),
//           single_keyframe_vector,
//           dense_ba_->surfels_size(),
//           dense_ba_->surfels().get());
//       
//       UpdateVisualizationBuffersCUDA(
//           /*stream*/ 0,
//           surfel_vertices_,
//           dense_ba_->surfels_size(),
//           dense_ba_->surfels()->ToCUDA(),
//           /*visualize_normals*/ false,
//           /*visualize_gradientmags*/ false,
//           /*visualize_radii*/ false);
//       UpdateVisualizationCloudCUDA(dense_ba_->surfels_size());
//       cudaDeviceSynchronize();
//     }
//     
//     if (kVisualizeDescriptors) {
//       vector<shared_ptr<Keyframe>> single_keyframe_vector;
//       single_keyframe_vector.push_back(shared_ptr<Keyframe>(keyframe, [](Keyframe* /*k*/){}));  // HACK: Use empty deleter
//       AssignDescriptorColorsCUDA(
//           /*stream*/ 0,
//           dense_ba_->color_camera(),
//           dense_ba_->depth_camera(),
//           dense_ba_->depth_params(),
//           single_keyframe_vector,
//           dense_ba_->surfels_size(),
//           dense_ba_->surfels().get());
//       
//       UpdateVisualizationBuffersCUDA(
//           /*stream*/ 0,
//           surfel_vertices_,
//           dense_ba_->surfels_size(),
//           dense_ba_->surfels()->ToCUDA(),
//           /*visualize_normals*/ false,
//           /*visualize_gradientmags*/ false,
//           /*visualize_radii*/ false);
//       UpdateVisualizationCloudCUDA(dense_ba_->surfels_size());
//       cudaDeviceSynchronize();
//     }
//     
//     if (kVisualizePointCloud) {
//       int depth_width = keyframe->depth_buffer().width();
//       int depth_height = keyframe->depth_buffer().height();
//       
//       Image<u16> depth_buffer(depth_width, depth_height);
//       keyframe->depth_buffer().DownloadAsync(/*stream*/ 0, &depth_buffer);
//       
//       Image<float> cfactor_buffer_cpu(dense_ba_->cfactor_buffer()->width(), dense_ba_->cfactor_buffer()->height());
//       dense_ba_->cfactor_buffer()->DownloadAsync(/*stream*/ 0, &cfactor_buffer_cpu);
//       
//       usize point_count = 0;
//       for (u32 y = 0; y < depth_buffer.height(); ++ y) {
//         const u16* ptr = depth_buffer.row(y);
//         const u16* end = ptr + depth_buffer.width();
//         while (ptr < end) {
//           if (!(*ptr & kInvalidDepthBit)) {
//             ++ point_count;
//           }
//           ++ ptr;
//         }
//       }
//       
//       current_frame_cloud.reset(new Point3fC3u8Cloud(point_count));
//       usize point_index = 0;
//       for (int y = 0; y < depth_height; ++ y) {
//         for (int x = 0; x < depth_width; ++ x) {
//           u16 depth_u16 = depth_buffer(x, y);
//           if (depth_u16 & kInvalidDepthBit) {
//             continue;
//           }
//           float depth = RawToCalibratedDepth(
//               dense_ba_->a(),
//               cfactor_buffer_cpu(x / dense_ba_->sparse_surfel_cell_size(),
//                                  y / dense_ba_->sparse_surfel_cell_size()),
//               dense_ba_->depth_params().raw_to_float_depth,
//               depth_u16);
//           
//           Point3fC3u8& point = current_frame_cloud->at(point_index);
//           point.position() = depth * dense_ba_->depth_camera().UnprojectFromPixelCenterConv(Vec2f(x, y));
//           point.color() = Vec3u8(80, 80, 255);
//           ++ point_index;
//         }
//       }
//     }
//     
//     render_lock.unlock();
//     
//     if (kVisualizePointCloud) {
//       SetFramePointCloud(
//           current_frame_cloud,
//           keyframe->global_T_frame());
//     }
//     RenderFrame();
//   }
}

void BadSlamRenderWindow::MouseMove(int x, int y) {
  bool move_camera = false;
  bool rotate_camera = false;
  
  move_camera = m_pressed_ ||
                (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kMiddle)) ||
                ((pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft)) &&
                  (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kRight)));
  rotate_camera = pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft);
  
  int x_distance = x - last_drag_x_;
  int y_distance = y - last_drag_y_;
  
  if ((x_distance != 0 || y_distance != 0) &&
      (pressed_mouse_buttons_ & static_cast<int>(MouseButton::kLeft))) {
    dragging_ = true;
  }

  if (move_camera) {
    const float right_phi = camera_free_orbit_phi_ + 0.5f * M_PI;
    const Eigen::Vector3f right_vector =
        Eigen::Vector3f(cosf(right_phi), sinf(right_phi), 0.f);
    const float up_theta = camera_free_orbit_theta_ + 0.5f * M_PI;
    const float phi = camera_free_orbit_phi_;
    const Eigen::Vector3f up_vector =
        -1 * Eigen::Vector3f(sinf(up_theta) * cosf(phi),
                              sinf(up_theta) * sinf(phi), cosf(up_theta));
    
    // Camera move speed in units per pixel for 1 unit orbit radius.
    constexpr float kCameraMoveSpeed = 0.001f;
    unique_lock<mutex> lock(camera_mutex_);
    camera_free_orbit_offset_ -= x_distance * kCameraMoveSpeed *
                                  camera_free_orbit_radius_ * right_vector;
    camera_free_orbit_offset_ += y_distance * kCameraMoveSpeed *
                                  camera_free_orbit_radius_ * up_vector;
    lock.unlock();
    
    window_->RenderFrame();
  } else if (rotate_camera) {
    unique_lock<mutex> lock(camera_mutex_);
    camera_free_orbit_theta_ -= y_distance * 0.01f;
    camera_free_orbit_phi_ -= x_distance * 0.01f;

    camera_free_orbit_theta_ = fmin(camera_free_orbit_theta_, 3.14f);
    camera_free_orbit_theta_ = fmax(camera_free_orbit_theta_, 0.01f);
    lock.unlock();
    
    window_->RenderFrame();
  }
  
  last_drag_x_ = x;
  last_drag_y_ = y;
}

void BadSlamRenderWindow::MouseUp(MouseButton button, int x, int y) {
  pressed_mouse_buttons_ &= ~static_cast<int>(button);
  
  if (embedded_in_gui_ && button == MouseButton::kLeft) {
    if (!dragging_) {
      // Left mouse click without dragging.
      if (tool_ == Tool::kSelectKeyframe) {
        unique_lock<mutex> render_lock(render_mutex_);
        int clicked_kf_index = DetermineClickedKeyframe(x, y);
        render_lock.unlock();
        
        if (clicked_kf_index >= 0) {
          LOG(INFO) << "Clicked KF: " << clicked_kf_index;
          signal_helper_.EmitClickedKeyframe(clicked_kf_index);
        }
      }
    }
    
    dragging_ = false;
  }
}

void BadSlamRenderWindow::WheelRotated(float degrees, Modifier modifiers) {
  double num_steps = -1 * (degrees / 15.0);
  
  if (static_cast<int>(modifiers) & static_cast<int>(RenderWindowCallbacks::Modifier::kCtrl)) {
    // Change point size.
    splat_half_extent_in_pixels_ -= 0.5f * num_steps;
    splat_half_extent_in_pixels_ = std::max(0.1f, splat_half_extent_in_pixels_);
  } else {
    // Zoom camera.
    double scale_factor = powf(powf(2.0, 1.0 / 5.0), num_steps);
    camera_free_orbit_radius_ *= scale_factor;
  }
  
  window_->RenderFrame();
}

void BadSlamRenderWindow::KeyPressed(char key, Modifier /*modifiers*/) {
  if (key == 'm') {
    m_pressed_ = true;
  } else if (key == 'c') {
    CopyView();
  } else if (key == 'v') {
    PasteView();
  } else if (key == 'u') {
    // TODO: Implement this functionality in the GUI in addition
    // Set up direction.
    unique_lock<mutex> lock(camera_mutex_);
    SE3f world_T_camera = camera_T_world_.inverse();
    Vec3f up_direction = world_T_camera * Vec3f(0, 0, -1.f);
    lock.unlock();
    SetUpDirection(up_direction);
    window_->RenderFrame();
  }
  
  if (embedded_in_gui_) {
    return;
  }
  
// TODO: Implement this functionality in the GUI instead
//    else if (key == 'r') {
//     // Toggle rotating camera.
//     use_rotating_camera_ = !use_rotating_camera_;
//     if (!use_rotating_camera_) {
//       follow_camera_initialized_ = false;
//     }
//     window_->RenderFrame();
//   } else if (key == 'k') {
//     // Add keyframe.
//     std::cout << "Enter frame index (starting at -1); prepend with d to delete; prepend with g to goto: " << std::flush;
//     entering_number = true;
//     text = "";
//   } else if (key == /*return*/ 13) {
//     entering_number = false;
//     std::cout << std::endl;
//     bool delete_keyframe = false;
//     bool goto_keyframe = false;
//     if (!text.empty() && text[0] == 'd') {
//       delete_keyframe = true;
//       text.erase(text.begin());
//     } else if (!text.empty() && text[0] == 'g') {
//       goto_keyframe = true;
//       text.erase(text.begin());
//     }
//     int frame_index = atoi(text.c_str());
//     
//     bool found = false;
//     for (usize i = 0; i < spline_frame_indices.size(); ++ i) {
//       if (frame_index == spline_frame_indices[i]) {
//         if (delete_keyframe) {
//           // Delete this keyframe.
//           spline_frame_indices.erase(spline_frame_indices.begin() + i);
//           offset_x_spline_points.erase(offset_x_spline_points.begin() + i);
//           offset_y_spline_points.erase(offset_y_spline_points.begin() + i);
//           offset_z_spline_points.erase(offset_z_spline_points.begin() + i);
//           radius_spline_points.erase(radius_spline_points.begin() + i);
//           theta_spline_points.erase(theta_spline_points.begin() + i);
//           phi_spline_points.erase(phi_spline_points.begin() + i);
//         } else if (goto_keyframe) {
//           Vec3f offset(offset_x_spline_points[i],
//                        offset_y_spline_points[i],
//                        offset_z_spline_points[i]);
//           SetViewParameters(offset, radius_spline_points[i], theta_spline_points[i], phi_spline_points[i], max_depth_);
//         } else {
//           // Replace this keyframe.
//           spline_frame_indices[i] = frame_index;
//           offset_x_spline_points[i] = FloatForSpline(camera_free_orbit_offset_.x());
//           offset_y_spline_points[i] = FloatForSpline(camera_free_orbit_offset_.y());
//           offset_z_spline_points[i] = FloatForSpline(camera_free_orbit_offset_.z());
//           radius_spline_points[i] = FloatForSpline(camera_free_orbit_radius_);
//           theta_spline_points[i] = FloatForSpline(camera_free_orbit_theta_);
//           phi_spline_points[i] = FloatForSpline(camera_free_orbit_phi_);
//         }
//         
//         found = true;
//         break;
//       } else if (frame_index < spline_frame_indices[i]) {
//         if (delete_keyframe || goto_keyframe) {
//           LOG(WARNING) << "Did not find the keyframe to delete / goto";
//           return;
//         }
//         // Insert the new keyframe before this keyframe.
//         spline_frame_indices.insert(spline_frame_indices.begin() + i, frame_index);
//         offset_x_spline_points.insert(offset_x_spline_points.begin() + i, FloatForSpline(camera_free_orbit_offset_.x()));
//         offset_y_spline_points.insert(offset_y_spline_points.begin() + i, FloatForSpline(camera_free_orbit_offset_.y()));
//         offset_z_spline_points.insert(offset_z_spline_points.begin() + i, FloatForSpline(camera_free_orbit_offset_.z()));
//         radius_spline_points.insert(radius_spline_points.begin() + i, FloatForSpline(camera_free_orbit_radius_));
//         theta_spline_points.insert(theta_spline_points.begin() + i, FloatForSpline(camera_free_orbit_theta_));
//         phi_spline_points.insert(phi_spline_points.begin() + i, FloatForSpline(camera_free_orbit_phi_));
//         
//         found = true;
//         break;
//       }
//     }
//     // If the new keyframe was not inserted yet, append it at the end.
//     if (!found) {
//       if (delete_keyframe || goto_keyframe) {
//         LOG(WARNING) << "Did not find the keyframe to delete / goto";
//         return;
//       }
//       spline_frame_indices.push_back(frame_index);
//       offset_x_spline_points.push_back(FloatForSpline(camera_free_orbit_offset_.x()));
//       offset_y_spline_points.push_back(FloatForSpline(camera_free_orbit_offset_.y()));
//       offset_z_spline_points.push_back(FloatForSpline(camera_free_orbit_offset_.z()));
//       radius_spline_points.push_back(FloatForSpline(camera_free_orbit_radius_));
//       theta_spline_points.push_back(FloatForSpline(camera_free_orbit_theta_));
//       phi_spline_points.push_back(FloatForSpline(camera_free_orbit_phi_));
//     }
//     
//     // Re-build the splines.
//     offset_x_spline.reset(new UniformCRSpline<FloatForSpline>(offset_x_spline_points));
//     offset_y_spline.reset(new UniformCRSpline<FloatForSpline>(offset_y_spline_points));
//     offset_z_spline.reset(new UniformCRSpline<FloatForSpline>(offset_z_spline_points));
//     radius_spline.reset(new UniformCRSpline<FloatForSpline>(radius_spline_points));
//     theta_spline.reset(new UniformCRSpline<FloatForSpline>(theta_spline_points));
//     phi_spline.reset(new UniformCRSpline<FloatForSpline>(phi_spline_points));
//     
//     // Print keyframe info
//     std::cout << "### Keyframes ###" << std::endl;
//     for (usize i = 0; i < spline_frame_indices.size(); ++ i) {
//       std::cout << "keyframe " << spline_frame_indices[i]
//                                << " " << offset_x_spline_points[i]
//                                << " " << offset_y_spline_points[i]
//                                << " " << offset_z_spline_points[i]
//                                << " " << radius_spline_points[i]
//                                << " " << theta_spline_points[i]
//                                << " " << phi_spline_points[i]
//                                << std::endl;
//     }
//   } else if (key == 'p') {
//     // Play back the keyframe-based animation.
//     if (spline_frame_indices.size() < 4) {
//       LOG(ERROR) << "Playback requires at least 4 keyframes.";
//       return;
//     }
//     
//     render_playback_ = (modifiers == RenderWindowCallbacks::Modifier::kCtrl);
//     
//     if (playback_frame_ < 0) {
//       playback_frame_ = 0;
//     } else {
//       playback_frame_ = -1;
//     }
//     window_->RenderFrame();
//   }
}

void BadSlamRenderWindow::KeyReleased(char key, Modifier /*modifiers*/) {
  if (key == 'm') {
    m_pressed_ = false;
  }
}

void BadSlamRenderWindow::InitializeForCUDAInterop(
    usize max_point_count,
    OpenGLContext* context,
    OpenGLContext* context2,
    const Camera& camera) {
  // Unfortunately, it is not possible to make a QOpenGLContext current in
  // another thread, so we have to send these variables to the render thread.
  // This thread will then call InitializeForCUDAInteropInRenderingThread()
  // while the context is active to do the actial initialization.
  
  unique_lock<mutex> render_lock(render_mutex_);
  init_max_point_count_ = max_point_count;
  init_camera_ = &camera;
  init_done_ = false;
  render_lock.unlock();
  
  window_->RenderFrame();
  
  unique_lock<mutex> init_lock(init_mutex_);
  while (!init_done_) {
    init_condition_.wait(init_lock);
  }
  init_lock.unlock();
  
  // Initialize an OpenGL context which shares names with the Qt OpenGL context.
  context->InitializeWindowless(&qt_gl_context_);
  context2->InitializeWindowless(&qt_gl_context_);
  qt_gl_context_.Detach();
}

void BadSlamRenderWindow::InitializeForCUDAInteropInRenderingThread() {
  // Initialize vertex buffer for surfels.
  visualization_cloud_.AllocateBuffer(
      init_max_point_count_ * sizeof(Point3fC3u8), GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  
  // Register the vertex buffer with CUDA.
  cudaGraphicsGLRegisterBuffer(&surfel_vertices_,
                               visualization_cloud_.buffer_name(),
                               cudaGraphicsMapFlagsWriteDiscard);
  CHECK_CUDA_NO_ERROR();
  
  // Create camera frustum.
  camera_frustum_.Create(*init_camera_);
  
  // Store the context.
  qt_gl_context_.AttachToCurrent();
  
  
  // Reset init_max_point_count_ such that this function is not called again.
  init_max_point_count_ = 0;
  
  // Signal completion.
  unique_lock<mutex> init_lock(init_mutex_);
  init_done_ = true;
  init_lock.unlock();
  init_condition_.notify_all();
}

void BadSlamRenderWindow::SaveScreenshotImpl() {
  // QOpenGLWidget uses an FBO for its rendering, and we activate multisampling
  // for it. It is not possible to directly use glReadPixels() on a multisampled
  // FBO. Instead, we have to create a temporary non-multisampled FBO first,
  // use glBlitFramebuffer() to copy the content of the multisampled buffer into
  // it (while converting to single-sample format), and then read from that.
  
  GLuint default_fbo = dynamic_cast<RenderWindowQtOpenGL*>(window_)->widget()->defaultFramebufferObject();
  
  // Create temporary non-multisampled FBO
  GLuint fbo_id;
  glGenFramebuffers(1, &fbo_id);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_id);
  
  // Create a texture to attach to the FBO as color attachment
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  CHECK_OPENGL_NO_ERROR();
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 0,
                          GL_TEXTURE_2D, texture, 0);
  CHECK_OPENGL_NO_ERROR();
  glBindTexture(GL_TEXTURE_2D, 0);
  
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Framebuffer not complete";
  }
  
  // Write the contents of the default FBO into the temporary FBO
  glBindFramebuffer(GL_READ_FRAMEBUFFER, default_fbo);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_id);
  glBlitFramebuffer(0, 0, width_, height_, 0, 0, width_, height_, GL_COLOR_BUFFER_BIT, GL_NEAREST);
  CHECK_OPENGL_NO_ERROR();
  
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_id); // must rebind, otherwise it won't work
  
  // Read the pixels
  Image<Vec4u8> image(width_, height_, width_ * sizeof(Vec4u8), 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, image.data());
  CHECK_OPENGL_NO_ERROR();
  
  // Clean up the temporary buffer
  glBindFramebuffer(GL_FRAMEBUFFER, default_fbo);
  glDeleteTextures(1, &texture);
  glDeleteFramebuffers(1, &fbo_id);
  
  // Convert the image from pre-multiplied alpha to standard (non-pre-multiplied) alpha
  for (int y = 0; y < image.height(); ++ y) {
    for (int x = 0; x < image.width(); ++ x) {
      Vec4u8& color = image(x, y);
      if (color(3) != 0) {
        color.topRows<3>() = ((color.topRows<3>().cast<float>() * 255.f / color(3)) + Vec3f::Constant(0.5f)).cast<u8>();
      }
    }
  }
  
  // Make (0, 0) be at the top-left of the image instead of bottom-left
  image.FlipY();
  
//   // Blend with a background image:
//   static Image<Vec3u8> background_image;
//   if (background_image.empty()) {
//     CHECK(background_image.Read("/path/to/image.png"));
//   }
//   CHECK_EQ(image.size(), background_image.size());
//   for (int y = 0; y < image.height(); ++ y) {
//     for (int x = 0; x < image.width(); ++ x) {
//       Vec4u8& color = image(x, y);
//       const Vec3u8& bg_color = background_image(x, y);
//       
//       float alpha = color(3) / 255.f;
//       color.topRows<3>() = ((color.topRows<3>().cast<float>() * alpha + 
//                               bg_color.cast<float>() * (1 - alpha)) + Vec3f::Constant(0.5f)).cast<u8>();
//       color(3) = 255;
//     }
//   }
  
  image.Write(screenshot_path_);
}

void BadSlamRenderWindow::UpdateVisualizationCloudCUDA(u32 surfel_count) {
  unique_lock<mutex> lock(visualization_cloud_mutex_);
  new_visualization_cloud_size_ = surfel_count;
}

void BadSlamRenderWindow::SetPoseCorrectionNoLock(const SE3f& pose_correction) {
  pose_correction_matrix_ = pose_correction.matrix();
}

void BadSlamRenderWindow::SetCameraNoLock(const PinholeCamera4f& camera) {
  camera_update_ = true;
  updated_camera_ = camera;
}

void BadSlamRenderWindow::SetKeyframePosesNoLock(vector<Mat4f>&& global_T_keyframe, vector<int>&& keyframe_ids) {
  global_T_keyframe_ = std::move(global_T_keyframe);
  keyframe_ids_ = std::move(keyframe_ids);
}

void BadSlamRenderWindow::SetQueuedKeyframePosesNoLock(vector<Mat4f>&& global_T_keyframe, vector<int>&& keyframe_ids) {
  queued_global_T_keyframe_ = std::move(global_T_keyframe);
  queued_keyframe_ids_ = std::move(keyframe_ids);
}

void BadSlamRenderWindow::SetCurrentFramePose(const Mat4f& global_T_current_frame) {
  unique_lock<mutex> render_lock(render_mutex_);
  global_T_current_frame_ = global_T_current_frame;
  current_frame_pose_set_ = true;
}

void BadSlamRenderWindow::SetCurrentFramePoseNoLock(const Mat4f& global_T_current_frame) {
  global_T_current_frame_ = global_T_current_frame;
  current_frame_pose_set_ = true;
}

void BadSlamRenderWindow::SetGroundTruthTrajectory(vector<Vec3f>& gt_trajectory) {
  unique_lock<mutex> render_lock(render_mutex_);
  gt_trajectory_ = gt_trajectory;
}

void BadSlamRenderWindow::SetEstimatedTrajectoryNoLock(vector<Vec3f>&& estimated_trajectory) {
  estimated_trajectory_ = std::move(estimated_trajectory);
}

void BadSlamRenderWindow::SetFramePointCloud(
    const shared_ptr<Point3fC3u8Cloud>& cloud,
    const SE3f& global_T_frame) {
  unique_lock<mutex> render_lock(render_mutex_);
  current_frame_cloud_ = cloud;
  current_frame_cloud_global_T_frame_ = global_T_frame;
  current_frame_cloud_set_ = true;
}

void BadSlamRenderWindow::UnsetFramePointCloud() {
  current_frame_cloud_set_ = false;
}

void BadSlamRenderWindow::SetUpDirection(const Vec3f& direction) {
  unique_lock<mutex> lock(camera_mutex_);
  up_direction_rotation_ = Quaternionf::FromTwoVectors(direction, Vec3f(0, 0, 1)).toRotationMatrix();
}

void BadSlamRenderWindow::CenterViewOn(const Vec3f& position) {
  unique_lock<mutex> lock(camera_mutex_);
  SE3f up_direction_rotation_transformation =
      SE3f(up_direction_rotation_, Vec3f::Zero());
  camera_free_orbit_offset_ = up_direction_rotation_transformation * position;
}

void BadSlamRenderWindow::SetView(const Vec3f& look_at, const Vec3f& camera_pos) {
  unique_lock<mutex> lock(camera_mutex_);
  
  SE3f up_direction_rotation_transformation =
      SE3f(up_direction_rotation_, Vec3f::Zero());
  
  use_camera_matrix_ = false;
  
  camera_free_orbit_offset_ = up_direction_rotation_transformation * look_at;
  
  Vec3f look_at_to_camera = up_direction_rotation_ * (camera_pos - look_at);
  camera_free_orbit_radius_ = look_at_to_camera.norm();
  camera_free_orbit_theta_ = acos(look_at_to_camera.z() / camera_free_orbit_radius_);
  camera_free_orbit_phi_ = atan2(look_at_to_camera.y(), look_at_to_camera.x());
}

void BadSlamRenderWindow::SetView2(const Vec3f& x, const Vec3f& y, const Vec3f& z, const Vec3f& eye) {
  unique_lock<mutex> lock(camera_mutex_);
  
  use_camera_matrix_ = true;
  camera_matrix_ << x(0),  x(1),  x(2),  -(x.dot(eye)),
                    y(0),  y(1),  y(2),  -(y.dot(eye)),
                    z(0),  z(1),  z(2),  -(z.dot(eye)),
                       0,     0,     0,              1;
}

void BadSlamRenderWindow::SetViewParameters(
    const Vec3f& camera_free_orbit_offset,
    float camera_free_orbit_radius,
    float camera_free_orbit_theta,
    float camera_free_orbit_phi,
    float max_depth) {
  unique_lock<mutex> lock(camera_mutex_);
  
  use_camera_matrix_ = false;
  
  camera_free_orbit_offset_ = camera_free_orbit_offset;
  camera_free_orbit_radius_ = camera_free_orbit_radius;
  camera_free_orbit_theta_ = camera_free_orbit_theta;
  camera_free_orbit_phi_ = camera_free_orbit_phi;
  
  max_depth_ = max_depth;
}

void BadSlamRenderWindow::CopyView() {
  unique_lock<mutex> lock(camera_mutex_);
  
  QClipboard* clipboard = QApplication::clipboard();
  clipboard->setText(
      QString::number(camera_free_orbit_offset_.x()) + " " +
      QString::number(camera_free_orbit_offset_.y()) + " " +
      QString::number(camera_free_orbit_offset_.z()) + " " +
      QString::number(camera_free_orbit_radius_) + " " +
      QString::number(camera_free_orbit_theta_) + " " +
      QString::number(camera_free_orbit_phi_));
}

void BadSlamRenderWindow::PasteView() {
  QClipboard* clipboard = QApplication::clipboard();
  QString text = clipboard->text();
  QStringList list = text.split(' ');
  if (list.size() != 6) {
    LOG(ERROR) << "Cannot parse clipboard content as camera pose!";
  } else {
    unique_lock<mutex> lock(camera_mutex_);
    
    camera_free_orbit_offset_.x() = list[0].toFloat();
    camera_free_orbit_offset_.y() = list[1].toFloat();
    camera_free_orbit_offset_.z() = list[2].toFloat();
    camera_free_orbit_radius_ = list[3].toFloat();
    camera_free_orbit_theta_ = list[4].toFloat();
    camera_free_orbit_phi_ = list[5].toFloat();
    
    lock.unlock();
    window_->RenderFrame();
  }
}

void BadSlamRenderWindow::UseFollowCamera(bool enable) {
  use_follow_camera_ = enable;
}

void BadSlamRenderWindow::ChangeSplatSize(int num_steps) {
  splat_half_extent_in_pixels_ += 0.5f * num_steps;
  splat_half_extent_in_pixels_ = std::max(0.1f, splat_half_extent_in_pixels_);
}

void BadSlamRenderWindow::RenderFrame() {
  window_->RenderFrame();
}

void BadSlamRenderWindow::SaveScreenshot(const char* filepath, bool process_events) {
  // Use a transparent background
  Vec4f old_background_color = background_color_;
  background_color_ = Vec4f::Zero();
  
  // Use render_lock to make sure that a new frame is rendered for the screenshot.
  // This way any previous calls to update the camera pose, for example, should
  // take effect. The rendering iteration will call SaveScreenshotImpl() at the end.
  unique_lock<mutex> render_lock(render_mutex_);
  unique_lock<mutex> lock(screenshot_mutex_);
  screenshot_path_ = filepath;
  lock.unlock();
  render_lock.unlock();
  
  window_->RenderFrame();
  
  unique_lock<mutex> lock2(screenshot_mutex_);
  while (!screenshot_path_.empty()) {
    if (process_events) {
      lock2.unlock();
      QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
      lock2.lock();
    } else {
      screenshot_condition_.wait(lock2);
    }
  }
  lock2.unlock();
  
  background_color_ = old_background_color;
}

void BadSlamRenderWindow::SetCamera() {
  float camera_parameters[4];
  camera_parameters[0] = height_;  // fx
  camera_parameters[1] = height_;  // fy
  camera_parameters[2] = 0.5 * width_ - 0.5f;  // cx
  camera_parameters[3] = 0.5 * height_ - 0.5f;  // cy
  render_camera_ = PinholeCamera4f(width_, height_, camera_parameters);
}

void BadSlamRenderWindow::SetViewpoint() {
  if (use_camera_matrix_) {
    camera_T_world_ = SE3f(camera_matrix_);
  } else {
    Vec3f look_at = camera_free_orbit_offset_;
    float r = camera_free_orbit_radius_;
    float t = camera_free_orbit_theta_;
    float p = camera_free_orbit_phi_;
    Vec3f look_from =
        look_at + Vec3f(r * sinf(t) * cosf(p), r * sinf(t) * sinf(p),
                                  r * cosf(t));
    
    Vec3f forward = (look_at - look_from).normalized();
    Vec3f up_temp = Vec3f(0, 0, 1);
    Vec3f right = forward.cross(up_temp).normalized();
    Vec3f up = right.cross(forward);
    
    Mat3f world_R_camera;
    world_R_camera.col(0) = right;
    world_R_camera.col(1) = -up;  // Y will be mirrored by the projection matrix to remove the discrepancy between OpenGL's and our coordinate system.
    world_R_camera.col(2) = forward;
    
    SE3f world_T_camera(world_R_camera, look_from);
    camera_T_world_ = world_T_camera.inverse();
    
    SE3f up_direction_rotation_transformation =
        SE3f(up_direction_rotation_, Vec3f::Zero());
    camera_T_world_ = camera_T_world_ * up_direction_rotation_transformation;
  }
}

void BadSlamRenderWindow::ComputeProjectionMatrix() {
  CHECK_GT(max_depth_, min_depth_);
  CHECK_GT(min_depth_, 0);

  const float fx = render_camera_.parameters()[0];
  const float fy = render_camera_.parameters()[1];
  const float cx = render_camera_.parameters()[2];
  const float cy = render_camera_.parameters()[3];

  // Row-wise projection matrix construction.
  projection_matrix_(0, 0) = (2 * fx) / render_camera_.width();
  projection_matrix_(0, 1) = 0;
  projection_matrix_(0, 2) = 2 * (0.5f + cx) / render_camera_.width() - 1.0f;
  projection_matrix_(0, 3) = 0;
  
  projection_matrix_(1, 0) = 0;
  projection_matrix_(1, 1) = -1 * ((2 * fy) / render_camera_.height());
  projection_matrix_(1, 2) = -1 * (2 * (0.5f + cy) / render_camera_.height() - 1.0f);
  projection_matrix_(1, 3) = 0;
  
  projection_matrix_(2, 0) = 0;
  projection_matrix_(2, 1) = 0;
  projection_matrix_(2, 2) = (max_depth_ + min_depth_) / (max_depth_ - min_depth_);
  projection_matrix_(2, 3) = -(2 * max_depth_ * min_depth_) / (max_depth_ - min_depth_);
  
  projection_matrix_(3, 0) = 0;
  projection_matrix_(3, 1) = 0;
  projection_matrix_(3, 2) = 1;
  projection_matrix_(3, 3) = 0;
}

void BadSlamRenderWindow::SetupViewport() {
  glViewport(0, 0, render_camera_.width(), render_camera_.height());
}

void BadSlamRenderWindow::CreateSplatProgram() {
  // for brightening up ETH3D SLAM datasets, replace the color pass-through with:
  // "  var1_color = in_color + vec3(0.2, 0.2, 0.2);\n"
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "out vec3 var1_color;\n"
      "void main() {\n"
      "  var1_color = in_color;\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_EXT_geometry_shader : enable\n"
      "layout(points) in;\n"
      "layout(triangle_strip, max_vertices = 4) out;\n"
      "\n"
      "uniform float u_point_size_x;\n"
      "uniform float u_point_size_y;\n"
      "\n"
      "in vec3 var1_color[];\n"
      "out vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  var2_color = var1_color[0];\n"
      "  vec4 base_pos = vec4(gl_in[0].gl_Position.xyz / gl_in[0].gl_Position.w, 1.0);\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, -u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(-u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  gl_Position = base_pos + vec4(u_point_size_x, u_point_size_y, 0.0, 0.0);\n"
      "  EmitVertex();\n"
      "  \n"
      "  EndPrimitive();\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kGeometryShader));
  
  CHECK(splat_program_.AttachShader(
      "#version 150\n"
      "#extension GL_ARB_explicit_attrib_location : enable\n"
      "layout(location = 0) out lowp vec4 out_color;\n"
      "\n"
      "in lowp vec3 var2_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = vec4(var2_color, 1.0);\n"
      // For highlighting the splats in red:
//      "  out_color = vec3(1.0, 0.0, 0.0);\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(splat_program_.LinkProgram());
  
  splat_program_.UseProgram();
  
  splat_u_model_view_projection_matrix_location_ =
      splat_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  splat_u_point_size_x_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_x");
  splat_u_point_size_y_location_ =
      splat_program_.GetUniformLocationOrAbort("u_point_size_y");
}

void BadSlamRenderWindow::CreateConstantColorProgram() {
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "uniform mat4 u_model_view_projection_matrix;\n"
      "in vec4 in_position;\n"
      "in vec3 in_color;\n"
      "void main() {\n"
      "  gl_Position = u_model_view_projection_matrix * in_position;\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kVertexShader));
  
  CHECK(constant_color_program_.AttachShader(
      "#version 300 es\n"
      "layout(location = 0) out lowp vec4 out_color;\n"
      "\n"
      "uniform lowp vec3 u_constant_color;\n"
      "\n"
      "void main() {\n"
      "  out_color = vec4(u_constant_color, 1.0);\n"
      "}\n",
      ShaderProgramOpenGL::ShaderType::kFragmentShader));
  
  CHECK(constant_color_program_.LinkProgram());
  
  constant_color_program_.UseProgram();
  
  constant_color_u_model_view_projection_matrix_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_model_view_projection_matrix");
  constant_color_u_constant_color_location_ =
      constant_color_program_.GetUniformLocationOrAbort("u_constant_color");
}

int BadSlamRenderWindow::DetermineClickedKeyframe(int x, int y) {
  unique_lock<mutex> camera_mutex_lock(camera_mutex_);
  Mat4f camera_T_global = camera_T_world_.matrix() * pose_correction_matrix_;
  camera_mutex_lock.unlock();
  
  Vec2f click_pos(x, y);
  
  float smallest_kf_dist_sq = numeric_limits<float>::infinity();
  int closest_kf_id = -1;
  
  auto determine_clicked = [&](const Mat4f& global_T_keyframe, int id) {
    Vec3f cam_kf_origin = (camera_T_global * global_T_keyframe.block<4, 1>(0, 3)).hnormalized();
    if (cam_kf_origin.z() > 0) {
      Vec2f image_kf_origin = render_camera_.ProjectToPixelCenterConv(cam_kf_origin);
      float kf_dist_sq = (image_kf_origin - click_pos).squaredNorm();
      
      if (kf_dist_sq < smallest_kf_dist_sq) {
        smallest_kf_dist_sq = kf_dist_sq;
        closest_kf_id = id;
      }
    }
  };
  
  for (usize i = 0; i < global_T_keyframe_.size(); ++ i) {
    determine_clicked(global_T_keyframe_[i], keyframe_ids_[i]);
  }
  for (usize i = 0; i < queued_global_T_keyframe_.size(); ++ i) {
    determine_clicked(queued_global_T_keyframe_[i], queued_keyframe_ids_[i]);
  }
  
  return (smallest_kf_dist_sq > 80 * 80) ? -1 : closest_kf_id;
}

}

#include "render_window.moc"
