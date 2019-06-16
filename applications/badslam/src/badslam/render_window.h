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

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <cuda_runtime.h>
#include <libvis/camera.h>
#include <libvis/camera_frustum_opengl.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/mesh_opengl.h>
#include <libvis/opengl.h>
#include <libvis/opengl_context.h>
#include <libvis/point_cloud_opengl.h>
#include <libvis/render_window.h>
#include <libvis/sophus.h>
#include <spline_library/splines/uniform_cr_spline.h>
#include <QObject>

#include <cuda_gl_interop.h>  // Must be included late so that glew.h is included before gl.h

#include <libvis/image.h>

namespace vis {

class DirectBA;
class Keyframe;
struct TimerHelper;

class BadSlamRenderWindowSignalHelper : public QObject {
 Q_OBJECT
 signals:
  void ClickedKeyframe(int clicked_kf_index);
 public:
  inline void EmitClickedKeyframe(int clicked_kf_index) { emit ClickedKeyframe(clicked_kf_index); }
};

// Renders the 3D visualization for BAD SLAM.
class BadSlamRenderWindow : public RenderWindowCallbacks {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  enum class Tool {
    kNoTool = 0,
    kSelectKeyframe
  };
  
  // Constructor. No OpenGL context required to call this.
  BadSlamRenderWindow(
      float splat_half_extent_in_pixels,
      bool embedded_in_gui = false);
  
  // Destructor, unregisters the CUDA-OpenGL interoperation.
  virtual ~BadSlamRenderWindow();
  
  // Tells the render window about the DirectBA instance. Required for some
  // debugging functionality only: right-clicking keyframes can access the
  // keyframe data then.
  void SetDirectBA(DirectBA* dense_ba);
  
  // Overrides from RenderWindowCallbacks for initialization, rendering, and
  // window event handling. NOTE: These must be called from the Qt thread only.
  // Do not call these yourself.
  
  virtual void Initialize() override;
  virtual void Deinitialize() override;
  
  virtual void Resize(int width, int height) override;
  
  virtual void Render() override;
  
  virtual void MouseDown(MouseButton button, int x, int y) override;
  virtual void MouseMove(int x, int y) override;
  virtual void MouseUp(MouseButton button, int x, int y) override;
  virtual void WheelRotated(float degrees, Modifier modifiers) override;
  virtual void KeyPressed(char key, Modifier modifiers) override;
  virtual void KeyReleased(char key, Modifier modifiers) override;
  
  // Performs initializations, sets up CUDA-OpenGL interoperation.
  void InitializeForCUDAInterop(
      usize max_point_count,
      OpenGLContext* context,
      OpenGLContext* context2,
      const Camera& camera);
  
  // Updates the surfel visualization.
  void UpdateVisualizationCloudCUDA(u32 surfel_count);
  
  // Updates the camera frustum. The render mutex must be locked already.
  void SetCameraNoLock(const PinholeCamera4f& camera);
  
  // Updates the pose correction transformation. The render mutex must be locked
  // already.
  void SetPoseCorrectionNoLock(const SE3f& pose_correction);
  
  // Updates the keyframe poses. The render mutex must be locked already.
  void SetKeyframePosesNoLock(
      vector<Mat4f>&& global_T_keyframe, vector<int>&& keyframe_ids);
  
  // Updates the keyframe poses. The render mutex must be locked already.
  void SetQueuedKeyframePosesNoLock(
      vector<Mat4f>&& global_T_keyframe, vector<int>&& keyframe_ids);
  
  // Updates the current frame's pose.
  void SetCurrentFramePose(const Mat4f& global_T_current_frame);
  
  // Updates the current frame's pose. The render mutex must be locked already.
  void SetCurrentFramePoseNoLock(const Mat4f& global_T_current_frame);
  
  // Updates the ground truth trajectory.
  void SetGroundTruthTrajectory(vector<Vec3f>& gt_trajectory);
  
  // Udpates the estimated trajectory. The render mutex must be locked already.
  void SetEstimatedTrajectoryNoLock(vector<Vec3f>&& estimated_trajectory);
  
  // Udpates the displayed point cloud (corresponding to a single frame).
  void SetFramePointCloud(
      const shared_ptr<Point3fC3u8Cloud>& cloud,
      const SE3f& global_T_frame);
  
  // Clears the displayed point cloud (corresponding to a single frame).
  void UnsetFramePointCloud();
  
  // Sets the direction which is "up" (for camera control).
  void SetUpDirection(const Vec3f& direction);
  
  // Centers the view on the given position.
  void CenterViewOn(const Vec3f& position);
  
  // Sets a view which is computed from the given look_at point and camera
  // position. Also sets the input camera pose.
  void SetView(const Vec3f& look_at, const Vec3f& camera_pos);
  
  // Sets an arbitrary view with the given axes and eye position. Also sets
  // the input camera pose.
  void SetView2(const Vec3f& x, const Vec3f& y, const Vec3f& z, const Vec3f& eye);
  
  // Sets the view parameters directly. Also sets the input camera pose.
  void SetViewParameters(
      const Vec3f& camera_free_orbit_offset,
      float camera_free_orbit_radius,
      float camera_free_orbit_theta,
      float camera_free_orbit_phi,
      float max_depth);
  
  // Copy the view pose (as plain text)
  void CopyView();
  
  // Paste a copied view pose
  void PasteView();
  
  // Makes the visualization use a camera that automatically follows the
  // trajectory with a delay. This also activates regular rendering updates
  // of the visualization, such that this camera movement can be shown.
  void UseFollowCamera(bool enable);
  
  // Change the size that the surfel splats are rendered with.
  void ChangeSplatSize(int num_steps);
  
  // Invokes rendering of a frame. Useful to cause re-rendering after a change.
  void RenderFrame();
  
  // Takes and saves a screenshot of the rendering window.
  void SaveScreenshot(const char* filepath, bool process_events);
  
  inline void SetRenderCurrentFrameFrustum(bool enable) { render_current_frame_frustum_ = enable; }
  inline void SetRenderEstimatedTrajectory(bool enable) { render_estimated_trajectory_ = enable; }
  inline void SetRenderGroundTruthTrajectory(bool enable) { render_ground_truth_trajectory_ = enable; }
  inline void SetRenderKeyframes(bool enable) { render_keyframes_ = enable; }
  inline void SetRenderSurfels(bool enable) { render_surfels_ = enable; }
  
  inline void SetTool(Tool tool) { tool_ = tool; }
  
  // Returns the render mutex, such that it can be locked. This must be done for
  // calling the functions with suffix "NoLock".
  inline std::mutex& render_mutex() { return render_mutex_; }
  inline const std::mutex& render_mutex() const { return render_mutex_; }
  
  // Returns the CUDA-OpenGL interop resource that allows updating the surfel
  // vertex buffer from CUDA.
  inline cudaGraphicsResource_t surfel_vertices() const { return surfel_vertices_; }
  
  inline const BadSlamRenderWindowSignalHelper& signal_helper() const { return signal_helper_; }
  inline BadSlamRenderWindowSignalHelper& signal_helper() { return signal_helper_; }
  
  
 private:
  void RenderPointSplats();
  void RenderSparseKeypoints();
  void RenderCameraFrustums();
  void RenderCurrentFrameFrustum();
  void RenderCurrentFrameCloud();
  void RenderGroundTruthTrajectory();
  void RenderEstimatedTrajectory();
  
  void InitializeForCUDAInteropInRenderingThread();
  void SaveScreenshotImpl();
  
  void SetCamera();
  void SetViewpoint();
  void ComputeProjectionMatrix();
  void SetupViewport();
  
  void CreateSplatProgram();
  void CreateConstantColorProgram();
  
  // render_mutex_ must be locked when calling this.
  // If no keyframe is near the given position, -1 will be returned.
  int DetermineClickedKeyframe(int x, int y);
  
  
  // Settings.
  float splat_half_extent_in_pixels_;
  bool render_current_frame_frustum_;
  bool render_estimated_trajectory_;
  bool render_ground_truth_trajectory_;
  bool render_keyframes_;
  bool render_surfels_;
  
  Vec4f background_color_;
  
  int width_;
  int height_;
  
  // Input handling.
  bool dragging_;
  int last_drag_x_;
  int last_drag_y_;
  int pressed_mouse_buttons_;
  bool m_pressed_;
  
  // Render camera and pose.
  SE3f camera_T_world_;
  PinholeCamera4f render_camera_;
  
  float min_depth_;
  float max_depth_;
  
  std::mutex camera_mutex_;
  Mat3f up_direction_rotation_;
  Vec3f camera_free_orbit_offset_;
  float camera_free_orbit_radius_;
  float camera_free_orbit_theta_;
  float camera_free_orbit_phi_;
  
  Mat4f camera_matrix_;
  bool use_camera_matrix_;
  
  Mat4f projection_matrix_;
  Mat4f model_view_projection_matrix_;
  
  bool camera_update_;
  PinholeCamera4f updated_camera_;
  CameraFrustumOpenGL camera_frustum_;
  
  // Vertex buffer handling.
  std::mutex visualization_cloud_mutex_;
  bool have_visualization_cloud_;
  Point3fC3u8CloudOpenGL visualization_cloud_;
  std::shared_ptr<Point3fC3u8Cloud> current_visualization_cloud_;
  usize visualization_cloud_size_;
  usize new_visualization_cloud_size_;
  cudaGraphicsResource_t surfel_vertices_;
  
  // Pose correction (for correcting movements of the anchor keyframe).
  Mat4f pose_correction_matrix_;
  
  // Keyframes.
  vector<Mat4f> global_T_keyframe_;
  vector<int> keyframe_ids_;
  vector<Mat4f> queued_global_T_keyframe_;
  vector<int> queued_keyframe_ids_;
  
  // Current frame.
  Mat4f global_T_current_frame_;
  bool current_frame_pose_set_;
  
  shared_ptr<Point3fC3u8Cloud> current_frame_cloud_;
  Point3fC3u8CloudOpenGL current_frame_cloud_opengl_;
  SE3f current_frame_cloud_global_T_frame_;
  atomic<bool> current_frame_cloud_set_;
  
  // Ground truth trajectory.
  vector<Vec3f> gt_trajectory_;
  Point3fCloudOpenGL gt_trajectory_cloud_;
  
  // Estimated trajectory.
  vector<Vec3f> estimated_trajectory_;
  Point3fCloudOpenGL estimated_trajectory_cloud_;
  
  // Splat program.
  ShaderProgramOpenGL splat_program_;
  GLint splat_u_model_view_projection_matrix_location_;
  GLint splat_u_point_size_x_location_;
  GLint splat_u_point_size_y_location_;
  
  // Constant color program.
  ShaderProgramOpenGL constant_color_program_;
  GLint constant_color_u_model_view_projection_matrix_location_;
  GLint constant_color_u_constant_color_location_;
  
  // Parameters passed to the render thread by InitializeForCUDAInterop().
  usize init_max_point_count_;
  const Camera* init_camera_;
  std::atomic<bool> init_done_;
  mutex init_mutex_;
  condition_variable init_condition_;
  
  // Screenshot handling.
  string screenshot_path_;
  mutex screenshot_mutex_;
  condition_variable screenshot_condition_;
  
  // Follow-camera handling.
  bool use_follow_camera_ = false;
  bool use_rotating_camera_ = false;
  bool follow_camera_initialized_ = false;
  chrono::steady_clock::time_point last_render_time_;
  unique_ptr<TimerHelper> timer_helper_;
  
  // Keyframe-based camera animation.
  // Helper to use splines from the used spline library with single-dimension values.
  struct FloatForSpline {
    FloatForSpline(float value)
        : value(value) {}
    
    float length() const {
      return std::fabs(value);
    }
    
    operator float() const {
      return value;
    }
    
    float value;
  };
  
  std::vector<float> spline_frame_indices;
  std::vector<FloatForSpline> offset_x_spline_points;
  std::vector<FloatForSpline> offset_y_spline_points;
  std::vector<FloatForSpline> offset_z_spline_points;
  std::vector<FloatForSpline> radius_spline_points;
  std::vector<FloatForSpline> theta_spline_points;
  std::vector<FloatForSpline> phi_spline_points;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_x_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_y_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> offset_z_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> radius_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> theta_spline;
  unique_ptr<UniformCRSpline<FloatForSpline>> phi_spline;
  
  int playback_frame_ = -1;  // if negative, playback is disabled
  bool render_playback_ = false;
  
  // Other.
  Tool tool_;
  BadSlamRenderWindowSignalHelper signal_helper_;
  
  std::mutex render_mutex_;
  OpenGLContext qt_gl_context_;
  
  bool embedded_in_gui_;
  DirectBA* dense_ba_;  // non-threadsafe access, for debugging only!
};

}
