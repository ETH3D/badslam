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
#include <thread>

#include <libvis/libvis.h>
#include <libvis/opengl_context.h>
#include <libvis/rgbd_video.h>
#include <QApplication>
#include <QMainWindow>
#include <QProgressBar>
#include <QRadioButton>
#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>

#include "badslam/bad_slam_config.h"

namespace vis {

void ShowMainWindow(
    QApplication& qapp,
    bool start_paused,
    BadSlamConfig& config,
    const string& program_path,
    const string& dataset_folder_path,
    float depth_scaling,
    float splat_half_extent_in_pixels,
    bool show_current_frame_cloud,
    bool show_input_images,
    int window_width,
    int window_height);


class BadSlam;
class BadSlamRenderWindow;
class ImageDisplayQtWindow;
class RenderWindowQtOpenGL;

class MainWindow : public QMainWindow {
 Q_OBJECT
 public:
  MainWindow(
      bool start_paused,
      BadSlamConfig& config,
      const string& program_path,
      const string& dataset_folder_path,
      float depth_scaling,
      float splat_half_extent_in_pixels,
      bool show_current_frame_cloud,
      bool show_input_images,
      QWidget* parent = nullptr,
      Qt::WindowFlags flags = Qt::WindowFlags());
  
  ~MainWindow();
  
 signals:
  void CouldNotLoadDatasetSignal(const QString& dataset_path);
  void RunStateChangedSignal(bool running);
  void IntrinsicsUpdatedSignal();
  void DatasetPlaybackFinishedSignal();
  
  void UpdateCurrentFrameSignal(int current_frame, int frame_count);
  void UpdateGPUMemoryUsageSignal();
  void UpdateSurfelUsageSignal(int surfel_count);
  void UpdateCurrentFrameImagesSignal(int frame_index, bool images_in_use_elsewhere);
  
 private slots:
  // File menu
  void SaveState();
  void LoadState();
  void SaveEstimatedTrajectory();
  void SaveSurfelCloud();
  void SaveCalibration();
  
  // View menu
  void CopyView();
  void PasteView();
  void FollowCameraChanged();
  void SurfelDisplayModeChanged();
  void ShowStateChanged();
  void ShowCurrentFrameCloud();
  void ShowCurrentFrameImages();
  void ShowIntrinsicsAndDepthDeformation();
  void EnlargeSurfelSplats();
  void ShrinkSurfelSplats();
  
  // Tools menu
  void MoveFrameManually();
  void MoveFrameManuallyButton(int right_left, int top_bottom, int forward_backward, int roll, const QString& index_text, const QString& amount_text, bool rotate);
  void ClearMotionModel();
  void SetFrameIndex();
  void MergeClosestSuccessiveKeyframes();
  
  // Help menu
  void ShowAboutDialog();
  
  // Toolbar
  void Settings();
  void StartOrPause();
  void SingleStep();
  void KFStep();
  void SkipFrame();
  void SingleStepBackwards();
  void ManualBundleAdjustment();
  void DensifySurfels();
  void Screenshot();
  void SelectKeyframeTool();
  void DeleteKeyframeTool();
  
  // 3D view actions
  void ClickedKeyframe(int index);
  
  // NOTE: The following functions must be called from the GUI thread.
  void CouldNotLoadDataset(const QString& dataset_path);
  void RunStateChanged(bool running);
  void IntrinsicsUpdated();
  void DatasetPlaybackFinished();
  void UpdateCurrentFrame(int current_frame, int frame_count);
  void UpdateGPUMemoryUsage();
  void UpdateSurfelUsage(int surfel_count);
  void UpdateCurrentFrameImages(int frame_index, bool images_in_use_elsewhere);
  void ShowCurrentFrameImagesOnceSLAMInitialized();
  void EnableRunButtons(bool enable);
  
 private:
  bool UsingLiveInput();
  
  void WorkerThreadMain();
  
  
  // SLAM
  atomic<bool> run_;  // whether the reconstruction should run
  atomic<bool> is_running_;  // whether the reconstruction is actually running right now
  atomic<bool> single_step_;  // whether to do only a single step
  atomic<bool> create_kf_;  // whether to create a keyframe for this frame
  atomic<bool> skip_frame_;  // whether to skip the next frame
  bool backwards_;
  atomic<usize> frame_index_;
  mutex run_mutex_;
  condition_variable run_condition_;
  std::atomic<bool> quit_requested_;
  std::atomic<bool> quit_done_;
  condition_variable quit_condition_;
  
  atomic<bool> bad_slam_set_;
  unique_ptr<BadSlam> bad_slam_;
  RGBDVideo<Vec3u8, u16> rgbd_video_;
  std::mutex rgbd_video_mutex_;
  
  OpenGLContext opengl_context;  // context for the main thread
  OpenGLContext opengl_context_2;  // context for the BA thread
  
  QTimer* slam_init_timer_ = nullptr;
  
  // GUI
  // File menu
  QAction* save_state_act;
  QAction* load_state_act;
  QAction* save_trajectory_act;
  QAction* save_surfel_cloud_act;
  QAction* save_calibration_act;
  
  QString last_save_dir_ = ".";
  
  // View menu
  QAction* copy_view_act;
  QAction* paste_view_act;
  
  QAction* follow_camera_act;
  
  QAction* show_current_frame_act;
  QAction* show_estimated_trajectory_act;
  QAction* show_keyframes_act;
  QAction* show_surfels_act;
  QAction* show_current_frame_cloud_act;
  QAction* show_current_frame_images_act;
  QDialog* current_frame_images_dialog = nullptr;
  ImageDisplayQtWindow* current_frame_combined_display = nullptr;
  ImageDisplayQtWindow* current_frame_color_display = nullptr;
  ImageDisplayQtWindow* current_frame_depth_display = nullptr;
  QAction* show_intrinsics_act;
  QDialog* intrinsics_dialog = nullptr;
  QLineEdit* color_intrinsics_edit = nullptr;
  QLineEdit* depth_intrinsics_edit = nullptr;
  QLineEdit* depth_alpha_edit = nullptr;
  QLabel* min_max_calibrated_depth_label = nullptr;
  ImageDisplayQtWindow* depth_deformation_display = nullptr;
  QAction* enlarge_surfel_splats_act;
  QAction* shrink_surfel_splats_act;
  QAction* surfel_display_colored_action;
  QAction* surfel_display_normals_action;
  QAction* surfel_display_descriptors_action;
  QAction* surfel_display_radii_action;
  
  // Tools menu
  QAction* move_frame_manually_act;
  QDialog* move_frame_manually_dialog_ = nullptr;
  QAction* clear_motion_model_act;
  QAction* set_frame_index_act;
  QAction* merge_closest_keyframes_act;
  
  // Help menu
  QAction* about_act;
  
  // Toolbar
  QAction* settings_act;
  QAction* start_or_pause_act;
  QAction* single_step_act;
  QAction* single_step_backwards_act;
  QAction* kf_step_act;
  QAction* skip_frame_act;
  QAction* ba_act;
  QAction* densify_act;
  QAction* select_keyframe_act;
  QAction* delete_keyframe_act;
  atomic<bool> show_current_frame_;
  
  
  shared_ptr<BadSlamRenderWindow> render_window_;
  shared_ptr<RenderWindowQtOpenGL> render_window_gl_;
  
  QLabel* status_label_;
  QProgressBar* surfel_count_bar_;
  QProgressBar* gpu_memory_bar_;
  
  unique_ptr<std::thread> worker_thread_;
  std::thread::id gui_thread_id_;
  
  // Settings
  string program_path_;
  string dataset_folder_path_;
  float depth_scaling_;
  BadSlamConfig& config_;
};

}
