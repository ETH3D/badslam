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

// Must be included before Qt includes to avoid "foreach" conflict
#include "badslam/input_realsense.h"
#include "badslam/input_structure.h"
#include "badslam/input_azurekinect.h"

#include "badslam/gui_main_window.h"

#include <boost/filesystem.hpp>
#include <libvis/opengl_context.h>
#include <libvis/render_window_qt_opengl.h>
#include <libvis/rgbd_video_io_tum_dataset.h>
#include <QApplication>
#include <QBoxLayout>
#include <QLabel>
#include <QStatusBar>
#include <QMenuBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QToolBar>
#include <QToolButton>
#include <QPushButton>
#include <QLineEdit>
#include <QListWidget>
#include <QProgressDialog>
#include <QPixmap>
#include <QComboBox>
#include <QGroupBox>
#include <QSettings>
#include <QTextEdit>
#include <QInputDialog>

#include "badslam/bad_slam.h"
#include "badslam/cuda_image_processing.h"
#include "badslam/gui_keyframe_dialog.h"
#include "badslam/gui_settings_window.h"
#include "badslam/io.h"
#include "badslam/licenses.h"
#include "badslam/pre_load_thread.h"
#include "badslam/render_window.h"
#include "badslam/util.cuh"
#include "badslam/util.h"

// This must be done outside of any namespace according to the Qt documentation.
void InitQtResources() {
  static std::mutex init_mutex;
  std::unique_lock<std::mutex> lock(init_mutex);
  static bool resources_initialized = false;
  if (!resources_initialized) {
    Q_INIT_RESOURCE(badslam_resources);
    resources_initialized = true;
  }
}

namespace vis {

void ShowMainWindow(
    QApplication& qapp,
    bool start_paused,
    BadSlamConfig& config,
    const string& program_path,
    const string& dataset_folder_path,
    const string& import_calibration_path,
    float depth_scaling,
    float splat_half_extent_in_pixels,
    bool show_current_frame_cloud,
    bool show_input_images,
    int window_width,
    int window_height) {
  InitQtResources();
  
  MainWindow main_window(
      start_paused,
      config,
      program_path,
      dataset_folder_path,
      import_calibration_path,
      depth_scaling,
      splat_half_extent_in_pixels,
      show_current_frame_cloud,
      show_input_images,
      nullptr,
      Qt::WindowFlags());
  main_window.setVisible(true);
  main_window.raise();
  main_window.resize(window_width, window_height);
  
  qapp.exec();
}

MainWindow::MainWindow(
    bool start_paused,
    BadSlamConfig& config,
    const string& program_path,
    const string& dataset_folder_path,
    const string& import_calibration_path,
    float depth_scaling,
    float splat_half_extent_in_pixels,
    bool show_current_frame_cloud,
    bool show_input_images,
    QWidget* parent,
    Qt::WindowFlags flags)
    : QMainWindow(parent, flags),
      program_path_(program_path),
      dataset_folder_path_(dataset_folder_path),
      import_calibration_path_(import_calibration_path),
      depth_scaling_(depth_scaling),
      config_(config) {
  setWindowTitle("BAD SLAM");
  setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QSettings settings;
  last_save_dir_ = settings.value("last_save_dir").toString();
  
  // File menu
  save_state_act = new QAction(tr("Save SLAM state ..."), this);
  save_state_act->setStatusTip(tr("Save the complete SLAM state as a binary file"));
  connect(save_state_act, &QAction::triggered, this, &MainWindow::SaveState);
  
  load_state_act = new QAction(tr("Load SLAM state"), this);
  load_state_act->setStatusTip(tr("Load the complete SLAM state from a binary file"));
  connect(load_state_act, &QAction::triggered, this, &MainWindow::LoadState);
  
  save_trajectory_act = new QAction(tr("Export estimated trajectory ..."), this);
  save_trajectory_act->setStatusTip(tr("Export the estimated trajectory in TUM RGB-D format"));
  connect(save_trajectory_act, &QAction::triggered, this, &MainWindow::SaveEstimatedTrajectory);
  
  save_surfel_cloud_act = new QAction(tr("Export surfel cloud ..."), this);
  save_surfel_cloud_act->setStatusTip(tr("Export the surfel cloud as a point cloud"));
  connect(save_surfel_cloud_act, &QAction::triggered, this, &MainWindow::SaveSurfelCloud);
  
  save_calibration_act = new QAction(tr("Export intrinsic calibration ..."), this);
  save_calibration_act->setStatusTip(tr("Export the camera intrinsic calibration"));
  connect(save_calibration_act, &QAction::triggered, this, &MainWindow::SaveCalibration);
  
  QAction* quit_act = new QAction(tr("Quit"), this);
  connect(quit_act, &QAction::triggered, this, &MainWindow::close);
  
  QMenu* file_menu = menuBar()->addMenu(tr("&File"));
  file_menu->addAction(save_state_act);
  file_menu->addAction(load_state_act);
  file_menu->addSeparator();
  file_menu->addAction(save_trajectory_act);
  file_menu->addAction(save_surfel_cloud_act);
  file_menu->addAction(save_calibration_act);
  file_menu->addSeparator();
  file_menu->addAction(quit_act);
  
  // View menu
  copy_view_act = new QAction(tr("Copy view pose (as text)"), this);
  copy_view_act->setStatusTip(tr("Copy the pose of the 3D view, which can be pasted and stored as plain text"));
  connect(copy_view_act, &QAction::triggered, this, &MainWindow::CopyView);
  
  paste_view_act = new QAction(tr("Paste view pose"), this);
  paste_view_act->setStatusTip(tr("Paste the pose of the 3D view"));
  connect(paste_view_act, &QAction::triggered, this, &MainWindow::PasteView);
  
  follow_camera_act = new QAction(tr("Follow camera"), this);
  follow_camera_act->setStatusTip(tr("Make the 3D view follow the current estimate of the camera position"));
  follow_camera_act->setCheckable(true);
  connect(follow_camera_act, &QAction::triggered, this, &MainWindow::FollowCameraChanged);
  
  show_current_frame_act = new QAction(tr("Render current frame"), this);
  show_current_frame_act->setStatusTip(tr("Render the camera frustum corresponding to the current frame of the video in the 3D view"));
  show_current_frame_act->setCheckable(true);
  show_current_frame_act->setChecked(true);
  show_current_frame_ = false;
  connect(show_current_frame_act, &QAction::triggered, this, &MainWindow::ShowStateChanged);
  
  show_estimated_trajectory_act = new QAction(tr("Render estimated trajectory"), this);
  show_estimated_trajectory_act->setStatusTip(tr("Render the estimated trajectory as a line in the 3D view"));
  show_estimated_trajectory_act->setCheckable(true);
  show_estimated_trajectory_act->setChecked(true);
  connect(show_estimated_trajectory_act, &QAction::triggered, this, &MainWindow::ShowStateChanged);
  
  show_keyframes_act = new QAction(tr("Render keyframes"), this);
  show_keyframes_act->setStatusTip(tr("Render the camera frusta corresponding to the keyframes in the 3D view"));
  show_keyframes_act->setCheckable(true);
  show_keyframes_act->setChecked(true);
  connect(show_keyframes_act, &QAction::triggered, this, &MainWindow::ShowStateChanged);
  
  show_surfels_act = new QAction(tr("Render surfels"), this);
  show_surfels_act->setStatusTip(tr("Render the reconstructed surfel cloud as splats in the 3D view"));
  show_surfels_act->setCheckable(true);
  show_surfels_act->setChecked(true);
  connect(show_surfels_act, &QAction::triggered, this, &MainWindow::ShowStateChanged);
  
  show_current_frame_cloud_act = new QAction(tr("Show the current frame's point cloud"), this);
  show_current_frame_cloud_act->setStatusTip(tr("Show the current frame's point cloud"));
  show_current_frame_cloud_act->setCheckable(true);
  show_current_frame_cloud_act->setChecked(show_current_frame_cloud);
  connect(show_current_frame_cloud_act, &QAction::triggered, this, &MainWindow::ShowCurrentFrameCloud);
  
  show_current_frame_images_act = new QAction(tr("Show the current frame's images"), this);
  show_current_frame_images_act->setStatusTip(tr("Show the current frame's images"));
  show_current_frame_images_act->setCheckable(true);
  show_current_frame_images_act->setChecked(show_input_images);
  connect(show_current_frame_images_act, &QAction::triggered, this, &MainWindow::ShowCurrentFrameImages);
  
  show_intrinsics_act = new QAction(tr("Show intrinsics and depth deformation"), this);
  show_intrinsics_act->setStatusTip(tr("Show intrinsics and depth deformation"));
  show_intrinsics_act->setCheckable(true);
  connect(show_intrinsics_act, &QAction::triggered, this, &MainWindow::ShowIntrinsicsAndDepthDeformation);
  
  enlarge_surfel_splats_act = new QAction(tr("Enlarge surfel splats"), this);
  enlarge_surfel_splats_act->setStatusTip(tr("Enlarge the surfel splat display (can also be changed with Ctrl + mouse wheel on 3D view)"));
  connect(enlarge_surfel_splats_act, &QAction::triggered, this, &MainWindow::EnlargeSurfelSplats);
  
  shrink_surfel_splats_act = new QAction(tr("Shrink surfel splats"), this);
  shrink_surfel_splats_act->setStatusTip(tr("Shrink the surfel splat display (can also be changed with Ctrl + mouse wheel on 3D view)"));
  connect(shrink_surfel_splats_act, &QAction::triggered, this, &MainWindow::ShrinkSurfelSplats);
  
  QActionGroup* surfel_display_action_group = new QActionGroup(this);
  
  surfel_display_colored_action = new QAction(tr("Colored"), this);
  surfel_display_colored_action->setCheckable(true);
  surfel_display_colored_action->setChecked(true);
  connect(surfel_display_colored_action, &QAction::triggered, this, &MainWindow::SurfelDisplayModeChanged);
  
  surfel_display_normals_action = new QAction(tr("Normals"), this);
  surfel_display_normals_action->setCheckable(true);
  connect(surfel_display_normals_action, &QAction::triggered, this, &MainWindow::SurfelDisplayModeChanged);
  
  surfel_display_descriptors_action = new QAction(tr("Descriptors"), this);
  surfel_display_descriptors_action->setCheckable(true);
  connect(surfel_display_descriptors_action, &QAction::triggered, this, &MainWindow::SurfelDisplayModeChanged);
  
  surfel_display_radii_action = new QAction(tr("Radii"), this);
  surfel_display_radii_action->setCheckable(true);
  connect(surfel_display_radii_action, &QAction::triggered, this, &MainWindow::SurfelDisplayModeChanged);
  
  QMenu* surfel_display_menu = new QMenu(tr("Surfel display mode"));
  surfel_display_menu->addAction(surfel_display_colored_action);
  surfel_display_action_group->addAction(surfel_display_colored_action);
  surfel_display_menu->addAction(surfel_display_normals_action);
  surfel_display_action_group->addAction(surfel_display_normals_action);
  surfel_display_menu->addAction(surfel_display_descriptors_action);
  surfel_display_action_group->addAction(surfel_display_descriptors_action);
  surfel_display_menu->addAction(surfel_display_radii_action);
  surfel_display_action_group->addAction(surfel_display_radii_action);
  
  QMenu* view_menu = menuBar()->addMenu(tr("&View"));
  view_menu->addAction(copy_view_act);
  view_menu->addAction(paste_view_act);
  view_menu->addAction(follow_camera_act);
  view_menu->addSeparator();
  view_menu->addAction(show_current_frame_act);
  view_menu->addAction(show_estimated_trajectory_act);
  view_menu->addAction(show_keyframes_act);
  view_menu->addAction(show_surfels_act);
  view_menu->addAction(show_current_frame_cloud_act);
  view_menu->addAction(show_current_frame_images_act);
  view_menu->addAction(show_intrinsics_act);
  view_menu->addSeparator();
  view_menu->addAction(enlarge_surfel_splats_act);
  view_menu->addAction(shrink_surfel_splats_act);
  view_menu->addMenu(surfel_display_menu);
  
  // Tools menu
  move_frame_manually_act = new QAction(tr("Move frame manually"), this);
  move_frame_manually_act->setStatusTip(tr("Move a frame manually"));
  move_frame_manually_act->setCheckable(true);
  connect(move_frame_manually_act, &QAction::triggered, this, &MainWindow::MoveFrameManually);
  
  clear_motion_model_act = new QAction(tr("Clear motion model"), this);
  clear_motion_model_act->setStatusTip(tr("Clear motion model"));
  connect(clear_motion_model_act, &QAction::triggered, this, &MainWindow::ClearMotionModel);
  
  set_frame_index_act = new QAction(tr("Set current frame index"), this);
  set_frame_index_act->setStatusTip(tr("Set the current frame index"));
  connect(set_frame_index_act, &QAction::triggered, this, &MainWindow::SetFrameIndex);
  
  merge_closest_keyframes_act = new QAction(tr("Delete closest successive keyframes"), this);
  merge_closest_keyframes_act->setStatusTip(tr("Delete the closest successive keyframes (which are likely most redundant)"));
  connect(merge_closest_keyframes_act, &QAction::triggered, this, &MainWindow::MergeClosestSuccessiveKeyframes);
  
  QMenu* tools_menu = menuBar()->addMenu(tr("&Tools"));
  tools_menu->addAction(move_frame_manually_act);
  tools_menu->addAction(clear_motion_model_act);
  tools_menu->addAction(set_frame_index_act);
  tools_menu->addAction(merge_closest_keyframes_act);
  
  // Help menu
  about_act = new QAction(tr("About"), this);
  about_act->setStatusTip(tr("See information about this program"));
  connect(about_act, &QAction::triggered, this, &MainWindow::ShowAboutDialog);
  
  QMenu* help_menu = menuBar()->addMenu(tr("&Help"));
  help_menu->addAction(about_act);
  
  // Toolbar
  QToolBar* toolbar = new QToolBar(tr("Main toolbar"), this);
  
  start_or_pause_act = toolbar->addAction(QIcon(":/badslam/run.png"), tr("Run / Pause"), this, SLOT(StartOrPause()));
  start_or_pause_act->setStatusTip(tr("Run / Pause"));
  
  single_step_act = toolbar->addAction(QIcon(":/badslam/single_step.png"), tr("Step forward by a single frame"), this, SLOT(SingleStep()));
  single_step_act->setStatusTip(tr("Step forward by a single frame"));
  
  kf_step_act = toolbar->addAction(QIcon(":/badslam/kf_step.png"), tr("Create keyframe and step forward"), this, SLOT(KFStep()));
  kf_step_act->setStatusTip(tr("Make the current frame a keyframe, and step forward by a single frame"));
  
  skip_frame_act = toolbar->addAction(QIcon(":/badslam/skip_frame.png"), tr("Skip a frame"), this, SLOT(SkipFrame()));
  skip_frame_act->setStatusTip(tr("Skip over the next frame"));
  
  single_step_backwards_act = toolbar->addAction(QIcon(":/badslam/single_step_backwards.png"), tr("Step backwards by a single frame"), this, SLOT(SingleStepBackwards()));
  single_step_backwards_act->setStatusTip(tr("Step backwards by a single frame"));
  
  ba_act = toolbar->addAction(QIcon(":/badslam/ba.png"), tr("Run bundle adjustment manually"), this, SLOT(ManualBundleAdjustment()));
  ba_act->setStatusTip(tr("Run bundle adjustment manually"));
  
  densify_act = toolbar->addAction(QIcon(":/badslam/densify.png"), tr("Densify surfels"), this, SLOT(DensifySurfels()));
  densify_act->setStatusTip(tr("Densify surfels"));
  
  toolbar->addSeparator();
  
  settings_act = toolbar->addAction(QIcon(":/badslam/settings.png"), tr("Settings"), this, SLOT(Settings()));
  settings_act->setStatusTip(tr("Settings"));
  
  QAction* screenshot_act = toolbar->addAction(QIcon(":/badslam/screenshot.png"), tr("Take a screenshot"), this, SLOT(Screenshot()));
  screenshot_act->setStatusTip(tr("Take a screenshot"));
  
  toolbar->addSeparator();
  
  select_keyframe_act = toolbar->addAction(QIcon(":/badslam/select_keyframe.png"), tr("Select keyframes"), this, SLOT(SelectKeyframeTool()));
  select_keyframe_act->setCheckable(true);
  select_keyframe_act->setChecked(true);
  select_keyframe_act->setStatusTip(tr("Select keyframes"));
  
  delete_keyframe_act = toolbar->addAction(QIcon(":/badslam/delete_keyframe.png"), tr("Delete keyframes"), this, SLOT(DeleteKeyframeTool()));
  delete_keyframe_act->setCheckable(true);
  delete_keyframe_act->setStatusTip(tr("Delete keyframes"));
  
  addToolBar(Qt::TopToolBarArea, toolbar);
  
  // Window layout
  QHBoxLayout* horizontal_layout = new QHBoxLayout();
  horizontal_layout->setContentsMargins(0, 0, 0, 0);
  
  render_window_.reset(
      new BadSlamRenderWindow(splat_half_extent_in_pixels,
                              /*embedded_in_gui*/ true));
  render_window_gl_.reset(
      new RenderWindowQtOpenGL(
          "BAD SLAM",
          /*width*/ -1,
          /*height*/ -1,
          render_window_,
          /*use_qt_thread*/ false,
          /*show*/ false));
  render_window_gl_->window()->setWindowFlags(Qt::Widget);
  horizontal_layout->addWidget(render_window_gl_->window());
  
  connect(&render_window_->signal_helper(), &BadSlamRenderWindowSignalHelper::ClickedKeyframe, this, &MainWindow::ClickedKeyframe);
  connect(&render_window_->signal_helper(), &BadSlamRenderWindowSignalHelper::FollowCameraEnabled, follow_camera_act, &QAction::setChecked);
  
  QWidget* main_widget = new QWidget();
  main_widget->setLayout(horizontal_layout);
  main_widget->setAutoFillBackground(false);
  setCentralWidget(main_widget);
  
  // Status bar
  status_label_ = new QLabel();
  statusBar()->addWidget(status_label_);
  
  surfel_count_bar_ = new QProgressBar();
  surfel_count_bar_->setRange(0, 0);
  surfel_count_bar_->setValue(0);
  surfel_count_bar_->setFormat(tr("Surfels: %vM / %mM"));
  surfel_count_bar_->setTextVisible(true);
  surfel_count_bar_->setFixedWidth(250);
  statusBar()->addPermanentWidget(surfel_count_bar_);
  
  gpu_memory_bar_ = new QProgressBar();
  gpu_memory_bar_->setRange(0, 0);
  gpu_memory_bar_->setValue(0);
  gpu_memory_bar_->setFormat(tr("GPU Mem: %v MiB / %m MiB"));
  gpu_memory_bar_->setTextVisible(true);
  gpu_memory_bar_->setFixedWidth(250);
  statusBar()->addPermanentWidget(gpu_memory_bar_);
  
  statusBar()->setSizeGripEnabled(true);
  
  // Create queued signal-slot connections that allow to run functions in a different thread.
  // NOTE: Some of these connections are blocking, others are not.
  connect(this, &MainWindow::FatalErrorSignal, this, &MainWindow::FatalError, Qt::BlockingQueuedConnection);
  connect(this, &MainWindow::RunStateChangedSignal, this, &MainWindow::RunStateChanged, Qt::BlockingQueuedConnection);
  connect(this, &MainWindow::IntrinsicsUpdatedSignal, this, &MainWindow::IntrinsicsUpdated, Qt::BlockingQueuedConnection);
  connect(this, &MainWindow::DatasetPlaybackFinishedSignal, this, &MainWindow::DatasetPlaybackFinished, Qt::BlockingQueuedConnection);
  connect(this, &MainWindow::UpdateCurrentFrameSignal, this, &MainWindow::UpdateCurrentFrame, Qt::QueuedConnection);
  connect(this, &MainWindow::UpdateGPUMemoryUsageSignal, this, &MainWindow::UpdateGPUMemoryUsage, Qt::QueuedConnection);
  connect(this, &MainWindow::UpdateSurfelUsageSignal, this, &MainWindow::UpdateSurfelUsage, Qt::QueuedConnection);
  
  run_ = !start_paused;
  single_step_ = false;
  create_kf_ = false;
  skip_frame_ = false;
  backwards_ = false;
  is_running_ = false;
  gui_thread_id_ = std::this_thread::get_id();
  
  // Start the worker thread which will run SLAM and other long-running tasks
  bad_slam_set_ = false;
  quit_requested_ = false;
  quit_done_ = false;
  worker_thread_.reset(new std::thread(std::bind(&MainWindow::WorkerThreadMain, this)));
  
  if (show_input_images) {
    // Wait until the SLAM thread finished initializing with the help of a timer
    slam_init_timer_ = new QTimer(this);
    connect(slam_init_timer_, SIGNAL(timeout()), this, SLOT(ShowCurrentFrameImagesOnceSLAMInitialized()));
    slam_init_timer_->start(10);
  }
}

MainWindow::~MainWindow() {
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  // Signal to the worker thread that it should exit
  unique_lock<mutex> lock(run_mutex_);
  run_ = false;
  quit_requested_ = true;
  lock.unlock();
  run_condition_.notify_all();
  
  // Wait for the thread to exit
  unique_lock<mutex> quit_lock(run_mutex_);
  while (!quit_done_) {
    // Since the thread might still emit queued events, we need to process them
    // to avoid possible deadlocks.
    quit_lock.unlock();
    qApp->processEvents();
    quit_lock.lock();
  }
  quit_lock.unlock();
  
  worker_thread_->join();
  worker_thread_.reset();
  
  // Deinitialize BAD SLAM before discarding the OpenGL context that it may still use
  if (bad_slam_set_) {
    bad_slam_.reset();
    
    opengl_context.Deinitialize();
    opengl_context_2.Deinitialize();
  }
}

void MainWindow::SaveState() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  QString path = QFileDialog::getSaveFileName(this, tr("Save state"), last_save_dir_, tr("BAD SLAM state files (*.badslam)"));
  if (path.isEmpty()) {
    return;
  }
  last_save_dir_ = QFileInfo(path).absoluteDir().absolutePath();
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  if (vis::SaveState(*bad_slam_, path.toStdString())) {
    statusBar()->showMessage(tr("Saved state to: %1").arg(path), 3000);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Could not save state to: %1").arg(path));
  }
}

void MainWindow::LoadState() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  QString path = QFileDialog::getOpenFileName(this, tr("Load state"), last_save_dir_, tr("BAD SLAM state files (*.badslam)"));
  if (path.isEmpty()) {
    return;
  }
  last_save_dir_ = QFileInfo(path).absoluteDir().absolutePath();
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  // Show a progress bar while loading the state.
  QProgressDialog progress(tr("Loading state ..."), tr("Cancel"), 0, 0, this);
  progress.setWindowModality(Qt::WindowModal);
  progress.setMinimumDuration(0);
  auto progress_function = [&](int step, int count) {
    progress.setMaximum(count);
    progress.setValue(step);
    UpdateGPUMemoryUsage();
    if (progress.wasCanceled()) {
      LOG(INFO) << "Loading canceled.";
      return false;
    } else {
      return true;
    }
  };
  
  if (vis::LoadState(bad_slam_.get(), path.toStdString(), progress_function)) {
    frame_index_ = bad_slam_->last_frame_index();
    
    bad_slam_->UpdateOdometryVisualization(
        std::max(0, static_cast<int>(frame_index_) - 1),
        show_current_frame_);
    bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
    UpdateCurrentFrame(frame_index_ + 1, bad_slam_->rgbd_video()->frame_count());
    UpdateSurfelUsage(bad_slam_->direct_ba().surfel_count());
    render_window_->RenderFrame();
    
    statusBar()->showMessage(tr("Loaded state from: %1").arg(path), 3000);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Could not load state from: %1").arg(path));
  }
}

void MainWindow::SaveEstimatedTrajectory() {
  if (!bad_slam_set_) {
    return;
  }
  
  QString path = QFileDialog::getSaveFileName(this, tr("Save estimated trajectory"), last_save_dir_, tr("TUM RGB-D trajectory files (*.txt)"));
  if (path.isEmpty()) {
    return;
  }
  last_save_dir_ = QFileInfo(path).absoluteDir().absolutePath();
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  if (SavePoses(rgbd_video_, config_.use_geometric_residuals,
                config_.start_frame, path.toStdString())) {
    statusBar()->showMessage(tr("Saved trajectory to: %1").arg(path), 3000);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Could not save trajectory to: %1").arg(path));
  }
}

void MainWindow::SaveSurfelCloud() {
  if (!bad_slam_set_) {
    return;
  }
  
  QString path = QFileDialog::getSaveFileName(this, tr("Save surfels as point cloud"), last_save_dir_, tr("PLY files (*.ply)"));
  if (path.isEmpty()) {
    return;
  }
  last_save_dir_ = QFileInfo(path).absoluteDir().absolutePath();
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  if (SavePointCloudAsPLY(/*stream*/ 0, bad_slam_->direct_ba(), path.toStdString())) {
    statusBar()->showMessage(tr("Saved surfel point cloud to: %1").arg(path), 3000);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Could not save surfel point cloud to: %1").arg(path));
  }
}

void MainWindow::SaveCalibration() {
  if (!bad_slam_set_) {
    return;
  }
  
  QString path = QFileDialog::getSaveFileName(this, tr("Save calibration"), last_save_dir_, tr("Calibration file base paths (*)"));
  if (path.isEmpty()) {
    return;
  }
  last_save_dir_ = QFileInfo(path).absoluteDir().absolutePath();
  QSettings settings;
  settings.setValue("last_save_dir", last_save_dir_);
  
  if (vis::SaveCalibration(/*stream*/ 0, bad_slam_->direct_ba(), path.toStdString())) {
    statusBar()->showMessage(tr("Saved calibration to: %1").arg(path), 3000);
  } else {
    QMessageBox::warning(this, tr("Error"), tr("Could not save calibration to: %1").arg(path));
  }
}

void MainWindow::CopyView() {
  render_window_->CopyView();
}

void MainWindow::PasteView() {
  render_window_->PasteView();
}

void MainWindow::FollowCameraChanged() {
  render_window_->UseFollowCamera(follow_camera_act->isChecked());
}

void MainWindow::SurfelDisplayModeChanged() {
  if (bad_slam_set_) {
    bad_slam_->direct_ba().SetVisualization(
        surfel_display_normals_action->isChecked(),
        surfel_display_descriptors_action->isChecked(),
        surfel_display_radii_action->isChecked());
    
    if (!is_running_) {
      bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
      render_window_->RenderFrame();
    }
  }
}

void MainWindow::ShowStateChanged() {
  render_window_->SetRenderCurrentFrameFrustum(show_current_frame_act->isChecked());
  render_window_->SetRenderEstimatedTrajectory(show_estimated_trajectory_act->isChecked());
  render_window_->SetRenderKeyframes(show_keyframes_act->isChecked());
  render_window_->SetRenderSurfels(show_surfels_act->isChecked());
  
  if (!show_keyframes_act->isChecked()) {
    select_keyframe_act->setChecked(false);
    delete_keyframe_act->setChecked(false);
    render_window_->SetTool(BadSlamRenderWindow::Tool::kNoTool);
  }
  
  render_window_->RenderFrame();
}

void MainWindow::ShowCurrentFrameCloud() {
  show_current_frame_ = show_current_frame_cloud_act->isChecked();
  if (bad_slam_set_ && !is_running_ && frame_index_ > config_.start_frame) {
    bad_slam_->UpdateOdometryVisualization(
        std::max(0, static_cast<int>(frame_index_) - 1),
        show_current_frame_);
  }
  render_window_->RenderFrame();
}

void MainWindow::ShowCurrentFrameImages() {
  if (!bad_slam_set_) {
    return;
  }
  
  if (current_frame_images_dialog) {
    current_frame_images_dialog->close();
    current_frame_images_dialog = nullptr;
    show_current_frame_images_act->setChecked(false);
    return;
  }
  
  current_frame_images_dialog = new QDialog(this);
  current_frame_images_dialog->setWindowTitle(tr("Current frame"));
  current_frame_images_dialog->setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  current_frame_combined_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ current_frame_images_dialog);
  current_frame_combined_display->SetDisplayAsWidget();
  current_frame_combined_display->FitContent();
  
  current_frame_color_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ current_frame_images_dialog);
  current_frame_color_display->SetDisplayAsWidget();
  current_frame_color_display->FitContent();
  
  current_frame_depth_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ current_frame_images_dialog);
  current_frame_depth_display->SetDisplayAsWidget();
  current_frame_depth_display->FitContent();
  current_frame_depth_display->SetBlackWhiteValues(0, config_.max_depth / config_.raw_to_float_depth);
  
  // Tab widget with the images
  QTabWidget* tab_widget = new QTabWidget(current_frame_images_dialog);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget->addTab(current_frame_combined_display, tr("Combined"));
  tab_widget->addTab(current_frame_color_display, tr("Color"));
  tab_widget->addTab(current_frame_depth_display, tr("Depth"));
  
  QVBoxLayout* layout = new QVBoxLayout();
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(tab_widget);
  current_frame_images_dialog->setLayout(layout);
  
  connect(current_frame_images_dialog, &QDialog::rejected, [&](){
    current_frame_images_dialog = nullptr;
    show_current_frame_images_act->setChecked(false);
    disconnect(this, &MainWindow::UpdateCurrentFrameImagesSignal, this, &MainWindow::UpdateCurrentFrameImages);
  });
  connect(this, &MainWindow::UpdateCurrentFrameImagesSignal, this, &MainWindow::UpdateCurrentFrameImages, Qt::BlockingQueuedConnection);
  
  show_current_frame_images_act->setChecked(true);
  
  // This will show the dialog once images are available
  UpdateCurrentFrameImages(std::max(0, static_cast<int>(frame_index_) - 1), false);
}

void MainWindow::ShowIntrinsicsAndDepthDeformation() {
  if (!bad_slam_set_) {
    return;
  }
  
  if (intrinsics_dialog) {
    intrinsics_dialog->close();
    intrinsics_dialog = nullptr;
    show_intrinsics_act->setChecked(false);
    return;
  }
  
  intrinsics_dialog = new QDialog(this);
  intrinsics_dialog->setWindowTitle(tr("Intrinsics and depth deformation"));
  intrinsics_dialog->setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QString help_text = tr("Values not available (open this dialog while SLAM is paused to get them)");
  bool running = is_running_;  // might change while this function is executed, but that should not matter much
  
  QVBoxLayout* layout = new QVBoxLayout();
  
  QGridLayout* intrinsics_layout = new QGridLayout();
  
  intrinsics_layout->addWidget(new QLabel(tr("Color camera intrinsics (fx fy cx cy): ")), 0, 0);
  color_intrinsics_edit = new QLineEdit(help_text);
  color_intrinsics_edit->setEnabled(!running);
  connect(color_intrinsics_edit, &QLineEdit::textEdited, [&](const QString& text) {
    float parameters[4];
    if (sscanf(text.toStdString().c_str(), "%f %f %f %f", &parameters[0], &parameters[1], &parameters[2], &parameters[3]) != 4) {
      return;
    }
    PinholeCamera4f color_camera = bad_slam_->direct_ba().color_camera();
    bad_slam_->direct_ba().SetColorCamera(PinholeCamera4f(color_camera.width(), color_camera.height(), parameters));
  });
  intrinsics_layout->addWidget(color_intrinsics_edit, 0, 1);
  
  intrinsics_layout->addWidget(new QLabel(tr("Depth camera intrinsics (fx fy cx cy): ")), 1, 0);
  depth_intrinsics_edit = new QLineEdit(help_text);
  depth_intrinsics_edit->setEnabled(!running);
  connect(depth_intrinsics_edit, &QLineEdit::textEdited, [&](const QString& text) {
    float parameters[4];
    if (sscanf(text.toStdString().c_str(), "%f %f %f %f", &parameters[0], &parameters[1], &parameters[2], &parameters[3]) != 4) {
      return;
    }
    PinholeCamera4f depth_camera = bad_slam_->direct_ba().depth_camera();
    bad_slam_->direct_ba().SetDepthCamera(PinholeCamera4f(depth_camera.width(), depth_camera.height(), parameters));
  });
  intrinsics_layout->addWidget(depth_intrinsics_edit, 1, 1);
  
  intrinsics_layout->addWidget(new QLabel(tr("Depth deformation parameter alpha: ")), 2, 0);
  depth_alpha_edit = new QLineEdit(help_text);
  depth_alpha_edit->setEnabled(!running);
  connect(depth_alpha_edit, &QLineEdit::textEdited, [&](const QString& text) {
    float a;
    if (sscanf(text.toStdString().c_str(), "%f", &a) != 1) {
      return;
    }
    DepthParameters params = bad_slam_->direct_ba().depth_params();
    params.a = a;
    bad_slam_->direct_ba().SetDepthParams(params);
  });
  intrinsics_layout->addWidget(depth_alpha_edit, 2, 1);
  
  layout->addLayout(intrinsics_layout);
  
  min_max_calibrated_depth_label = new QLabel();
  layout->addWidget(min_max_calibrated_depth_label);
  
  depth_deformation_display = new ImageDisplayQtWindow(/*display*/ nullptr, /*parent*/ intrinsics_dialog);
  depth_deformation_display->SetDisplayAsWidget();
  depth_deformation_display->FitContent();
  layout->addWidget(depth_deformation_display);
  
  intrinsics_dialog->setLayout(layout);
  intrinsics_dialog->resize(2 * intrinsics_dialog->sizeHint());
  intrinsics_dialog->show();
  show_intrinsics_act->setChecked(true);
  
  connect(this, &MainWindow::RunStateChangedSignal, intrinsics_dialog, [&](bool running) {
    color_intrinsics_edit->setEnabled(!running);
    depth_intrinsics_edit->setEnabled(!running);
    depth_alpha_edit->setEnabled(!running);
  });
  
  connect(intrinsics_dialog, &QDialog::rejected, [&](){
    intrinsics_dialog = nullptr;
    show_intrinsics_act->setChecked(false);
  });
  
  if (!is_running_) {
    IntrinsicsUpdated();
  }
}

void MainWindow::EnlargeSurfelSplats() {
  render_window_->ChangeSplatSize(2);
  render_window_->RenderFrame();
}

void MainWindow::ShrinkSurfelSplats() {
  render_window_->ChangeSplatSize(-2);
  render_window_->RenderFrame();
}

void MainWindow::MoveFrameManually() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  if (move_frame_manually_dialog_) {
    delete move_frame_manually_dialog_;
    move_frame_manually_dialog_ = nullptr;
    move_frame_manually_act->setChecked(false);
    return;
  }
  
  move_frame_manually_dialog_ = new QDialog(this);
  move_frame_manually_dialog_->setWindowTitle(tr("Move frame manually"));
  move_frame_manually_dialog_->setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QGridLayout* top_layout = new QGridLayout();
  
  QLabel* frame_index_label = new QLabel(tr("Frame index: "));
  top_layout->addWidget(frame_index_label, 0, 0);
  QLineEdit* frame_index_edit = new QLineEdit(QString::number(std::max(0, static_cast<int>(frame_index_) - 1)));
  top_layout->addWidget(frame_index_edit, 0, 1);
  
  QLabel* amount_label = new QLabel(tr("Amount: "));
  top_layout->addWidget(amount_label, 1, 0);
  QLineEdit* amount_edit = new QLineEdit("0.1");
  top_layout->addWidget(amount_edit, 1, 1);
  
  QHBoxLayout* mid_layout = new QHBoxLayout();
  
  QRadioButton* translate_radio = new QRadioButton(tr("Translate"));
  translate_radio->setChecked(true);
  mid_layout->addWidget(translate_radio);
  QRadioButton* rotate_radio = new QRadioButton(tr("Rotate"));
  mid_layout->addWidget(rotate_radio);
  
  QGridLayout* bottom_layout = new QGridLayout();
  
  QPushButton* forward_button = new QPushButton("/\\");
  QPushButton* left_button = new QPushButton("<-");
  QPushButton* up_button = new QPushButton(tr("up"));
  QPushButton* down_button = new QPushButton(tr("down"));
  QPushButton* right_button = new QPushButton("->");
  QPushButton* backward_button = new QPushButton("\\/");
  
  bottom_layout->addWidget(forward_button, 0, 1);
  bottom_layout->addWidget(left_button, 1, 0, 2, 1);
  bottom_layout->addWidget(up_button, 1, 1);
  bottom_layout->addWidget(down_button, 2, 1);
  bottom_layout->addWidget(right_button, 1, 2, 2, 1);
  bottom_layout->addWidget(backward_button, 3, 1);
  
  bottom_layout->setRowStretch(0, 2);
  bottom_layout->setRowStretch(1, 1);
  bottom_layout->setRowStretch(2, 1);
  bottom_layout->setRowStretch(3, 2);
  
  QHBoxLayout* roll_layout = new QHBoxLayout();
  QPushButton* roll_left_button = new QPushButton("roll left");
  QPushButton* roll_right_button = new QPushButton("roll right");
  roll_layout->addWidget(roll_left_button);
  roll_layout->addWidget(roll_right_button);
  
  QVBoxLayout* layout = new QVBoxLayout(move_frame_manually_dialog_);
  layout->addLayout(top_layout);
  layout->addLayout(mid_layout);
  layout->addLayout(bottom_layout);
  layout->addLayout(roll_layout);
  move_frame_manually_dialog_->setLayout(layout);
  move_frame_manually_dialog_->show();
  move_frame_manually_act->setChecked(true);
  
  connect(move_frame_manually_dialog_, &QDialog::rejected, [&](){
    move_frame_manually_dialog_ = nullptr;
    move_frame_manually_act->setChecked(false);
  });
  
  connect(forward_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, 0, -1, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(left_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(1, 0, 0, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(up_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, 1, 0, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(down_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, -1, 0, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(right_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(-1, 0, 0, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(backward_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, 0, 1, 0, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(roll_left_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, 0, 0, -1, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
  connect(roll_right_button, &QPushButton::clicked, [=](){
    MoveFrameManuallyButton(0, 0, 0, 1, frame_index_edit->text(), amount_edit->text(), rotate_radio->isChecked());
  });
}

void MainWindow::MoveFrameManuallyButton(int right_left, int top_bottom, int forward_backward, int roll, const QString& index_text, const QString& amount_text, bool rotate) {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  bool ok = true;
  int index = index_text.toInt(&ok);
  if (!ok) {
    QMessageBox::warning(move_frame_manually_dialog_, tr("Error"), tr("Cannot parse the frame index."));
    return;
  }
  if (index < 0 || index >= bad_slam_->rgbd_video()->frame_count()) {
    QMessageBox::warning(move_frame_manually_dialog_, tr("Error"), tr("Frame index is out of range."));
    return;
  }
  
  float amount = amount_text.toFloat(&ok);
  if (!ok) {
    QMessageBox::warning(move_frame_manually_dialog_, tr("Error"), tr("Cannot parse the move amount."));
    return;
  }
  
  if (index >= frame_index_) {
    QMessageBox::information(move_frame_manually_dialog_, tr("Warning"), tr("Frame index is beyond the current frame."));
  }
  
  auto& depth_frame = bad_slam_->rgbd_video()->depth_frame_mutable(index);
  auto& color_frame = bad_slam_->rgbd_video()->color_frame_mutable(index);
  
  SE3f global_tr_frame = depth_frame->global_T_frame();
  
  if (rotate) {
    if (right_left != 0) {
      global_tr_frame *= SE3f(Quaternionf(AngleAxisf(-amount * right_left, Vec3f(0, 1, 0))), Vec3f::Zero());
    }
    if (top_bottom != 0 || forward_backward != 0) {
      global_tr_frame *= SE3f(Quaternionf(AngleAxisf(amount * (top_bottom - forward_backward), Vec3f(1, 0, 0))), Vec3f::Zero());
    }
    if (roll != 0) {
      global_tr_frame *= SE3f(Quaternionf(AngleAxisf(amount * roll, Vec3f(0, 0, 1))), Vec3f::Zero());
    }
  } else {
    global_tr_frame *= SE3f(Mat3f::Identity(), -amount * Vec3f(right_left, top_bottom, forward_backward));
  }
  
  // If this frame is a keyframe, it must be moved with the keyframe's
  // set_global_T_frame (or set_frame_T_global) function to update the
  // keyframe's cached CUDA poses.
  bool found_keyframe = false;
  auto& keyframes = bad_slam_->direct_ba().keyframes();
  for (auto& keyframe : keyframes) {
    if (keyframe && keyframe->frame_index() == index) {
      keyframe->set_global_T_frame(global_tr_frame);
      bad_slam_->direct_ba().UpdateKeyframeCoVisibility(keyframe);
      
      found_keyframe = true;
      break;
    }
  }
  if (!found_keyframe) {
    depth_frame->SetGlobalTFrame(global_tr_frame);
    color_frame->SetGlobalTFrame(global_tr_frame);
  }
  
  // If this frame is the current frame, reset the motion model
  if (frame_index_ == index ||
      frame_index_ == index + 1) {
    bad_slam_->ClearMotionModel(static_cast<int>(frame_index_) - 1);
  }
  
  bad_slam_->UpdateOdometryVisualization(
      std::max(0, static_cast<int>(frame_index_) - 1),
      show_current_frame_);
  if (found_keyframe) {
    bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
  }
  render_window_->RenderFrame();
  
  // TODO: If a keyframe is moved, (optionally?) adapt the poses of the non-keyframes around it to follow
}

void MainWindow::ClearMotionModel() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  bad_slam_->ClearMotionModel(static_cast<int>(frame_index_) - 1);
}

void MainWindow::SetFrameIndex() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  bool ok = true;
  int new_frame_index = QInputDialog::getInt(
      this,
      tr("Set current frame index"),
      tr("Current frame index:"),
      frame_index_,
      0,
      bad_slam_->rgbd_video()->frame_count() - 1,
      1,
      &ok);
  
  if (ok) {
    frame_index_ = new_frame_index;
    bad_slam_->UpdateOdometryVisualization(
        std::max(0, static_cast<int>(frame_index_) - 1),
        show_current_frame_);
    UpdateCurrentFrame(frame_index_ + 1, bad_slam_->rgbd_video()->frame_count());
    render_window_->RenderFrame();
    
    bad_slam_->ClearMotionModel(static_cast<int>(frame_index_) - 1);
  }
}

void MainWindow::MergeClosestSuccessiveKeyframes() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  bool ok = true;
  int approx_merge_count = QInputDialog::getInt(
      this,
      tr("Delete closest successive keyframes"),
      tr("Approximate number of keyframes to delete:"),
      10,
      0,
      bad_slam_->direct_ba().keyframes().size() - 1,
      1,
      &ok);
  
  if (ok) {
    bad_slam_->direct_ba().MergeKeyframes(0, bad_slam_->loop_detector(), approx_merge_count);
    bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
    render_window_->RenderFrame();
    UpdateGPUMemoryUsage();
  }
}

void MainWindow::ShowAboutDialog() {
  QDialog about_dialog(this);
  about_dialog.setWindowTitle(tr("About BAD SLAM"));
  about_dialog.setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QHBoxLayout* top_layout = new QHBoxLayout();
  
  QLabel* icon_label = new QLabel();
  icon_label->setPixmap(QPixmap(":/badslam/badslam.png"));
  top_layout->addWidget(icon_label);
  
  top_layout->addSpacing(16);
  
  QLabel* top_label = new QLabel(tr("<b>BAD SLAM 1.0</b> - Bundle Adjusted Direct RGB-D SLAM<br/>This software was developed by Thomas Schöps (ETH Zurich).<br/>The Windows port and Kinect-for-Azure (K4A) integration has been contributed by Silvano Galliani (Microsoft AI & Vision Zurich)."));
  top_layout->addWidget(top_label);
  
  QGroupBox* licenses_group = new QGroupBox("");
  QVBoxLayout* licenses_layout = new QVBoxLayout();
  
  QLabel* licenses_label = new QLabel(tr("The licenses of BAD SLAM and its third-party components are listed here."));
  licenses_layout->addWidget(licenses_label);
  
  QHBoxLayout* license_text_layout = new QHBoxLayout();
  QListWidget* license_list = new QListWidget();
  license_text_layout->addWidget(license_list, 1);
  
  QTextEdit* license_text = new QTextEdit();
  license_text->setReadOnly(true);
  license_text_layout->addWidget(license_text, 3);
  
  QPushButton* about_qt_button = new QPushButton(tr("About Qt"));
  about_qt_button->setVisible(false);
  connect(about_qt_button, &QPushButton::clicked, qApp, &QApplication::aboutQt);
  license_text_layout->addWidget(about_qt_button, 3);
  
  licenses_layout->addLayout(license_text_layout);
  licenses_group->setLayout(licenses_layout);
  
  QHBoxLayout* buttons_layout = new QHBoxLayout();
  buttons_layout->addStretch(1);
  QPushButton* close_button = new QPushButton(tr("Close"));
  connect(close_button, &QPushButton::clicked, &about_dialog, &QDialog::accept);
  buttons_layout->addWidget(close_button);
  
  QVBoxLayout* layout = new QVBoxLayout(&about_dialog);
  layout->addLayout(top_layout);
  layout->addWidget(licenses_group);
  layout->addLayout(buttons_layout);
  about_dialog.setLayout(layout);
  
  // Add licenses to list
  vector<License> licenses;
  GetLicenses(&licenses);
  
  for (const License& license : licenses) {
    QListWidgetItem* item = new QListWidgetItem(QString::fromStdString(license.component));
    license_list->addItem(item);
  }
  
  connect(license_list, &QListWidget::currentItemChanged, [&]() {
    int index = license_list->currentRow();
    if (index >= 0 && index < licenses.size()) {
      if (licenses[index].license_text == string("about_qt")) {
        about_qt_button->setVisible(true);
        license_text->setVisible(false);
      } else {
        about_qt_button->setVisible(false);
        license_text->setVisible(true);
        license_text->setPlainText(QString::fromStdString(licenses[index].license_text));
      }
    }
  });
  
  about_dialog.exec();
}

void MainWindow::Settings() {
  QString dataset_path = QString::fromStdString(dataset_folder_path_);
  SettingsDialog settings_dialog(&dataset_path, &config_, nullptr, this);
  settings_dialog.DisablePathEditing();
  settings_dialog.ShowWarningForLiveChanges();
  if (settings_dialog.exec() == QDialog::Rejected) {
    return;
  }
  
  bad_slam_->config() = config_;
  
  bad_slam_->direct_ba().SetUseDescriptorResiduals(config_.use_photometric_residuals);
  bad_slam_->direct_ba().SetUseDepthResiduals(config_.use_geometric_residuals);
  bad_slam_->direct_ba().SetMinObservationCount(config_.min_observation_count);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping1(config_.min_observation_count_while_bootstrapping_1);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping2(config_.min_observation_count_while_bootstrapping_2);
}

void MainWindow::StartOrPause() {
  EnableRunButtons(false);
  backwards_ = false;
  run_ = !run_;
  if (run_) {
    run_condition_.notify_all();
  }
}

void MainWindow::SingleStep() {
  EnableRunButtons(false);
  backwards_ = false;
  single_step_ = true;
  run_ = true;
  run_condition_.notify_all();
}

void MainWindow::KFStep() {
  EnableRunButtons(false);
  backwards_ = false;
  single_step_ = true;
  create_kf_ = true;
  run_ = true;
  run_condition_.notify_all();
}

void MainWindow::SkipFrame() {
  EnableRunButtons(false);
  backwards_ = false;
  single_step_ = true;
  skip_frame_ = true;
  run_ = true;
  run_condition_.notify_all();
}

void MainWindow::SingleStepBackwards() {
  EnableRunButtons(false);
  backwards_ = true;
  single_step_ = true;
  run_ = true;
  run_condition_.notify_all();
}

void MainWindow::ManualBundleAdjustment() {
  // TODO: There should also be a help text for the options.
  auto add_option = [&](const QString& label, QWidget* widget, QGridLayout* layout, int* row) {
    layout->addWidget(new QLabel(label), *row, 0);
    layout->addWidget(widget, *row, 1);
    ++ *row;
  };
  
  QDialog ba_dialog(this);
  ba_dialog.setWindowTitle(tr("Manual bundle adjustment"));
  ba_dialog.setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  
  QGroupBox* components_group = new QGroupBox(tr("Components to optimize"));
  QVBoxLayout* components_layout = new QVBoxLayout();
  
  QCheckBox* optimize_depth_intrinsics_checkbox = new QCheckBox(tr("Depth intrinsics and deformation"));
  optimize_depth_intrinsics_checkbox->setChecked(config_.optimize_intrinsics && config_.use_geometric_residuals);
  components_layout->addWidget(optimize_depth_intrinsics_checkbox);
  
  QCheckBox* optimize_color_intrinsics_checkbox = new QCheckBox(tr("Color intrinsics"));
  optimize_color_intrinsics_checkbox->setChecked(config_.optimize_intrinsics && config_.use_photometric_residuals);
  components_layout->addWidget(optimize_color_intrinsics_checkbox);
  
  QCheckBox* optimize_poses_checkbox = new QCheckBox(tr("Keyframe poses"));
  optimize_poses_checkbox->setChecked(true);
  components_layout->addWidget(optimize_poses_checkbox);
  
  QCheckBox* optimize_geometry_checkbox = new QCheckBox(tr("Surfel geometry"));
  optimize_geometry_checkbox->setChecked(true);
  components_layout->addWidget(optimize_geometry_checkbox);
  
  components_layout->addStretch(1);
  components_group->setLayout(components_layout);
  
  
  QGroupBox* optimization_group = new QGroupBox(tr("Optimization"));
  QGridLayout* optimization_layout = new QGridLayout();
  int row = 0;
  
  QComboBox* approach_combo = new QComboBox();
  approach_combo->addItem(tr("Alternating"));
  approach_combo->addItem(tr("PCG-based Gauss-Newton"));
  constexpr int kPCGIndex = 1;  // index of the PCG item in approach_combo
  add_option(tr("Approach: "), approach_combo, optimization_layout, &row);
  
  QLineEdit* min_iterations_edit = new QLineEdit(QString::number(1));
  add_option(tr("Minimum iterations: "), min_iterations_edit, optimization_layout, &row);
  
  QLineEdit* max_iterations_edit = new QLineEdit(QString::number(10));
  add_option(tr("Maximum iterations: "), max_iterations_edit, optimization_layout, &row);
  
  QLineEdit* pcg_max_inner_iterations_edit = new QLineEdit(QString::number(30));
  add_option(tr("PCG: Maximum inner iterations: "), pcg_max_inner_iterations_edit, optimization_layout, &row);
  
  optimization_layout->setRowStretch(row, 1);
  optimization_group->setLayout(optimization_layout);
  
  
  QPushButton* close_button = new QPushButton(tr("Close"));
  connect(close_button, &QPushButton::clicked, &ba_dialog, &QDialog::reject);
  
  QPushButton* run_button = new QPushButton(tr("Run"));
  run_button->setDefault(true);
  
  QHBoxLayout* buttons_layout = new QHBoxLayout();
  buttons_layout->addStretch(1);
  buttons_layout->addWidget(run_button);
  buttons_layout->addWidget(close_button);
  
  
  QLabel* iterations_warning_label = new QLabel(tr(
      "Warning: Surfels are only created at the start of a BA scheme run."
      " Because of this, doing many iterations in one run might cause them to become too sparse."));
  iterations_warning_label->setWordWrap(true);
  
  QVBoxLayout* dialog_layout = new QVBoxLayout(&ba_dialog);
  dialog_layout->addWidget(components_group);
  dialog_layout->addWidget(optimization_group);
  dialog_layout->addWidget(iterations_warning_label);
  dialog_layout->addLayout(buttons_layout);
  ba_dialog.setLayout(dialog_layout);
  
  auto solver_changed = [&](int index) {
    pcg_max_inner_iterations_edit->setEnabled(index == kPCGIndex);
  };
  solver_changed(0);
  connect(approach_combo, QOverload<int>::of(&QComboBox::currentIndexChanged), solver_changed);
  
  connect(run_button, &QPushButton::clicked, [&]() {
    auto report_error = [&](const QString& option_name, const QString& invalid_value) {
      QMessageBox::warning(this, tr("Error"), tr("Entered value for %1 is invalid: %2").arg(option_name).arg(invalid_value));
    };
    
    bool ok = true;
    
    int min_iterations = min_iterations_edit->text().toInt(&ok);
    if (!ok) { report_error("min_iterations", min_iterations_edit->text()); return; }
    
    int max_iterations = max_iterations_edit->text().toInt(&ok);
    if (!ok) { report_error("max_iterations", max_iterations_edit->text()); return; }
    
    int pcg_max_inner_iterations = pcg_max_inner_iterations_edit->text().toInt(&ok);
    if (!ok) { report_error("pcg_max_inner_iterations", pcg_max_inner_iterations_edit->text()); return; }
    
    // Show a progress bar while the computation below is running.
    QProgressDialog progress(tr("Bundle adjustment ..."), tr("Stop"), 0, max_iterations, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    auto progress_function = [&](int step) {
      progress.setValue(step);
      if (progress.wasCanceled()) {
        LOG(INFO) << "Bundle adjustment stopped.";
        return false;
      } else {
        return true;
      }
    };
    
    BadSlamConfig& config = bad_slam_->config();
    bool old_use_pcg = config.use_pcg;
    config.use_pcg = approach_combo->currentIndex() == kPCGIndex;
    bad_slam_->RunBundleAdjustment(
        frame_index_,
        optimize_depth_intrinsics_checkbox->isChecked(),
        optimize_color_intrinsics_checkbox->isChecked(),
        optimize_poses_checkbox->isChecked(),
        optimize_geometry_checkbox->isChecked(),
        min_iterations,
        max_iterations,
        0,
        bad_slam_->direct_ba().keyframes().size() - 1,
        /*increase_ba_iteration_count*/ true,
        nullptr,
        nullptr,
        0,
        nullptr,
        pcg_max_inner_iterations,
        bad_slam_->direct_ba().keyframes().size(),
        progress_function);
    config.use_pcg = old_use_pcg;
    
    bad_slam_->UpdateOdometryVisualization(
        std::max(0, static_cast<int>(frame_index_) - 1),
        show_current_frame_);
    
    statusBar()->showMessage(tr("Manual bundle adjustment done"), 3000);
  });
  
  ba_dialog.exec();
}

void MainWindow::DensifySurfels() {
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  int reconstruction_sparse_surfel_cell_size = 1;
  int reconstruction_minimum_observation_count = 1;
  
  QDialog densify_dialog(this);
  densify_dialog.setWindowTitle(tr("Densify surfels"));
  densify_dialog.setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  QGridLayout* settings_layout = new QGridLayout();
  
  QLabel* surfel_creation_cell_size_label = new QLabel(tr("Surfel creation cell size (in SLAM: %1; densest: 1): ").arg(bad_slam_->direct_ba().sparse_surfel_cell_size()));
  QLineEdit* surfel_creation_cell_size_edit = new QLineEdit("1");
  settings_layout->addWidget(surfel_creation_cell_size_label, 0, 0);
  settings_layout->addWidget(surfel_creation_cell_size_edit, 0, 1);
  
  QLabel* minimum_observation_count_label = new QLabel(tr("Minimum observation count (in SLAM: %1; densest: 1): ").arg(bad_slam_->direct_ba().min_observation_count()));
  QLineEdit* minimum_observation_count_edit = new QLineEdit("1");
  settings_layout->addWidget(minimum_observation_count_label, 1, 0);
  settings_layout->addWidget(minimum_observation_count_edit, 1, 1);
  
  // TODO: Checkbox to enable using all frames for the densification, not only
  //       the keyframes; optionally after refining the localization of these
  //       frames (by localizing them against the surfel model, just like the
  //       keyframes).
  
  QHBoxLayout* buttons_layout = new QHBoxLayout();
  buttons_layout->addStretch(1);
  QPushButton* run_button = new QPushButton(tr("Run"));
  connect(run_button, &QPushButton::clicked, [&]() {
    bool ok = true;
    
    reconstruction_sparse_surfel_cell_size = surfel_creation_cell_size_edit->text().toInt(&ok);
    if (!ok) {
      QMessageBox::warning(&densify_dialog, tr("Error"), tr("Cannot parse the value for reconstruction_sparse_surfel_cell_size."));
      return;
    }
    
    reconstruction_minimum_observation_count = minimum_observation_count_edit->text().toInt(&ok);
    if (!ok) {
      QMessageBox::warning(&densify_dialog, tr("Error"), tr("Cannot parse the value for reconstruction_minimum_observation_count."));
      return;
    }
    
    densify_dialog.accept();
  });
  buttons_layout->addWidget(run_button);
  QPushButton* cancel_button = new QPushButton(tr("Cancel"));
  connect(cancel_button, &QPushButton::clicked, &densify_dialog, &QDialog::reject);
  buttons_layout->addWidget(cancel_button);
  
  QVBoxLayout* layout = new QVBoxLayout();
  layout->addLayout(settings_layout);
  layout->addLayout(buttons_layout);
  densify_dialog.setLayout(layout);
  
  if (densify_dialog.exec() == QDialog::Rejected) {
    return;
  }
  
  // Show a progress bar while the computation below is running.
  QProgressDialog progress(tr("Densifying surfels ..."), tr("Stop"), 0, bad_slam_->direct_ba().keyframes().size(), this);
  progress.setWindowModality(Qt::WindowModal);
  progress.setMinimumDuration(0);
  
  // Upscale the cfactor buffer to full resolution (since it uses the same
  // sparsification as the surfels).
  CUDABufferPtr<float> cfactor_buffer = bad_slam_->direct_ba().cfactor_buffer();
  CUDABufferPtr<float> scaled_cfactor_buffer(new CUDABuffer<float>(
      (bad_slam_->direct_ba().depth_camera().height() - 1) / reconstruction_sparse_surfel_cell_size + 1,
      (bad_slam_->direct_ba().depth_camera().width() - 1) / reconstruction_sparse_surfel_cell_size + 1));
  // TODO: Exclude pixels with zero observations from the interpolation
  //       (Note that we currently don't even store the observation count ...)
  UpscaleBufferBilinearly(/*stream*/ 0, *cfactor_buffer, scaled_cfactor_buffer.get());
  bad_slam_->direct_ba().SetCFactorBuffer(scaled_cfactor_buffer);
  
  // Run geometry-only BA without sparsification and without the descriptor residuals.
  // Use a sliding window for activating the keyframes to avoid allocating
  // a large number of surfels as an intermediate step.
  bool old_use_photometric_residuals = bad_slam_->direct_ba().use_descriptor_residuals();
  bad_slam_->direct_ba().SetUseDescriptorResiduals(false);
  
  int old_sparse_surfel_cell_size = bad_slam_->direct_ba().sparse_surfel_cell_size();
  bad_slam_->direct_ba().SetSparsificationSideFactor(reconstruction_sparse_surfel_cell_size);
  
  int old_minimum_observation_count = bad_slam_->direct_ba().min_observation_count();
  int old_minimum_observation_count1 = bad_slam_->direct_ba().min_observation_count_while_bootstrapping_1();
  int old_minimum_observation_count2 = bad_slam_->direct_ba().min_observation_count_while_bootstrapping_2();
  bad_slam_->direct_ba().SetMinObservationCount(reconstruction_minimum_observation_count);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping1(reconstruction_minimum_observation_count);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping2(reconstruction_minimum_observation_count);
  
  bool old_do_surfel_updates = bad_slam_->config().do_surfel_updates;
  bad_slam_->config().do_surfel_updates = true;
  
  constexpr int kWindowSize = 16;
  for (int window_start_index = - kWindowSize / 2;
      window_start_index < static_cast<int>(bad_slam_->direct_ba().keyframes().size());
      window_start_index += kWindowSize / 2) {
    progress.setValue(std::max(0, window_start_index));
    if (progress.wasCanceled()) {
      LOG(INFO) << "Surfel densification stopped.";
      break;
    }
    
    // TODO: Currently this runs BA twice (since the window is only advanced by
    //       half of its size) since the surfel radii are only updated at the
    //       end (and they have to be updated with the new
    //       sparse_surfel_cell_size). Make a separate function to update
    //       this?
    bad_slam_->RunBundleAdjustment(
        /*frame_index*/ std::max(0, static_cast<int>(frame_index_) - 1),
        /*optimize_depth_intrinsics*/ false,
        /*optimize_color_intrinsics*/ false,
        /*optimize_poses*/ false,
        /*optimize_geometry*/ true,
        /*min_iterations*/ 5,
        /*max_iterations*/ 10,
        window_start_index,
        window_start_index + kWindowSize - 1,
        /*increase_ba_iteration_count*/ true,
        nullptr, nullptr, 0, nullptr);
    
    UpdateSurfelUsage(bad_slam_->direct_ba().surfel_count());
  }
  
  bad_slam_->direct_ba().AssignColors(/*stream*/ 0);
  
  // Reset changed settings
  bad_slam_->direct_ba().SetCFactorBuffer(cfactor_buffer);
  bad_slam_->direct_ba().SetSparsificationSideFactor(old_sparse_surfel_cell_size);
  bad_slam_->direct_ba().SetMinObservationCount(old_minimum_observation_count);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping1(old_minimum_observation_count1);
  bad_slam_->direct_ba().SetMinObservationCountWhileBootstrapping2(old_minimum_observation_count2);
  bad_slam_->direct_ba().SetUseDescriptorResiduals(old_use_photometric_residuals);
  bad_slam_->config().do_surfel_updates = old_do_surfel_updates;
  
  // Update display
  bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
  render_window_->RenderFrame();
}

void MainWindow::Screenshot() {
  QString path = QFileDialog::getSaveFileName(this, tr("Save screenshot"), ".", tr("Image files (*.png; *.jpg; *.bmp)"));
  if (path.isEmpty()) {
    return;
  }
  
  render_window_->SaveScreenshot(path.toStdString().c_str(), true);
  
  // SaveScreenshot() renders a frame with altered background color.
  // To "reset" the 3D view back to normal appearance, render another frame.
  render_window_->RenderFrame();
}

void MainWindow::SelectKeyframeTool() {
  select_keyframe_act->setChecked(true);
  delete_keyframe_act->setChecked(false);
  render_window_->SetTool(BadSlamRenderWindow::Tool::kSelectKeyframe);
  if (!show_keyframes_act->isChecked()) {
    show_keyframes_act->setChecked(true);
    ShowStateChanged();
  }
}

void MainWindow::DeleteKeyframeTool() {
  select_keyframe_act->setChecked(false);
  delete_keyframe_act->setChecked(true);
  render_window_->SetTool(BadSlamRenderWindow::Tool::kSelectKeyframe);
  if (!show_keyframes_act->isChecked()) {
    show_keyframes_act->setChecked(true);
    ShowStateChanged();
  }
}

void MainWindow::ClickedKeyframe(int index) {
  // Do not risk race conditions, so only allow keyframe selection if the
  // reconstruction is not running.
  if (!bad_slam_set_ || is_running_) {
    return;
  }
  
  auto& keyframes = bad_slam_->direct_ba().keyframes();
  if (index < 0 ||
      index >= keyframes.size() ||
      !keyframes[index]) {
    return;
  }
  
  if (delete_keyframe_act->isChecked()) {
    bad_slam_->direct_ba().DeleteKeyframe(index, bad_slam_->loop_detector());
    bad_slam_->direct_ba().UpdateBAVisualization(/*stream*/ 0);
    render_window_->RenderFrame();
  } else if (select_keyframe_act->isChecked()) {
    KeyframeDialog* dialog = new KeyframeDialog(&frame_index_, index, keyframes[index], config_,  bad_slam_.get(), render_window_, this);
    dialog->show();
  }
}

bool MainWindow::UsingLiveInput() {
  return dataset_folder_path_.substr(0, 7) == string("live://");
}

void MainWindow::WorkerThreadMain() {
  // TODO: This duplicates a lot of code from main.cc, de-duplicate this!
  
  RealSenseInputThread rs_input;
  StructureInputThread structure_input;
  K4AInputThread k4a_input;
  int live_input = 0; // 1 realsense, 2 k4a, 3 structure
  
  if (dataset_folder_path_ == string("live://realsense")) {
    rs_input.Start(&rgbd_video_, &depth_scaling_);
    live_input = 1;
  } else if (dataset_folder_path_ == string("live://structure")) {
    structure_input.Start(&rgbd_video_, &depth_scaling_, config_);
    live_input = 3;
  } else if (dataset_folder_path_ == "live://k4a") {
    k4a_input.Start(
        &rgbd_video_, 
        &depth_scaling_, 
        config_.k4a_fps, 
        config_.k4a_resolution, 
        config_.k4a_factor,
        config_.k4a_use_depth,
        config_.k4a_mode,
        config_.k4a_exposure);
    live_input = 2;
  } else {
    if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(
            dataset_folder_path_.c_str(),
            nullptr,  // TODO:  trajectory_path.empty() ? nullptr : trajectory_path.c_str(),
            &rgbd_video_)) {
      LOG(ERROR) << "Could not read dataset at: " << dataset_folder_path_.c_str();
      
      unique_lock<mutex> lock(run_mutex_);
      quit_done_ = true;
      lock.unlock();
      quit_condition_.notify_all();
      
      emit FatalErrorSignal(tr("Could not load dataset: %1").arg(QString::fromStdString(dataset_folder_path_)));
      return;
    }
    
    CHECK_EQ(rgbd_video_.depth_frames_mutable()->size(),
             rgbd_video_.color_frames_mutable()->size());
    LOG(INFO) << "Read dataset with " << rgbd_video_.frame_count() << " frames";
  }
  
  // Initialize depth scale. This must be done after rs_input.Start() in the
  // live-input case since that may update depth_scaling_.
  config_.raw_to_float_depth = 1.0f / depth_scaling_;
  if (config_.max_depth * depth_scaling_ >= 1 << 15) {
    LOG(FATAL) << "max_depth * depth_scaling_ >= 1 << 15. This is too large"
                  " since it conflicts with the depth validity flag.";
  }
  
  // Get initial depth and color camera intrinsics. The generic camera type is
  // casted to the pinhole camera type; only pinhole cameras are supported.
  shared_ptr<Camera> initial_depth_camera(
      rgbd_video_.depth_camera()->Scaled(/*TODO depth_camera_scaling*/ 1));
  CHECK_EQ(initial_depth_camera->type_int(),
           static_cast<int>(Camera::Type::kPinholeCamera4f));
  *rgbd_video_.depth_camera_mutable() = initial_depth_camera;
  
  shared_ptr<Camera> initial_color_camera(
      rgbd_video_.color_camera()->Scaled(/*TODO color_camera_scaling*/ 1));
  CHECK_EQ(initial_color_camera->type_int(),
           static_cast<int>(Camera::Type::kPinholeCamera4f));
  *rgbd_video_.color_camera_mutable() = initial_color_camera;
  
  // Render widget and OpenGL context initialization
  OpenGLContext no_opengl_context;  // stores the original (non-existing) context
  render_window_->InitializeForCUDAInterop(
        config_.max_surfel_count,
        &opengl_context,
        &opengl_context_2,
        *initial_depth_camera);
  SwitchOpenGLContext(opengl_context, &no_opengl_context);
  
  // Extract ground truth trajectory for visualization.
  vector<Vec3f> gt_trajectory;
  // TODO
//   if (!trajectory_path.empty()) {
//     gt_trajectory.resize(rgbd_video_.frame_count());
//     for (usize frame_index = 0;
//          frame_index < rgbd_video_.frame_count();
//          ++ frame_index) {
//       gt_trajectory[frame_index] =
//           rgbd_video_.depth_frame(frame_index)->global_T_frame().translation();
//       
//       // Make sure that loading the ground truth does not accidentally influence
//       // the results (apart from setting the start frame's pose) by resetting
//       // all frame poses.
//       if (frame_index != static_cast<usize>(config_.start_frame)) {
//         rgbd_video_.depth_frame_mutable(frame_index)->SetGlobalTFrame(SE3f());
//         rgbd_video_.color_frame_mutable(frame_index)->SetGlobalTFrame(SE3f());
//       }
//     }
//   }
  
  // Initialize viewpoint and up direction of 3D visualization.
  if (config_.start_frame < static_cast<int>(rgbd_video_.frame_count())) {
    const auto& start_frame = rgbd_video_.depth_frame(config_.start_frame);
    
    if (!gt_trajectory.empty()) {
      // Assume that (0, 0, 1) is the global up direction in the ground truth
      // trajectory. The pose of the first keyframe is taken over from the
      // ground truth trajectory.
      render_window_->SetUpDirection(Vec3f(0, 0, 1));
    } else {
      // Set the up direction of the first frame as the global up direction.
      render_window_->SetUpDirection(
          start_frame->frame_T_global().rotationMatrix().transpose() *
              Vec3f(0, -1, 0));
    }
    
    // Set the look-at point where the first frame looks at.
    render_window_->SetView(start_frame->global_T_frame() * Vec3f(0, 0, 2),
                            start_frame->global_T_frame() * Vec3f(0, 0, -0.5f));
  }
  
  // Show gt_trajectory in the render window.
  if (!gt_trajectory.empty()) {
    render_window_->SetGroundTruthTrajectory(gt_trajectory);
  }
  
  
  // Load poses from a file?
  // TODO
//   if (!load_poses.empty()) {
//     // NOTE: We'd only need to load the poses here, not the whole dataset.
//     if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(
//         dataset_folder_path.c_str(), load_poses.c_str(), &rgbd_video_)) {
//       LOG(ERROR) << "Could not read pose file: " << load_poses;
//       return EXIT_FAILURE;
//     }
//   }
  
  // If end_frame is non-zero, remove all frames which would extend beyond
  // this length.
  if (config_.end_frame > 0 &&
      rgbd_video_.color_frames_mutable()->size() > static_cast<usize>(config_.end_frame)) {
    rgbd_video_.color_frames_mutable()->resize(config_.end_frame);
    rgbd_video_.depth_frames_mutable()->resize(config_.end_frame);
  }
  
  frame_index_ = config_.start_frame;
  
  // Initialize image pre-loading thread.
  PreLoadThread pre_load_thread(&rgbd_video_);
  
  // Initialize BAD SLAM.
  if (config_.enable_loop_detection) {
    boost::filesystem::path program_dir =
        boost::filesystem::path(program_path_).parent_path();
    config_.loop_detection_vocabulary_path =
        (program_dir / "resources" / "brief_k10L6.voc").string();
    config_.loop_detection_pattern_path =
        (program_dir / "resources" / "brief_pattern.yml").string();
    config_.loop_detection_images_width = rgbd_video_.color_camera()->width();
    config_.loop_detection_images_height = rgbd_video_.color_camera()->height();
  }
  
  bad_slam_.reset(new BadSlam(config_, &rgbd_video_,
                              render_window_, &opengl_context_2));
  bad_slam_set_ = true;
  
  if (!import_calibration_path_.empty()) {
    if (!LoadCalibration(&bad_slam_->direct_ba(), import_calibration_path_)) {
      unique_lock<mutex> lock(run_mutex_);
      quit_done_ = true;
      lock.unlock();
      quit_condition_.notify_all();
      emit FatalErrorSignal(tr("Could not load calibration: %1").arg(QString::fromStdString(import_calibration_path_)));
      return;
    }
  }
  
  render_window_->SetDirectBA(&bad_slam_->direct_ba());
  
  bad_slam_->direct_ba().SetVisualization(
      surfel_display_normals_action->isChecked(),
      surfel_display_descriptors_action->isChecked(),
      surfel_display_radii_action->isChecked());
  
  bad_slam_->direct_ba().SetIntrinsicsUpdatedCallback([&]() {
    if (std::this_thread::get_id() == gui_thread_id_) {
      IntrinsicsUpdated();
    } else {
      emit IntrinsicsUpdatedSignal();
    }
  });
  
  emit UpdateCurrentFrameSignal(0, rgbd_video_.frame_count());
  emit UpdateGPUMemoryUsageSignal();
  emit UpdateSurfelUsageSignal(bad_slam_->direct_ba().surfel_count());
  
  if (run_) {
    unique_lock<mutex> lock(run_mutex_);
    emit RunStateChangedSignal(true);  // blocking queued connection
  }
  
  
  // ### Main loop ###
  bool quit = false;
  for (frame_index_ = config_.start_frame;
       (live_input || frame_index_ < rgbd_video_.frame_count()) && !quit;
       backwards_ ? (-- frame_index_) : (++ frame_index_)) {
    pre_load_thread.WaitUntilDone();
    
    if (single_step_) {
      run_ = false;
      single_step_ = false;
    }
    if (skip_frame_) {
      skip_frame_ = false;
    }
    
    unique_lock<mutex> lock(run_mutex_);
    if (!run_) {
      // Avoid calling a blocking queued connection if quitting was requested
      if (quit_requested_) {
        break;
      }
      
      // Stop SLAM and update GUI
      bad_slam_->StopBAThreadAndWaitForIt();
      emit UpdateSurfelUsageSignal(bad_slam_->direct_ba().surfel_count());
      emit RunStateChangedSignal(false);  // blocking queued connection
      
      while (!run_ && !quit_requested_) {
        run_condition_.wait(lock);
      }
      if (quit_requested_) {
        break;
      }
      
      // Restart SLAM and update GUI
      emit RunStateChangedSignal(true);  // blocking queued connection
      if (bad_slam_->config().parallel_ba && !skip_frame_) {
        bad_slam_->RestartBAThread();
      }
    }
    lock.unlock();
    
    if (live_input == 1) {
      rs_input.GetNextFrame();
    } else if (live_input == 2) {
      k4a_input.GetNextFrame();
    } else if (live_input == 3) {
      structure_input.GetNextFrame();
    }
    
    // Get the current RGB-D frame's RGB and depth images. This may wait for I/O
    // to complete in case it did not complete in the pre-loading thread yet.
    rgbd_video_mutex_.lock();
    // const Image<Vec3u8>* rgb_image =
        rgbd_video_.color_frame_mutable(frame_index_)->GetImage().get();
    // const Image<u16>* depth_image =
        rgbd_video_.depth_frame_mutable(frame_index_)->GetImage().get();
    
    // Pre-load the next frame.
    if (frame_index_ < rgbd_video_.frame_count() - 1) {
      pre_load_thread.PreLoad(frame_index_ + 1);
    }
    
    // Optionally, visualize the input images.
    emit UpdateCurrentFrameImagesSignal(frame_index_, true);
    
    // Let BAD SLAM process the current RGB-D frame. This function does the
    // actual work.
    if (skip_frame_) {
      const SE3f& prev_global_T_frame = rgbd_video_.depth_frame_mutable(std::max<int>(0, static_cast<int>(frame_index_) - 1))->global_T_frame();
      rgbd_video_.color_frame_mutable(frame_index_)->SetGlobalTFrame(prev_global_T_frame);
      rgbd_video_.depth_frame_mutable(frame_index_)->SetGlobalTFrame(prev_global_T_frame);
      
      // Preprocess the frame such that its point cloud visualization will be available
      bad_slam_->PreprocessFrame(frame_index_, &bad_slam_->final_depth_buffer(), nullptr);
    } else {
      bad_slam_->ProcessFrame(frame_index_, create_kf_);
      create_kf_ = false;
    }
    
    // Update the 3D visualization.
    bad_slam_->UpdateOdometryVisualization(frame_index_, show_current_frame_);
    
    // Measure the frame time, and optionally restrict the frames per second.
    if (!skip_frame_) {
      bad_slam_->EndFrame();
    }
    
    // Release memory.
    rgbd_video_.depth_frame_mutable(frame_index_)->ClearImageAndDerivedData();
    rgbd_video_.color_frame_mutable(frame_index_)->ClearImageAndDerivedData();
    rgbd_video_mutex_.unlock();
    
    emit UpdateCurrentFrameSignal(frame_index_ + 1, rgbd_video_.frame_count());
    emit UpdateGPUMemoryUsageSignal();
    emit UpdateSurfelUsageSignal(bad_slam_->direct_ba().surfel_count());
  }  // end of main loop
  
  pre_load_thread.RequestExitAndWaitForIt();
  
  // Avoid interacting with the GUI thread when quitting was requested (which is
  // supposed to be waiting in this case, so calling a blocking queued
  // connection would deadlock).
  if (!quit_requested_) {
    bad_slam_->StopBAThreadAndWaitForIt();
    
    // Disable run buttons since the dataset finished.
    emit RunStateChangedSignal(false);  // blocking queued connection
    emit DatasetPlaybackFinishedSignal();
    
    // Do a final surfel count update
    emit UpdateSurfelUsageSignal(bad_slam_->direct_ba().surfel_count());
  }
  
  SwitchOpenGLContext(no_opengl_context);
  
  unique_lock<mutex> lock(run_mutex_);
  quit_done_ = true;
  lock.unlock();
  quit_condition_.notify_all();
}

void MainWindow::RunStateChanged(bool running) {
  bool using_live_input = UsingLiveInput();
  
  save_state_act->setEnabled(!using_live_input && !running);
  load_state_act->setEnabled(!using_live_input && !running);
  save_trajectory_act->setEnabled(!running);
  save_surfel_cloud_act->setEnabled(!running);
  save_calibration_act->setEnabled(!running);
  
  move_frame_manually_act->setEnabled(!running);
  clear_motion_model_act->setEnabled(!running);
  set_frame_index_act->setEnabled(!running);
  merge_closest_keyframes_act->setEnabled(!running);
  
  settings_act->setEnabled(!running);
  single_step_act->setEnabled(!running);
  single_step_backwards_act->setEnabled(!running);
  kf_step_act->setEnabled(!running);
  skip_frame_act->setEnabled(!running);
  ba_act->setEnabled(!running);
  densify_act->setEnabled(!running);
  select_keyframe_act->setEnabled(!running);
  
  if (running) {
    start_or_pause_act->setIcon(QIcon(":/badslam/pause.png"));
  } else {
    start_or_pause_act->setIcon(QIcon(":/badslam/run.png"));
  }
  
  EnableRunButtons(true);
  
  is_running_ = running;
}

void MainWindow::FatalError(const QString& text) {
  QMessageBox::warning(this, tr("Error"), text);
  close();
}

void MainWindow::DatasetPlaybackFinished() {
  start_or_pause_act->setIcon(QIcon(":/badslam/run.png"));
  EnableRunButtons(false);
  statusBar()->showMessage(tr("Dataset playback finished"), 3000);
}

void MainWindow::IntrinsicsUpdated() {
  if (!intrinsics_dialog) {
    return;
  }
  
  PinholeCamera4f color_camera = bad_slam_->direct_ba().color_camera();
  color_intrinsics_edit->setText(
      QString::number(color_camera.parameters()[0], 'g', 14) + " " +
      QString::number(color_camera.parameters()[1], 'g', 14) + " " +
      QString::number(color_camera.parameters()[2], 'g', 14) + " " +
      QString::number(color_camera.parameters()[3], 'g', 14));
  
  PinholeCamera4f depth_camera = bad_slam_->direct_ba().depth_camera();
  depth_intrinsics_edit->setText(
      QString::number(depth_camera.parameters()[0], 'g', 14) + " " +
      QString::number(depth_camera.parameters()[1], 'g', 14) + " " +
      QString::number(depth_camera.parameters()[2], 'g', 14) + " " +
      QString::number(depth_camera.parameters()[3], 'g', 14));
  
  depth_alpha_edit->setText(
      QString::number(bad_slam_->direct_ba().depth_params().a, 'g', 14));
  
  // Visualize depth deformation
  const auto& cfactor_buffer = bad_slam_->direct_ba().cfactor_buffer();
  Image<float> cfactor_buffer_cpu(cfactor_buffer->width(), cfactor_buffer->height());
  cfactor_buffer->DownloadAsync(0, &cfactor_buffer_cpu);
  
  vector<float> abs_factors;
  float min_cfactor = numeric_limits<float>::infinity();
  float max_cfactor = -numeric_limits<float>::infinity();
  abs_factors.reserve(cfactor_buffer_cpu.height() * cfactor_buffer_cpu.width());
  for (u32 y = 0; y < cfactor_buffer_cpu.height(); ++ y) {
    for (u32 x = 0; x < cfactor_buffer_cpu.width(); ++ x) {
      float cfactor = cfactor_buffer_cpu(x, y);
      min_cfactor = std::min(min_cfactor, cfactor);
      max_cfactor = std::max(max_cfactor, cfactor);
      abs_factors.emplace_back(fabs(cfactor));
    }
  }
  
  float depth_1 = RawToCalibratedDepth(bad_slam_->direct_ba().depth_params().a, min_cfactor, 3.0f);
  float depth_2 = RawToCalibratedDepth(bad_slam_->direct_ba().depth_params().a, max_cfactor, 3.0f);
  min_max_calibrated_depth_label->setText(tr("Min / max calibrated depth for 3m uncalibrated depth: %1 / %2")
      .arg(std::min(depth_1, depth_2))
      .arg(std::max(depth_1, depth_2)));
  
  Image<Vec3u8> debug_image(cfactor_buffer->width(), cfactor_buffer->height());
  
  for (u32 y = 0; y < cfactor_buffer_cpu.height(); ++ y) {
    for (u32 x = 0; x < cfactor_buffer_cpu.width(); ++ x) {
      float cfactor = cfactor_buffer_cpu(x, y);
      
      // Intensity based on distortion at 1m depth.
      constexpr float kVisualizationDepth = 1.0f;
      constexpr float max_displayed_deviation = 0.01f;
      float calib_depth = RawToCalibratedDepth(bad_slam_->direct_ba().depth_params().a, cfactor, kVisualizationDepth);
      u8 intensity = 255.99f * std::min(1.0f, fabs(calib_depth - kVisualizationDepth) / max_displayed_deviation);
      
      if (calib_depth > kVisualizationDepth) {
        debug_image(x, y) = Vec3u8(255, 255 - intensity, 255 - intensity);
      } else {
        debug_image(x, y) = Vec3u8(255 - intensity, 255, 255 - intensity);
      }
    }
  }
  
  depth_deformation_display->SetImage(debug_image);
}

void MainWindow::UpdateCurrentFrame(int current_frame, int frame_count) {
  status_label_->setText(tr("Frames processed: %1 / %2").arg(current_frame).arg(frame_count));
}

void MainWindow::UpdateGPUMemoryUsage() {
  size_t free_bytes;
  size_t total_bytes;
  CUDA_CHECKED_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
  size_t used_bytes = total_bytes - free_bytes;
  
  constexpr double kBytesToMiB = 1.0 / (1024.0 * 1024.0);
  int used_mib = static_cast<int>(kBytesToMiB * used_bytes + 0.5f);
  int total_mib = static_cast<int>(kBytesToMiB * total_bytes + 0.5f);
  
  gpu_memory_bar_->setRange(0, total_mib);
  gpu_memory_bar_->setValue(used_mib);
}

void MainWindow::UpdateSurfelUsage(int surfel_count) {
  surfel_count_bar_->setRange(0, config_.max_surfel_count);
  surfel_count_bar_->setValue(surfel_count);
  
  constexpr double kMillion = 1000000;
  double countM = surfel_count / kMillion;
  double maxM = config_.max_surfel_count / kMillion;
  surfel_count_bar_->setFormat(tr("Surfels: %1M / %2M").arg(countM, 0, 'g', 2).arg(maxM, 0, 'g', 2));
}

void MainWindow::UpdateCurrentFrameImages(int frame_index, bool images_in_use_elsewhere) {
  if (!current_frame_images_dialog) {
    return;
  }
  
  const Image<Vec3u8>* rgb_image;
  const Image<u16>* depth_image;
  
  if (images_in_use_elsewhere) {
    // Do not lock rgbd_video_mutex_, and do not unload the images after use.
    rgb_image = rgbd_video_.color_frame_mutable(frame_index)->GetImage().get();
    depth_image = rgbd_video_.depth_frame_mutable(frame_index)->GetImage().get();
  } else {
    // Lock rgbd_video_mutex_, and unload the images after use.
    rgbd_video_mutex_.lock();
    
    rgb_image = rgbd_video_.color_frame_mutable(frame_index)->GetImage().get();
    depth_image = rgbd_video_.depth_frame_mutable(frame_index)->GetImage().get();

    if (!rgb_image || !depth_image) {
      rgbd_video_mutex_.unlock();
      return;
    }
  }
  
  current_frame_color_display->SetImage(*rgb_image);
  current_frame_depth_display->SetImage(*depth_image);
  
  Image<Vec3u8> combined_image(depth_image->size());
  for (int y = 0; y < combined_image.height(); ++ y) {
    for (int x = 0; x < combined_image.width(); ++ x) {
      Vec3u8 color = rgb_image->at(x, y);
      u16 depth = depth_image->at(x, y);
      if (depth > 0) {
        float d = std::min(1.f, config_.raw_to_float_depth * depth / config_.max_depth);
        color = ((0.5f * color.cast<float>() + 0.5f * Vec3f(255.99f * (1 - d), 255.99f * d, 255)) + Vec3f::Constant(0.5f)).cast<u8>();
      }
      combined_image(x, y) = color;
    }
  }
  current_frame_combined_display->SetImage(combined_image);
  
  if (!images_in_use_elsewhere) {
    rgbd_video_.color_frame_mutable(frame_index)->ClearImageAndDerivedData();
    rgbd_video_.depth_frame_mutable(frame_index)->ClearImageAndDerivedData();
    
    rgbd_video_mutex_.unlock();
  }
  
  if (!current_frame_images_dialog->isVisible()) {
    current_frame_images_dialog->show();
  }
}

void MainWindow::ShowCurrentFrameImagesOnceSLAMInitialized() {
  if (!slam_init_timer_) {
    return;
  }
  if (bad_slam_set_) {
    ShowCurrentFrameImages();
    delete slam_init_timer_;
    slam_init_timer_ = nullptr;
  }
}

void MainWindow::EnableRunButtons(bool enable) {
  start_or_pause_act->setEnabled(enable);
  single_step_act->setEnabled(enable);
  single_step_backwards_act->setEnabled(enable);
  kf_step_act->setEnabled(enable);
  skip_frame_act->setEnabled(enable);
}

}
