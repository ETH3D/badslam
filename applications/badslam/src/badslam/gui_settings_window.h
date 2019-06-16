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

#include <memory>

#include <libvis/libvis.h>
#include <QDialog>
#include <QString>
#include <QLineEdit>
#include <QCheckBox>
#include <QLabel>

#include "badslam/bad_slam_config.h"

namespace vis {

bool ShowSettingsWindow(string* dataset_path, BadSlamConfig* config, bool* start_paused);

class SettingsDialog : public QDialog {
 Q_OBJECT
 public:
  SettingsDialog(QString* dataset_path, BadSlamConfig* config, bool* start_paused, QWidget* parent = nullptr);
  
  void DisablePathEditing();
  void ShowWarningForLiveChanges();
  
 public slots:
  void ChooseDatasetClicked();
  void RealSenseLiveInputClicked();
  
  void StartClicked();
  void StartPausedClicked();
  
  void OkClicked();
  
 private:
  bool ParseSettings();
  
  QLabel* live_changes_warning;
  
  QLineEdit* dataset_path_edit;
  
  // Dataset playback settings
  QLineEdit* raw_to_float_depth_edit;
  QLineEdit* target_frame_rate_edit;
  QLineEdit* restrict_fps_to_edit;
  QLineEdit* start_frame_edit;
  QLineEdit* end_frame_edit;
  QLineEdit* pyramid_level_for_depth_edit;
  QLineEdit* pyramid_level_for_color_edit;
  QLineEdit* load_poses_path_edit;
  
  // Odometry settings
  QLineEdit* num_scales_edit;
  QCheckBox* use_motion_model_checkbox;
  
  // BA settings
  QLineEdit* keyframe_interval_edit;
  QLineEdit* max_num_ba_iterations_per_keyframe_edit;
  QCheckBox* enable_deactivation_checkbox;
  QCheckBox* use_geometric_residuals_checkbox;
  QCheckBox* use_photometric_residuals_checkbox;
  QCheckBox* optimize_intrinsics_checkbox;
  QLineEdit* intrinsics_optimization_interval_edit;
  QCheckBox* do_surfel_updates_checkbox;
  QCheckBox* parallel_ba_checkbox;
  QCheckBox* use_pcg_checkbox;
  QCheckBox* estimate_poses_checkbox;
  
  // Memory settings
  QLineEdit* min_free_gpu_memory_mb_edit;
  
  // Surfel reconstruction settings
  QLineEdit* max_surfel_count_edit;
  QLineEdit* sparse_surfel_cell_size_edit;
  QLineEdit* surfel_merge_dist_factor_edit;
  QLineEdit* min_observation_count_while_bootstrapping_1_edit;
  QLineEdit* min_observation_count_while_bootstrapping_2_edit;
  QLineEdit* min_observation_count_edit;
  
  // Loop closure settings
  QCheckBox* loop_closure_checkbox;
  QCheckBox* parallel_loop_detection_checkbox;
  QLineEdit* loop_detection_image_frequency_edit;
  
  // Depth preprocessing settings
  QLineEdit* max_depth_edit;
  QLineEdit* baseline_fx_edit;
  QLineEdit* median_filter_and_densify_iterations_edit;
  QLineEdit* bilateral_filter_sigma_xy_edit;
  QLineEdit* bilateral_filter_radius_factor_edit;
  QLineEdit* bilateral_filter_sigma_inv_depth_edit;
  
  BadSlamConfig* config;
  bool* start_paused;
  QString* dataset_path;
};

}
