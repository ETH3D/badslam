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

#include "badslam/gui_settings_window.h"

#include <boost/filesystem.hpp>
#include <QFileDialog>
#include <QString>
#include <QLabel>
#include <QLineEdit>
#include <QTabWidget>
#include <QPushButton>
#include <QBoxLayout>
#include <QGridLayout>
#include <QCheckBox>
#include <QMessageBox>
#include <QSettings>


namespace vis {

bool ShowSettingsWindow(string* dataset_path, BadSlamConfig* config, bool* start_paused) {
  QString qdataset_path = QString::fromStdString(*dataset_path);
  
  SettingsDialog settings_dialog(&qdataset_path, config, start_paused);
  QSize bestSize = settings_dialog.sizeHint();
  bestSize.setWidth(bestSize.width() * 2);
  settings_dialog.resize(bestSize);
  if (settings_dialog.exec() == QDialog::Rejected) {
    return false;
  }
  
  *dataset_path = qdataset_path.toStdString();
  return true;
}

SettingsDialog::SettingsDialog(QString* dataset_path, BadSlamConfig* config, bool* start_paused, QWidget* parent)
    : QDialog(parent),
      config(config),
      start_paused(start_paused),
      dataset_path(dataset_path) {
  setWindowTitle(tr("BAD SLAM Settings"));
  setWindowIcon(QIcon(":/badslam/badslam.png"));
  
  // Warning for live changes
  live_changes_warning = new QLabel(tr("<span style=\"color:red\">Warning: For many settings, changes after the reconstruction has been started will not be handled correctly.</span>"));
  live_changes_warning->setVisible(false);
  
  // Dataset path line
  QLabel* dataset_path_label = new QLabel(tr("Dataset: "));
  dataset_path_edit = new QLineEdit(*dataset_path);
  QPushButton* dataset_choose_button = new QPushButton(tr("..."));
  connect(dataset_choose_button, &QPushButton::clicked, this, &SettingsDialog::ChooseDatasetClicked);
  
  QHBoxLayout* dataset_layout = new QHBoxLayout();
  dataset_layout->addWidget(dataset_path_label);
  dataset_layout->addWidget(dataset_path_edit, 1);
  dataset_layout->addWidget(dataset_choose_button);
  
#ifdef BAD_SLAM_HAVE_REALSENSE
  QPushButton* realsense_button = new QPushButton(tr("RealSense live input"));
  connect(realsense_button, &QPushButton::clicked, this, &SettingsDialog::RealSenseLiveInputClicked);
  dataset_layout->addWidget(realsense_button);
#endif

#ifdef HAVE_K4A
  QPushButton* k4a_button = new QPushButton(tr("K4A live input"));
  connect(k4a_button, &QPushButton::clicked, this, &SettingsDialog::K4ALiveInputClicked);
  dataset_layout->addWidget(k4a_button);
#endif
  
  // TODO: Allow selecting the ground truth trajectory file
  
  // TODO: Maybe allow selecting some presets here in a second line (real-time, offline - low quality, offline - high quality)?
  
  // TODO: The help text must be available for the options as well. Also, this
  //       text should be stored only once for the GUI and command line interfaces,
  //       ideally in the config->h file such that we can also de-duplicate
  //       it with the comments there.
  auto add_option = [&](const QString& label, QWidget* widget, QGridLayout* layout, int* row) {
    layout->addWidget(new QLabel(label), *row, 0);
    layout->addWidget(widget, *row, 1);
    ++ *row;
  };
  
  
  // Settings tab: Dataset playback
  QWidget* dataset_playback_tab = new QWidget();
  QGridLayout* dataset_playback_layout = new QGridLayout();
  int row = 0;
  
  raw_to_float_depth_edit = new QLineEdit(QString::number(config->raw_to_float_depth));
  add_option(tr("Raw to metric depth conversion factor: "), raw_to_float_depth_edit, dataset_playback_layout, &row);
  
  target_frame_rate_edit = new QLineEdit(QString::number(config->target_frame_rate));
  add_option(tr("Target frame rate: "), target_frame_rate_edit, dataset_playback_layout, &row);
  
  restrict_fps_to_edit = new QLineEdit(QString::number(config->fps_restriction));
  add_option(tr("Restrict FPS to: "), restrict_fps_to_edit, dataset_playback_layout, &row);
  
  start_frame_edit = new QLineEdit(QString::number(config->start_frame));
  add_option(tr("Start frame: "), start_frame_edit, dataset_playback_layout, &row);
  
  end_frame_edit = new QLineEdit(QString::number(config->end_frame));
  add_option(tr("End frame: "), end_frame_edit, dataset_playback_layout, &row);
  
  pyramid_level_for_depth_edit = new QLineEdit(QString::number(config->pyramid_level_for_depth));
  add_option(tr("Pyramid level for depth: "), pyramid_level_for_depth_edit, dataset_playback_layout, &row);
  
  pyramid_level_for_color_edit = new QLineEdit(QString::number(config->pyramid_level_for_color));
  add_option(tr("Pyramid level for color: "), pyramid_level_for_color_edit, dataset_playback_layout, &row);
  
  load_poses_path_edit = new QLineEdit("");
  add_option(tr("Load poses from file: "), load_poses_path_edit, dataset_playback_layout, &row);
  
  dataset_playback_layout->setRowStretch(row, 1);
  dataset_playback_tab->setLayout(dataset_playback_layout);
  
  
  // Settings tab: Odometry
  QWidget* odometry_tab = new QWidget();
  QGridLayout* odometry_layout = new QGridLayout();
  row = 0;
  
  num_scales_edit = new QLineEdit(QString::number(config->num_scales));
  add_option(tr("Number of pyramid levels: "), num_scales_edit, odometry_layout, &row);
  
  use_motion_model_checkbox = new QCheckBox(tr("Use motion model"));
  use_motion_model_checkbox->setChecked(config->use_motion_model);
  odometry_layout->addWidget(use_motion_model_checkbox, row, 0, 1, 2);
  ++ row;
  
  odometry_layout->setRowStretch(row, 1);
  odometry_tab->setLayout(odometry_layout);
  
  
  // Settings tab: Bundle Adjustment
  QWidget* ba_tab = new QWidget();
  QGridLayout* ba_layout = new QGridLayout();
  row = 0;
  
  keyframe_interval_edit = new QLineEdit(QString::number(config->keyframe_interval));
  add_option(tr("Keyframe interval: "), keyframe_interval_edit, ba_layout, &row);
  
  max_num_ba_iterations_per_keyframe_edit = new QLineEdit(QString::number(config->max_num_ba_iterations_per_keyframe));
  add_option(tr("Max BA iteration count per keyframe: "), max_num_ba_iterations_per_keyframe_edit, ba_layout, &row);
  
  enable_deactivation_checkbox = new QCheckBox(tr("Enable deactivation of surfels and keyframes"));
  enable_deactivation_checkbox->setChecked(!config->disable_deactivation);
  ba_layout->addWidget(enable_deactivation_checkbox, row, 0, 1, 2);
  ++ row;
  
  use_geometric_residuals_checkbox = new QCheckBox(tr("Use geometric residuals"));
  use_geometric_residuals_checkbox->setChecked(config->use_geometric_residuals);
  ba_layout->addWidget(use_geometric_residuals_checkbox, row, 0, 1, 2);
  ++ row;
  
  use_photometric_residuals_checkbox = new QCheckBox(tr("Use photometric residuals"));
  use_photometric_residuals_checkbox->setChecked(config->use_photometric_residuals);
  ba_layout->addWidget(use_photometric_residuals_checkbox, row, 0, 1, 2);
  ++ row;
  
  optimize_intrinsics_checkbox = new QCheckBox(tr("Optimize camera intrinsics and depth deformation"));
  optimize_intrinsics_checkbox->setChecked(config->optimize_intrinsics);
  ba_layout->addWidget(optimize_intrinsics_checkbox, row, 0, 1, 2);
  ++ row;
  
  intrinsics_optimization_interval_edit = new QLineEdit(QString::number(config->intrinsics_optimization_interval));
  add_option(tr("Interval for intrinsics optimization (every Xth time BA is run): "), intrinsics_optimization_interval_edit, ba_layout, &row);
  
  // TODO: not in config
//   QLineEdit* final_ba_iterations_edit = new QLineEdit(QString::number(config->final_ba_iterations));
//   add_option(tr("Number of BA iterations to run after playback finishes: "), final_ba_iterations_edit, ba_layout, &row);
  
  do_surfel_updates_checkbox = new QCheckBox(tr("Perform surfel updates"));
  do_surfel_updates_checkbox->setChecked(config->do_surfel_updates);
  ba_layout->addWidget(do_surfel_updates_checkbox, row, 0, 1, 2);
  ++ row;
  
  parallel_ba_checkbox = new QCheckBox(tr("Run BA in parallel with odometry"));
  parallel_ba_checkbox->setChecked(config->parallel_ba);
  ba_layout->addWidget(parallel_ba_checkbox, row, 0, 1, 2);
  ++ row;
  
  use_pcg_checkbox = new QCheckBox(tr("Use PCG-based solver"));
  use_pcg_checkbox->setChecked(config->use_pcg);
  ba_layout->addWidget(use_pcg_checkbox, row, 0, 1, 2);
  ++ row;
  
  estimate_poses_checkbox = new QCheckBox(tr("Estimate poses"));
  estimate_poses_checkbox->setChecked(config->estimate_poses);
  ba_layout->addWidget(estimate_poses_checkbox, row, 0, 1, 2);
  ++ row;
  
  ba_layout->setRowStretch(row, 1);
  ba_tab->setLayout(ba_layout);
  
  
  // Settings tab: Memory
  QWidget* memory_tab = new QWidget();
  QGridLayout* memory_layout = new QGridLayout();
  row = 0;
  
  min_free_gpu_memory_mb_edit = new QLineEdit(QString::number(config->min_free_gpu_memory_mb));
  add_option(tr("Minimum GPU memory to remain free, in MB: "), min_free_gpu_memory_mb_edit, memory_layout, &row);
  
  memory_layout->setRowStretch(row, 1);
  memory_tab->setLayout(memory_layout);
  
  
  // Settings tab: Surfel reconstruction
  QWidget* surfel_reconstruction_tab = new QWidget();
  QGridLayout* surfel_reconstruction_layout = new QGridLayout();
  row = 0;
  
  max_surfel_count_edit = new QLineEdit(QString::number(config->max_surfel_count));
  add_option(tr("Maximum surfel count: "), max_surfel_count_edit, surfel_reconstruction_layout, &row);
  
  sparse_surfel_cell_size_edit = new QLineEdit(QString::number(config->sparse_surfel_cell_size));
  add_option(tr("Cell size for sparse surfel creation in pixels: "), sparse_surfel_cell_size_edit, surfel_reconstruction_layout, &row);
  
  surfel_merge_dist_factor_edit = new QLineEdit(QString::number(config->surfel_merge_dist_factor));
  add_option(tr("Factor on surfel merge distance: "), surfel_merge_dist_factor_edit, surfel_reconstruction_layout, &row);
  
  min_observation_count_while_bootstrapping_1_edit = new QLineEdit(QString::number(config->min_observation_count_while_bootstrapping_1));
  add_option(tr("Minimum surfel observation count (while bootstrapping, 1): "), min_observation_count_while_bootstrapping_1_edit, surfel_reconstruction_layout, &row);
  
  min_observation_count_while_bootstrapping_2_edit = new QLineEdit(QString::number(config->min_observation_count_while_bootstrapping_2));
  add_option(tr("Minimum surfel observation count (while bootstrapping, 2): "), min_observation_count_while_bootstrapping_2_edit, surfel_reconstruction_layout, &row);
  
  min_observation_count_edit = new QLineEdit(QString::number(config->min_observation_count));
  add_option(tr("Minimum surfel observation count: "), min_observation_count_edit, surfel_reconstruction_layout, &row);
  
  // TODO: Non-included option:
//   int reconstruction_sparse_surfel_cell_size = 1;
//   cmd_parser.NamedParameter(
//       "--reconstruction_sparsification",
//       &reconstruction_sparse_surfel_cell_size, /*required*/ false,
//       "Sparse surfel cell size for the final reconstruction that is done for"
//       " --export_reconstruction. See --sparsification.");
  
  surfel_reconstruction_layout->setRowStretch(row, 1);
  surfel_reconstruction_tab->setLayout(surfel_reconstruction_layout);
  
  
  // Settings tab: Loop closure
  QWidget* loop_closure_tab = new QWidget();
  QGridLayout* loop_closure_layout = new QGridLayout();
  row = 0;
  
  loop_closure_checkbox = new QCheckBox(tr("Enable loop detection"));
  loop_closure_checkbox->setChecked(config->enable_loop_detection);
  loop_closure_layout->addWidget(loop_closure_checkbox, row, 0, 1, 2);
  ++ row;
  
  parallel_loop_detection_checkbox = new QCheckBox(tr("Run loop detection in parallel with other tasks"));
  parallel_loop_detection_checkbox->setChecked(config->parallel_loop_detection);
  loop_closure_layout->addWidget(parallel_loop_detection_checkbox, row, 0, 1, 2);
  ++ row;
  
  loop_detection_image_frequency_edit = new QLineEdit(QString::number(config->loop_detection_image_frequency));
  add_option(tr("Image frequency: "), loop_detection_image_frequency_edit, loop_closure_layout, &row);
  
  loop_closure_layout->setRowStretch(row, 1);
  loop_closure_tab->setLayout(loop_closure_layout);
  
  
  // Settings tab: Depth preprocessing
  QWidget* depth_preprocessing_tab = new QWidget();
  QGridLayout* depth_preprocessing_layout = new QGridLayout();
  row = 0;
  
  max_depth_edit = new QLineEdit(QString::number(config->max_depth));
  add_option(tr("Maximum depth to use: "), max_depth_edit, depth_preprocessing_layout, &row);
  
  baseline_fx_edit = new QLineEdit(QString::number(config->baseline_fx));
  add_option(tr("Baseline * fx: "), baseline_fx_edit, depth_preprocessing_layout, &row);
  
  median_filter_and_densify_iterations_edit = new QLineEdit(QString::number(config->median_filter_and_densify_iterations));
  add_option(tr("Median filter and densify iterations: "), median_filter_and_densify_iterations_edit, depth_preprocessing_layout, &row);
  
  bilateral_filter_sigma_xy_edit = new QLineEdit(QString::number(config->bilateral_filter_sigma_xy));
  add_option(tr("Bilateral filter: sigma_xy: "), bilateral_filter_sigma_xy_edit, depth_preprocessing_layout, &row);
  
  bilateral_filter_radius_factor_edit = new QLineEdit(QString::number(config->bilateral_filter_radius_factor));
  add_option(tr("Bilateral filter: radius_factor: "), bilateral_filter_radius_factor_edit, depth_preprocessing_layout, &row);
  
  bilateral_filter_sigma_inv_depth_edit = new QLineEdit(QString::number(config->bilateral_filter_sigma_inv_depth));
  add_option(tr("Bilateral filter: sigma_inv_depth: "), bilateral_filter_sigma_inv_depth_edit, depth_preprocessing_layout, &row);
  
  depth_preprocessing_layout->setRowStretch(row, 1);
  depth_preprocessing_tab->setLayout(depth_preprocessing_layout);

  // Settings tab: K4A preprocessing
#ifdef HAVE_K4A
  QWidget* k4a_tab = new QWidget();
  QGridLayout* k4a_layout = new QGridLayout();
  row = 0;

  k4a_mode_edit = new QLineEdit("");
  add_option(tr("Mode(nfov) (nfov,nfov2x2,wfov,wfov2x2): "), k4a_mode_edit, k4a_layout, &row);

  k4a_fps_edit = new QLineEdit(QString::number(config->k4a_fps));
  add_option(tr("Fps(30) (30,15,5): "), k4a_fps_edit, k4a_layout, &row);

  k4a_resolution_edit = new QLineEdit(QString::number(config->k4a_resolution));
  add_option(tr("Resolution(720) (720,1080,1400): "), k4a_resolution_edit, k4a_layout, &row);

  k4a_factor_edit = new QLineEdit(QString::number(config->k4a_factor));
  add_option(tr("Downscaling factor (1)"), k4a_factor_edit, k4a_layout, &row);

  k4a_use_depth_edit = new QLineEdit(QString::number(config->k4a_use_depth));
  add_option(tr("Use depth and ir only (0)"), k4a_use_depth_edit, k4a_layout, &row);

  k4a_exposure_edit = new QLineEdit(QString::number(config->k4a_exposure));
  add_option(tr("Set exposure in ms(8000)"), k4a_exposure_edit, k4a_layout, &row);

  k4a_layout->setRowStretch(row, 1);
  k4a_tab->setLayout(k4a_layout);
#endif
  
  
  // Settings tab widget
  QTabWidget* tab_widget = new QTabWidget(this);
  tab_widget->setElideMode(Qt::TextElideMode::ElideRight);
  tab_widget->addTab(dataset_playback_tab, tr("Dataset playback"));
  tab_widget->addTab(depth_preprocessing_tab, tr("Depth preprocessing"));
  tab_widget->addTab(surfel_reconstruction_tab, tr("Surfel Reconstruction"));
  tab_widget->addTab(odometry_tab, tr("Odometry"));
  tab_widget->addTab(ba_tab, tr("Bundle Adjustment"));
  tab_widget->addTab(loop_closure_tab, tr("Loop closure"));
  tab_widget->addTab(memory_tab, tr("Memory"));
#ifdef HAVE_K4A
  tab_widget->addTab(k4a_tab, tr("Azure Kinect"));
#endif
  
  // Action buttons
  QHBoxLayout* buttons_layout = new QHBoxLayout();
  buttons_layout->addStretch(1);
  if (start_paused) {
    QPushButton* start_button = new QPushButton(tr("Start"));
    start_button->setDefault(true);
    connect(start_button, &QPushButton::clicked, this, &SettingsDialog::StartClicked);
    
    QPushButton* start_paused_button = new QPushButton(tr("Start (paused)"));
    connect(start_paused_button, &QPushButton::clicked, this, &SettingsDialog::StartPausedClicked);
    
    QPushButton* quit_button = new QPushButton(tr("Quit"));
    connect(quit_button, &QPushButton::clicked, this, &QDialog::reject);
    
    buttons_layout->addWidget(start_button);
    buttons_layout->addWidget(start_paused_button);
    buttons_layout->addWidget(quit_button);
  } else {
    QPushButton* ok_button = new QPushButton(tr("Ok"));
    ok_button->setDefault(true);
    connect(ok_button, &QPushButton::clicked, this, &SettingsDialog::OkClicked);
    
    QPushButton* cancel_button = new QPushButton(tr("Cancel"));
    connect(cancel_button, &QPushButton::clicked, this, &QDialog::reject);
    
    buttons_layout->addWidget(ok_button);
    buttons_layout->addWidget(cancel_button);
  }
  
  QVBoxLayout* dialog_layout = new QVBoxLayout(this);
  dialog_layout->addWidget(live_changes_warning);
  dialog_layout->addLayout(dataset_layout);
  dialog_layout->addWidget(tab_widget);
  dialog_layout->addLayout(buttons_layout);
  setLayout(dialog_layout);
}

void SettingsDialog::DisablePathEditing() {
  dataset_path_edit->setEnabled(false);
}

void SettingsDialog::ShowWarningForLiveChanges() {
  live_changes_warning->setVisible(true);
}

void SettingsDialog::ChooseDatasetClicked() {
  QSettings settings;
  QString dataset_path = QFileDialog::getExistingDirectory(
      this,
      tr("Choose dataset"),
      dataset_path_edit->text().isEmpty() ? settings.value("dataset_path").toString() : dataset_path_edit->text());
  if (dataset_path.isEmpty()) {
    return;
  }
  dataset_path_edit->setText(dataset_path);
}

void SettingsDialog::RealSenseLiveInputClicked() {
  dataset_path_edit->setText("live://realsense");
  
  if (QMessageBox::question(
      this,
      tr("RealSense live input"),
      tr("Set recommended default settings for Intel D435 live input?"
         " This will disable descriptor residuals, set --bilateral_filter_sigma_inv_depth to 0.01, set --max_depth to 4, and set --fps_restriction to 0."),
      QMessageBox::StandardButton::Yes | QMessageBox::StandardButton::No) == QMessageBox::StandardButton::Yes) {
    use_photometric_residuals_checkbox->setChecked(false);
    bilateral_filter_sigma_inv_depth_edit->setText("0.01");
    max_depth_edit->setText("4");
    restrict_fps_to_edit->setText("0");
  }
}
void SettingsDialog::K4ALiveInputClicked() {
  dataset_path_edit->setText("live://k4a");

  if (QMessageBox::question(
      this,
      tr("K4A live input"),
      tr("Set recommended default settings for Azure Kinect live input?"
         " This will disable descriptor residuals, set --bilateral_filter_sigma_inv_depth to 0.01, set --max_depth to 5, and set --fps_restriction to 0."),
      QMessageBox::StandardButton::Yes | QMessageBox::StandardButton::No) == QMessageBox::StandardButton::Yes) {
    use_photometric_residuals_checkbox->setChecked(false);
    bilateral_filter_sigma_inv_depth_edit->setText("0.01");
    max_depth_edit->setText("5");
    restrict_fps_to_edit->setText("0");
  }
}

void SettingsDialog::StartClicked() {
  if (ParseSettings()) {
    *start_paused = false;
    accept();
  }
}

void SettingsDialog::StartPausedClicked() {
  if (ParseSettings()) {
    *start_paused = true;
    accept();
  }
}

void SettingsDialog::OkClicked() {
  if (ParseSettings()) {
    accept();
  }
}

bool SettingsDialog::ParseSettings() {
  QSettings settings;
  bool ok = true;
  
  auto report_error = [&](const QString& option_name, const QString& invalid_value) {
    QMessageBox::warning(this, tr("Error"), tr("Entered value for %1 is invalid: %2").arg(option_name).arg(invalid_value));
  };
  
  // Validate the dataset path
  if (dataset_path_edit->text().isEmpty()) {
    QMessageBox::warning(this, tr("Error"), tr("Please choose a dataset or a source of live input."));
    return false;
  }
  *this->dataset_path = dataset_path_edit->text();
  settings.setValue("dataset_path", *this->dataset_path);
  if (!this->dataset_path->startsWith("live://")) {
    boost::filesystem::path dataset_path = this->dataset_path->toStdString();
    if (!boost::filesystem::exists(dataset_path / "depth") ||
        !boost::filesystem::exists(dataset_path / "rgb") ||
        !boost::filesystem::exists(dataset_path / "calibration.txt") ||
        !boost::filesystem::exists(dataset_path / "associated.txt")) {
      QMessageBox::warning(this, tr("Error"), tr("There does not seem to be a dataset at the chosen dataset path. It must contain the \"depth\" and \"rgb\" folders and the \"calibration.txt\" and \"associated.txt\" files."));
      return false;
    }
  }
  
  
  // Dataset playback settings
  config->raw_to_float_depth = raw_to_float_depth_edit->text().toDouble(&ok);
  if (!ok) { report_error("raw_to_float_depth", raw_to_float_depth_edit->text()); return false; }
  
  config->target_frame_rate = target_frame_rate_edit->text().toDouble(&ok);
  if (!ok) { report_error("target_frame_rate", target_frame_rate_edit->text()); return false; }
  
  config->fps_restriction = restrict_fps_to_edit->text().toInt(&ok);
  if (!ok) { report_error("fps_restriction", restrict_fps_to_edit->text()); return false; }
  
  config->start_frame = start_frame_edit->text().toInt(&ok);
  if (!ok) { report_error("start_frame", start_frame_edit->text()); return false; }
  
  config->end_frame = end_frame_edit->text().toInt(&ok);
  if (!ok) { report_error("end_frame", end_frame_edit->text()); return false; }
  
  config->pyramid_level_for_depth = pyramid_level_for_depth_edit->text().toInt(&ok);
  if (!ok) { report_error("pyramid_level_for_depth", pyramid_level_for_depth_edit->text()); return false; }
  
  config->pyramid_level_for_color = pyramid_level_for_color_edit->text().toInt(&ok);
  if (!ok) { report_error("pyramid_level_for_color", pyramid_level_for_color_edit->text()); return false; }
  
  // TODO: load_poses_path_edit
  
  
  // Odometry settings
  config->num_scales = num_scales_edit->text().toInt(&ok);
  if (!ok) { report_error("num_scales", num_scales_edit->text()); return false; }
  
  config->use_motion_model = use_motion_model_checkbox->isChecked();
  
  
  // BA settings
  config->keyframe_interval = keyframe_interval_edit->text().toInt(&ok);
  if (!ok) { report_error("keyframe_interval", keyframe_interval_edit->text()); return false; }
  
  config->max_num_ba_iterations_per_keyframe = max_num_ba_iterations_per_keyframe_edit->text().toInt(&ok);
  if (!ok) { report_error("max_num_ba_iterations_per_keyframe", max_num_ba_iterations_per_keyframe_edit->text()); return false; }
  
  config->disable_deactivation = !enable_deactivation_checkbox->isChecked();
  
  config->use_geometric_residuals = use_geometric_residuals_checkbox->isChecked();
  
  config->use_photometric_residuals = use_photometric_residuals_checkbox->isChecked();
  
  config->optimize_intrinsics = optimize_intrinsics_checkbox->isChecked();
  
  config->intrinsics_optimization_interval = intrinsics_optimization_interval_edit->text().toInt(&ok);
  if (!ok) { report_error("intrinsics_optimization_interval", intrinsics_optimization_interval_edit->text()); return false; }
  
  config->do_surfel_updates = do_surfel_updates_checkbox->isChecked();
  
  config->parallel_ba = parallel_ba_checkbox->isChecked();
  
  config->use_pcg = use_pcg_checkbox->isChecked();
  
  config->estimate_poses = estimate_poses_checkbox->isChecked();
  
  
  // Memory settings
  config->min_free_gpu_memory_mb = min_free_gpu_memory_mb_edit->text().toInt(&ok);
  if (!ok) { report_error("min_free_gpu_memory_mb", min_free_gpu_memory_mb_edit->text()); return false; }
  
  
  // Surfel reconstruction settings
  config->max_surfel_count = max_surfel_count_edit->text().toInt(&ok);
  if (!ok) { report_error("max_surfel_count", max_surfel_count_edit->text()); return false; }
  
  config->sparse_surfel_cell_size = sparse_surfel_cell_size_edit->text().toInt(&ok);
  if (!ok) { report_error("sparse_surfel_cell_size", sparse_surfel_cell_size_edit->text()); return false; }
  
  config->surfel_merge_dist_factor = surfel_merge_dist_factor_edit->text().toDouble(&ok);
  if (!ok) { report_error("surfel_merge_dist_factor", surfel_merge_dist_factor_edit->text()); return false; }
  
  config->min_observation_count_while_bootstrapping_1 = min_observation_count_while_bootstrapping_1_edit->text().toInt(&ok);
  if (!ok) { report_error("min_observation_count_while_bootstrapping_1", min_observation_count_while_bootstrapping_1_edit->text()); return false; }
  
  config->min_observation_count_while_bootstrapping_2 = min_observation_count_while_bootstrapping_2_edit->text().toInt(&ok);
  if (!ok) { report_error("min_observation_count_while_bootstrapping_2", min_observation_count_while_bootstrapping_2_edit->text()); return false; }
  
  config->min_observation_count = min_observation_count_edit->text().toInt(&ok);
  if (!ok) { report_error("min_observation_count", min_observation_count_edit->text()); return false; }
  
  
  // Loop closure settings
  config->enable_loop_detection = loop_closure_checkbox->isChecked();
  
  config->parallel_loop_detection = parallel_loop_detection_checkbox->isChecked();
  
  config->loop_detection_image_frequency = loop_detection_image_frequency_edit->text().toInt(&ok);
  if (!ok) { report_error("loop_detection_image_frequency", loop_detection_image_frequency_edit->text()); return false; }
  
  
  // Depth preprocessing settings
  config->max_depth = max_depth_edit->text().toDouble(&ok);
  if (!ok) { report_error("max_depth", max_depth_edit->text()); return false; }
  
  config->baseline_fx = baseline_fx_edit->text().toDouble(&ok);
  if (!ok) { report_error("baseline_fx", baseline_fx_edit->text()); return false; }
  
  config->median_filter_and_densify_iterations = median_filter_and_densify_iterations_edit->text().toInt(&ok);
  if (!ok) { report_error("median_filter_and_densify_iterations", median_filter_and_densify_iterations_edit->text()); return false; }
  
  config->bilateral_filter_sigma_xy = bilateral_filter_sigma_xy_edit->text().toDouble(&ok);
  if (!ok) { report_error("bilateral_filter_sigma_xy", bilateral_filter_sigma_xy_edit->text()); return false; }
  
  config->bilateral_filter_radius_factor = bilateral_filter_radius_factor_edit->text().toDouble(&ok);
  if (!ok) { report_error("bilateral_filter_radius_factor", bilateral_filter_radius_factor_edit->text()); return false; }
  
  config->bilateral_filter_sigma_inv_depth = bilateral_filter_sigma_inv_depth_edit->text().toDouble(&ok);
  if (!ok) { report_error("bilateral_filter_sigma_inv_depth", bilateral_filter_sigma_inv_depth_edit->text()); return false; }

  // k4a settings
  // TODO Silvano error check
#ifdef HAVE_K4A
  config->k4a_mode = k4a_mode_edit->text().toStdString();
  config->k4a_fps = k4a_fps_edit->text().toInt(&ok);
  config->k4a_resolution = k4a_resolution_edit->text().toInt(&ok);
  config->k4a_factor = k4a_factor_edit->text().toInt(&ok);
  config->k4a_use_depth = k4a_use_depth_edit->text().toInt(&ok);
  config->k4a_exposure = k4a_exposure_edit->text().toInt(&ok);
#endif
  
  return true;
}

}
