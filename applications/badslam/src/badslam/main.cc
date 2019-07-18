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


#define LIBVIS_ENABLE_TIMING

// librealsense must be included before any Qt include because some foreach will
// be misinterpreted otherwise
#include "badslam/input_realsense.h"
#include "badslam/input_structure.h"
#include "badslam/input_azurekinect.h"

#include <boost/filesystem.hpp>
#include <libvis/command_line_parser.h>
#include <libvis/cuda/cuda_auto_tuner.h>
#include <libvis/cuda/cuda_buffer.h>
#include <libvis/image_display.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video_io_tum_dataset.h>
#include <libvis/sophus.h>
#include <libvis/timing.h>
#include <QApplication>
#include <QSurfaceFormat>
#include <signal.h>

#include "badslam/bad_slam.h"
#include "badslam/cuda_depth_processing.cuh"
#include "badslam/cuda_image_processing.cuh"
#include "badslam/cuda_image_processing.h"
#include "badslam/direct_ba.h"
#include "badslam/gui_main_window.h"
#include "badslam/gui_settings_window.h"
#include "badslam/io.h"
#include "badslam/pre_load_thread.h"
#include "badslam/render_window.h"
#include "badslam/util.cuh"
#include "badslam/util.h"

using namespace vis;


int LIBVIS_QT_MAIN(int argc, char** argv) {
  // Initialize libvis
#ifdef WIN32
  ImageIOLibPngRegistrator image_io_libpng_registrator_;
  ImageIONetPBMRegistrator image_io_netpbm_registrator_;
#ifdef LIBVIS_HAVE_QT
  ImageIOQtRegistrator image_io_qt_registrator_;
#endif
#endif
  
  // Ignore SIGTTIN and SIGTTOU. I am not sure why they occurred: it seems that
  // they should only occur for background processes trying to interact with the
  // terminal, but they seemingly happened to me while there was no background
  // process and they interfered with using gdb.
  // TODO: Find out the reason for getting those signals
#ifndef WIN32
  signal(SIGTTIN, SIG_IGN);
  signal(SIGTTOU, SIG_IGN);
#endif
  
  BadSlamConfig bad_slam_config;
  
  // ### Parse parameters ###
  CommandLineParser cmd_parser(argc, argv);
  
  // Dataset playback parameters.
  float depth_scaling = 5000;  // The default is for TUM RGB-D datasets.
  cmd_parser.NamedParameter(
      "--depth_scaling", &depth_scaling, /*required*/ false,
      "Input depth scaling: input_depth = depth_scaling * depth_in_meters. The "
      "default is for TUM RGB-D benchmark datasets.");
  
  cmd_parser.NamedParameter(
      "--target_frame_rate", &bad_slam_config.target_frame_rate,
      /*required*/ false,
      bad_slam_config.target_frame_rate_help);
  
  cmd_parser.NamedParameter(
      "--restrict_fps_to", &bad_slam_config.fps_restriction, /*required*/ false,
      bad_slam_config.fps_restriction_help);
  
  cmd_parser.NamedParameter(
      "--start_frame", &bad_slam_config.start_frame, /*required*/ false,
      bad_slam_config.start_frame_help);
  
  cmd_parser.NamedParameter(
      "--end_frame", &bad_slam_config.end_frame, /*required*/ false,
      bad_slam_config.end_frame_help);
  
  cmd_parser.NamedParameter(
      "--pyramid_level_for_depth", &bad_slam_config.pyramid_level_for_depth,
      /*required*/ false, bad_slam_config.pyramid_level_for_depth_help);
  
  cmd_parser.NamedParameter(
      "--pyramid_level_for_color", &bad_slam_config.pyramid_level_for_color,
      /*required*/ false, bad_slam_config.pyramid_level_for_color_help);
  
  
  // Odometry parameters
  cmd_parser.NamedParameter(
      "--num_scales", &bad_slam_config.num_scales, /*required*/ false,
      bad_slam_config.num_scales_help);
  
  bad_slam_config.use_motion_model =
      !cmd_parser.Flag("--no_motion_model", "Disables the constant motion model that is used to predict the next frame's pose.");
  
  
  // Bundle adjustment parameters.
  cmd_parser.NamedParameter(
      "--keyframe_interval", &bad_slam_config.keyframe_interval,
      /*required*/ false, bad_slam_config.keyframe_interval_help);
  
  cmd_parser.NamedParameter(
      "--max_num_ba_iterations_per_keyframe",
      &bad_slam_config.max_num_ba_iterations_per_keyframe, /*required*/ false,
      bad_slam_config.max_num_ba_iterations_per_keyframe_help);
  
  bad_slam_config.disable_deactivation =
      !cmd_parser.Flag("--use_deactivation", "Enables deactivation of surfels "
      "and keyframes during bundle adjustment.");
  
  bad_slam_config.use_geometric_residuals =
      !cmd_parser.Flag("--no_geometric_residuals", "Disables the use of geometric"
      " residuals (comparing depth images and surfel positions).");
  
  bad_slam_config.use_photometric_residuals =
      !cmd_parser.Flag("--no_photometric_residuals", "Disables the use of"
      " photometric residuals (comparing visible-light images and surfel"
      " descriptors).");
  
  bad_slam_config.optimize_intrinsics =
      cmd_parser.Flag("--optimize_intrinsics", "Perform self-calibration of"
      " camera intrinsics and depth deformation during operation.");
  
  cmd_parser.NamedParameter(
      "--intrinsics_optimization_interval",
      &bad_slam_config.intrinsics_optimization_interval, /*required*/ false,
      bad_slam_config.intrinsics_optimization_interval_help);
  
  int final_ba_iterations = 0;
  cmd_parser.NamedParameter(
      "--final_ba_iterations", &final_ba_iterations, /*required*/ false,
      "Specifies a number of BA iterations to perform after dataset playback "
      "finishes (applies to command line mode only, not to the GUI).");
  
  bad_slam_config.do_surfel_updates =
      !cmd_parser.Flag("--no_surfel_updates", "Disables surfel updates "
      "(creation, merging) during BA. Only new keyframes will generate new "
      "surfels. Outlier surfels are still deleted.");
  
  bad_slam_config.parallel_ba =
      !cmd_parser.Flag("--sequential_ba", "Performs bundle adjustment "
      "sequentially instead of in parallel to odometry.");
  
  bad_slam_config.use_pcg =
      cmd_parser.Flag(
          "--use_pcg",
          "Use a PCG (preconditioned conjugate gradients) solver on the"
          " Gauss-Newton update equation, instead of the default alternating"
          " optimization.");
  
  
  // Memory parameters.
  cmd_parser.NamedParameter(
      "--min_free_gpu_memory_mb", &bad_slam_config.min_free_gpu_memory_mb,
      /*required*/ false, bad_slam_config.min_free_gpu_memory_mb_help);
  
  
  // Surfel reconstruction parameters.
  cmd_parser.NamedParameter(
      "--max_surfel_count", &bad_slam_config.max_surfel_count,
      /*required*/ false, bad_slam_config.max_surfel_count_help);
  
  cmd_parser.NamedParameter(
      "--sparsification", &bad_slam_config.sparse_surfel_cell_size,
      /*required*/ false, bad_slam_config.sparse_surfel_cell_size_help);
  
  cmd_parser.NamedParameter(
      "--surfel_merge_dist_factor", &bad_slam_config.surfel_merge_dist_factor,
      /*required*/ false, bad_slam_config.surfel_merge_dist_factor_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count_while_bootstrapping_1",
      &bad_slam_config.min_observation_count_while_bootstrapping_1,
      /*required*/ false, bad_slam_config.min_observation_count_while_bootstrapping_1_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count_while_bootstrapping_2",
      &bad_slam_config.min_observation_count_while_bootstrapping_2,
      /*required*/ false, bad_slam_config.min_observation_count_while_bootstrapping_2_help);
  
  cmd_parser.NamedParameter(
      "--min_observation_count", &bad_slam_config.min_observation_count,
      /*required*/ false, bad_slam_config.min_observation_count_help);
  
  int reconstruction_sparse_surfel_cell_size = 1;
  cmd_parser.NamedParameter(
      "--reconstruction_sparsification",
      &reconstruction_sparse_surfel_cell_size, /*required*/ false,
      "Sparse surfel cell size for the final reconstruction that is done for"
      " --export_reconstruction. See --sparsification.");
  
  
  // Loop closure parameters.
  bad_slam_config.enable_loop_detection = !cmd_parser.Flag(
      "--no_loop_detection", "Disables loop closure search.");
  
  bad_slam_config.parallel_loop_detection = !cmd_parser.Flag(
      "--sequential_loop_detection",
      "Runs loop detection sequentially instead of in parallel.");
  
  cmd_parser.NamedParameter(
      "--loop_detection_image_frequency",
      &bad_slam_config.loop_detection_image_frequency, /*required*/ false,
      bad_slam_config.loop_detection_image_frequency_help);
  
  
  // Depth preprocessing parameters.
  cmd_parser.NamedParameter(
      "--max_depth", &bad_slam_config.max_depth, /*required*/ false,
      bad_slam_config.max_depth_help);
  
  cmd_parser.NamedParameter(
      "--baseline_fx", &bad_slam_config.baseline_fx, /*required*/ false,
      bad_slam_config.baseline_fx_help);
  
  cmd_parser.NamedParameter(
      "--median_filter_and_densify_iterations",
      &bad_slam_config.median_filter_and_densify_iterations, /*required*/ false,
      bad_slam_config.median_filter_and_densify_iterations_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_xy", &bad_slam_config.bilateral_filter_sigma_xy,
      /*required*/ false, bad_slam_config.bilateral_filter_sigma_xy_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_radius_factor",
      &bad_slam_config.bilateral_filter_radius_factor, /*required*/ false,
      bad_slam_config.bilateral_filter_radius_factor_help);
  
  cmd_parser.NamedParameter(
      "--bilateral_filter_sigma_inv_depth",
      &bad_slam_config.bilateral_filter_sigma_inv_depth, /*required*/ false,
      bad_slam_config.bilateral_filter_sigma_inv_depth_help);
  
  
  // Visualization parameters.
  bool gui = cmd_parser.Flag(
      "--gui", "Show the GUI (starting with the settings window).");
  
  bool gui_run = cmd_parser.Flag(
      "--gui_run", "Show the GUI (and start running immediately).");
  
  bool show_input_images = cmd_parser.Flag(
      "--show_input_images", "Displays the input images.");
  
  float splat_half_extent_in_pixels = 3.0f;
  cmd_parser.NamedParameter(
      "--splat_half_extent_in_pixels", &splat_half_extent_in_pixels,
      /*required*/ false,
      "Half splat quad extent in pixels.");
  
  int window_default_width = 1280;
  cmd_parser.NamedParameter(
      "--window_default_width", &window_default_width,
      /*required*/ false,
      "Default width of the 3D visualization window.");
  
  int window_default_height = 720;
  cmd_parser.NamedParameter(
      "--window_default_height", &window_default_height,
      /*required*/ false,
      "Default height of the 3D visualization window.");
  
  bool show_current_frame_cloud =
      cmd_parser.Flag("--show_current_frame_cloud",
                      "Visualize the point cloud of the current frame.");
  
  
  // Auto-tuning.
  int auto_tuning_iteration = -1;
  cmd_parser.NamedParameter(
      "--auto_tuning_iteration", &auto_tuning_iteration, /*required*/ false,
      "Used by the auto-tuning script to signal that a tuning iteration is"
      " used.");
  
  
  // Output paths.
  std::string export_point_cloud_path;
  cmd_parser.NamedParameter(
      "--export_point_cloud", &export_point_cloud_path, /*required*/ false,
      "Save the final surfel point cloud to the given path (as a PLY file). Applies to the command line mode only, not to the GUI.");
  
  std::string export_reconstruction_path;
  cmd_parser.NamedParameter(
      "--export_reconstruction", &export_reconstruction_path, /*required*/ false,
      "Creates a reconstruction at the end (without, or with less"
      " sparsification) and saves it as a point cloud to the given path (as a"
      " PLY file). Applies to the command line mode only, not to the GUI.");
  
  std::string export_calibration_path;
  cmd_parser.NamedParameter(
      "--export_calibration", &export_calibration_path, /*required*/ false,
      "Save the final calibration to the given base path (as three files, with"
      " extensions .depth_intrinsics.txt, .color_intrinsics.txt, and"
      " .deformation.txt). Applies to the command line mode only, not to the GUI.");
  
  std::string export_final_timings_path;
  cmd_parser.NamedParameter(
      "--export_final_timings", &export_final_timings_path, /*required*/ false,
      "Save the final aggregated timing statistics to the given text file. Applies to the command line mode only, not to the GUI.");
  
  std::string save_timings_path;
  cmd_parser.NamedParameter(
      "--save_timings", &save_timings_path, /*required*/ false,
      "Save the detailed BA timings (for every time BA is run) to the given"
      " file. Applies to the command line mode only, not to the GUI.");
  
  std::string export_poses_path;
  cmd_parser.NamedParameter(
      "--export_poses", &export_poses_path, /*required*/ false,
      "Save the final poses to the given text file in TUM RGB-D format. Applies to the command line mode only, not to the GUI.");
  
  
  // Input paths.
  std::string import_calibration_path;
  cmd_parser.NamedParameter(
      "--import_calibration", &import_calibration_path, /*required*/ false,
      "Load the calibration from the given base path (as two files, with"
      " extensions .depth_intrinsics.txt, .color_intrinsics.txt, and"
      " .deformation.txt). Applies to the command line mode only, not to the GUI.");
  
  
  // Structure options.
  cmd_parser.NamedParameter(
      "--structure_depth_range",
      &bad_slam_config.structure_depth_range, /*required*/ false,
      bad_slam_config.structure_depth_range_help);
  
  bad_slam_config.structure_depth_only = cmd_parser.Flag(
      "--structure_depth_only",
      bad_slam_config.structure_depth_only_help);
  
  cmd_parser.NamedParameter(
      "--structure_depth_resolution",
      &bad_slam_config.structure_depth_resolution, /*required*/ false,
      bad_slam_config.structure_depth_resolution_help);
  
  bad_slam_config.structure_expensive_correction = cmd_parser.Flag(
      "--structure_expensive_correction",
      bad_slam_config.structure_expensive_correction_help);
  
  bad_slam_config.structure_one_shot_dynamic_calibration = cmd_parser.Flag(
      "--structure_one_shot_dynamic_calibration",
      bad_slam_config.structure_one_shot_dynamic_calibration_help);
  
  cmd_parser.NamedParameter(
      "--structure_depth_diff_threshold",
      &bad_slam_config.structure_depth_diff_threshold, /*required*/ false,
      bad_slam_config.structure_depth_diff_threshold_help);
  
  bad_slam_config.structure_infrared_auto_exposure = cmd_parser.Flag(
      "--structure_infrared_auto_exposure",
      bad_slam_config.structure_infrared_auto_exposure_help);
  
  
  // These sequential parameters must be specified last (in code).
  string dataset_folder_path;
  cmd_parser.SequentialParameter(
      &dataset_folder_path, "dataset_folder_path", false,
      "Path to the dataset in TUM RGB-D format.");
  
  string trajectory_path;
  cmd_parser.SequentialParameter(
      &trajectory_path, "gt_trajectory", false,
      "Filename of the ground truth trajectory in TUM RGB-D format (used for first"
      " frame only).");
  
  if (!cmd_parser.CheckParameters()) {
    return EXIT_FAILURE;
  }
  
  // Derive some parameters from program arguments.
  float depth_camera_scaling =
      1.0f / powf(2, bad_slam_config.pyramid_level_for_depth);
  float color_camera_scaling =
      1.0f / powf(2, bad_slam_config.pyramid_level_for_color);
  
  // Make it easier to use copy-pasted paths on Linux, which may be prefixed by
  // "file://".
  if (dataset_folder_path.size() > 7 &&
      dataset_folder_path.substr(0, 7) == "file://") {
    dataset_folder_path = dataset_folder_path.substr(7);
  }
  
  
  // ### Initialization ###
  
  // Handle CUDA kernel size auto-tuning.
  if (auto_tuning_iteration < 0) {
    boost::filesystem::path program_dir = boost::filesystem::path(argv[0]).parent_path();
    if (!CUDAAutoTuner::Instance().LoadParametersFile(
        (program_dir / "resources"  / "auto_tuning_result.txt").string().c_str())) {
      LOG(WARNING) << "No auto-tuning file found -> using default parameters."
                      " GPU performance is thus probably slightly worse than it"
                      " could be.";
    }
  } else {
    CUDAAutoTuner::Instance().SetTuningIteration(auto_tuning_iteration);
  }
  
  // Always create a QApplication, even if not using the GUI. It is required for
  // using libvis' Qt implementation for creating windowless OpenGL contexts.
  QSurfaceFormat surface_format;
  surface_format.setVersion(4, 4);
  surface_format.setProfile(QSurfaceFormat::CompatibilityProfile);
  surface_format.setSamples(4);
  surface_format.setAlphaBufferSize(0 /*8*/);
  QSurfaceFormat::setDefaultFormat(surface_format);
  QApplication qapp(argc, argv);
  QCoreApplication::setOrganizationName("ETH");
  QCoreApplication::setOrganizationDomain("eth3d.net");
  QCoreApplication::setApplicationName("BAD SLAM");
  
  // Load the dataset, respectively start the live input or show the GUI.
  RealSenseInputThread rs_input;
  StructureInputThread structure_input;
  K4AInputThread k4a_input;
  RGBDVideo<Vec3u8, u16> rgbd_video;
  int live_input = 0;
  
  if (dataset_folder_path.empty() || gui || gui_run) {
    if (!trajectory_path.empty()) {
      LOG(ERROR) << "Trajectory path given, but loading a ground truth trajectory is not supported yet: " << trajectory_path;
      return EXIT_FAILURE;
    }
    
    bool start_paused = false;
    if (!gui_run && !ShowSettingsWindow(&dataset_folder_path, &bad_slam_config, &start_paused)) {
      return EXIT_SUCCESS;
    }
    
    ShowMainWindow(
        qapp,
        start_paused,
        bad_slam_config,
        argv[0],
        dataset_folder_path,
        import_calibration_path,
        depth_scaling,
        splat_half_extent_in_pixels,
        show_current_frame_cloud,
        show_input_images,
        window_default_width,
        window_default_height);
    return EXIT_SUCCESS;
  } else if (dataset_folder_path == "live://realsense") {
    rs_input.Start(&rgbd_video, &depth_scaling);
    live_input = 1;
  } else if (dataset_folder_path == "live://structure") {
    structure_input.Start(&rgbd_video, &depth_scaling, bad_slam_config);
    live_input = 3;
  } else if (dataset_folder_path == "live://k4a") {
    k4a_input.Start(&rgbd_video, 
        &depth_scaling, 
        bad_slam_config.k4a_fps,
        bad_slam_config.k4a_resolution, 
        bad_slam_config.k4a_factor,
        bad_slam_config.k4a_use_depth,
        bad_slam_config.k4a_mode,
        bad_slam_config.k4a_exposure
    );
    live_input = 2;
  } else {
    if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(
                dataset_folder_path.c_str(),
                trajectory_path.empty() ? nullptr : trajectory_path.c_str(),
                &rgbd_video)) {
      LOG(ERROR) << "Could not read dataset.";
      return EXIT_FAILURE;
    }
    
    CHECK_EQ(rgbd_video.depth_frames_mutable()->size(),
             rgbd_video.color_frames_mutable()->size());
    LOG(INFO) << "Read dataset with " << rgbd_video.frame_count() << " frames";
  }
  
  // Initialize depth scale. This must be done after rs_input.Start() in the
  // live-input case since that may update depth_scaling.
  bad_slam_config.raw_to_float_depth = 1.0f / depth_scaling;
  if (bad_slam_config.max_depth * depth_scaling >= 1 << 15) {
    LOG(FATAL) << "max_depth * depth_scaling >= 1 << 15. This is too large"
                  " since it conflicts with the depth validity flag.";
  }
  
  // Get initial depth and color camera intrinsics. The generic camera type is
  // casted to the pinhole camera type; only pinhole cameras are supported.
  shared_ptr<Camera> initial_depth_camera(
      rgbd_video.depth_camera()->Scaled(depth_camera_scaling));
  CHECK_EQ(initial_depth_camera->type_int(),
           static_cast<int>(Camera::Type::kPinholeCamera4f));
  *rgbd_video.depth_camera_mutable() = initial_depth_camera;
  
  shared_ptr<Camera> initial_color_camera(
      rgbd_video.color_camera()->Scaled(color_camera_scaling));
  CHECK_EQ(initial_color_camera->type_int(),
           static_cast<int>(Camera::Type::kPinholeCamera4f));
  *rgbd_video.color_camera_mutable() = initial_color_camera;
  
  
  // Create the render window for visualization.
  shared_ptr<BadSlamRenderWindow> render_window;  // render window callbacks
  shared_ptr<RenderWindow> render_window_ui;  // render window UI object
  
  
  // If end_frame is non-zero, remove all frames which would extend beyond
  // this length.
  if (bad_slam_config.end_frame > 0 &&
      rgbd_video.color_frames_mutable()->size() > static_cast<usize>(bad_slam_config.end_frame)) {
    rgbd_video.color_frames_mutable()->resize(bad_slam_config.end_frame);
    rgbd_video.depth_frames_mutable()->resize(bad_slam_config.end_frame);
  }
  
  // Initialize image pre-loading thread.
  PreLoadThread pre_load_thread(&rgbd_video);
  
  // Allocate image displays.
  shared_ptr<ImageDisplay> image_display(new ImageDisplay());
  shared_ptr<ImageDisplay> depth_display(new ImageDisplay());
  
  // Initialize BAD SLAM.
  if (bad_slam_config.enable_loop_detection) {
    boost::filesystem::path program_dir =
        boost::filesystem::path(argv[0]).parent_path();
    bad_slam_config.loop_detection_vocabulary_path =
        (program_dir / "resources" / "brief_k10L6.voc").string();
    bad_slam_config.loop_detection_pattern_path =
        (program_dir / "resources" / "brief_pattern.yml").string();
    bad_slam_config.loop_detection_images_width = rgbd_video.color_camera()->width();
    bad_slam_config.loop_detection_images_height = rgbd_video.color_camera()->height();
  }
  
  unique_ptr<BadSlam> bad_slam(new BadSlam(bad_slam_config, &rgbd_video,
                                           render_window, nullptr));
  
  if (!import_calibration_path.empty()) {
    if (!LoadCalibration(&bad_slam->direct_ba(), import_calibration_path)) {
      return EXIT_FAILURE;
    }
  }
  
  std::ofstream save_timings_stream;
  if (!save_timings_path.empty()) {
    save_timings_stream.open(save_timings_path, std::ios::out);
    bad_slam->direct_ba().SetSaveTimings(&save_timings_stream);
  }
  
  // Print GPU memory usage after initialization. This can be used to see how
  // much free GPU memory remains for keyframes.
  // TODO: Some buffers are lazily allocated however, currently one should
  //       thus look at the memory use after the first BA iteration.
  LOG(INFO) << "Initial GPU memory usage:";
  PrintGPUMemoryUsage();
  
  
  // ### Main loop ###
  bool quit = false;
  bool program_aborted = false;
  for (usize frame_index = bad_slam_config.start_frame;
       (live_input || frame_index < rgbd_video.frame_count()) && !quit;
       ++ frame_index) {
    pre_load_thread.WaitUntilDone();
    if (live_input == 1) {
      rs_input.GetNextFrame();
    } else if (live_input == 2) {
      k4a_input.GetNextFrame();
    } else if (live_input == 3) {
      structure_input.GetNextFrame();
    }
    
    // Get the current RGB-D frame's RGB and depth images. This may wait for I/O
    // to complete in case it did not complete in the pre-loading thread yet.
    const Image<Vec3u8>* rgb_image =
        rgbd_video.color_frame_mutable(frame_index)->GetImage().get();
    const Image<u16>* depth_image =
        rgbd_video.depth_frame_mutable(frame_index)->GetImage().get();
    
    // Pre-load the next frame.
    if (frame_index < rgbd_video.frame_count() - 1) {
      pre_load_thread.PreLoad(frame_index + 1);
    }
    
    // Optionally, visualize the input images.
    if (show_input_images) {
      image_display->Update(*rgb_image, "image");
      depth_display->Update(*depth_image, "depth",
                            static_cast<u16>(0),
                            static_cast<u16>(depth_scaling *
                                             bad_slam_config.max_depth));
    }
    
    // Let BAD SLAM process the current RGB-D frame. This function does the
    // actual work.
    bad_slam->ProcessFrame(frame_index);
    
    // Update the 3D visualization.
    bad_slam->UpdateOdometryVisualization(frame_index, /*show_current_frame_cloud*/ false);
    
    // Get timings for processing this frame.
    float odometry_milliseconds;
    bad_slam->GetFrameTimings(&odometry_milliseconds);
    
    // Measure the frame time, and optionally restrict the frames per second.
    bad_slam->EndFrame();
    
    if (save_timings_stream.is_open()) {
      save_timings_stream << "odometry " << odometry_milliseconds << endl;
    }
    
    // TODO: Integrate this functionality into the new GUI
//     // Replace the keyframe poses with the ground truth poses (for
//     // convergence testing).
//     RGBDVideo<Vec3u8, u16> gt_rgbd_video;
//     if (!ReadTUMRGBDDatasetAssociatedAndCalibrated(
//             dataset_folder_path.c_str(), "groundtruth.txt", &gt_rgbd_video)) {
//       LOG(ERROR) << "Could not read dataset with ground truth trajectory.";
//     } else if (gt_rgbd_video.frame_count() != rgbd_video.frame_count()) {
//       LOG(ERROR) << "Cannot replace poses with poses from groundtruth.txt:"
//                     " frame count differs.";
//     } else {
//       for (usize i = 0; i < gt_rgbd_video.frame_count(); ++ i) {
//         rgbd_video.depth_frame_mutable(i)->SetFrameTGlobal(
//             gt_rgbd_video.depth_frame_mutable(i)->frame_T_global());
//         rgbd_video.color_frame_mutable(i)->SetFrameTGlobal(
//             gt_rgbd_video.color_frame_mutable(i)->frame_T_global());
//       }
//       
//       bad_slam->UpdateOdometryVisualization(
//           frame_index, /*show_current_frame_cloud*/ false);
//     }
//   } else if (key == 'z' || key == 'z') {
//     // Introduce some noise to the poses (for convergence testing).
//     constexpr float kNoiseAmount = 0.01f;
//     
//     srand(0);
//     for (const shared_ptr<Keyframe>& keyframe : bad_slam->direct_ba().keyframes()) {
//       SE3f new_frame_T_global =
//           SE3f::exp(kNoiseAmount * SE3f::Tangent::Random()) * keyframe->frame_T_global();
//       keyframe->set_frame_T_global(new_frame_T_global);
//     }
//     
//     bad_slam->UpdateOdometryVisualization(
//         frame_index, /*show_current_frame_cloud*/ false);
//   } else if (key == 'k' || key == 'K') {
//     // For --no_surfel_updates with --load_poses: manually create surfels for all keyframes.
//     for (const shared_ptr<Keyframe>& keyframe : bad_slam->direct_ba().keyframes()) {
//       bad_slam->direct_ba().CreateSurfelsForKeyframe(/*stream*/ 0, true, keyframe);
//     }
//     
//     bad_slam->UpdateOdometryVisualization(
//         frame_index, /*show_current_frame_cloud*/ false);
//   } else if (key == 'a') {
//     bad_slam->direct_ba().AssignColors(/*stream*/ 0);
//     bad_slam->UpdateOdometryVisualization(
//         frame_index, /*show_current_frame_cloud*/ false);
//     LOG(INFO) << "Colors assigned manually.";
//   }
    
    // Release memory.
    rgbd_video.depth_frame_mutable(frame_index)->ClearImageAndDerivedData();
    rgbd_video.color_frame_mutable(frame_index)->ClearImageAndDerivedData();
    
    if (render_window_ui && !render_window_ui->IsOpen()) {
      program_aborted = true;
      break;
    }
  }  // end of main loop
  
  
  if (!program_aborted) {
    // Run final BA iterations?
    if (final_ba_iterations > 0) {
      // First, run BA in a windowed way to avoid allocating an extreme number
      // of surfels in the case it has not been run before
      constexpr int kWindowSize = 16;
      for (u32 window_start_index = 0;
          window_start_index < bad_slam->direct_ba().keyframes().size();
          window_start_index += kWindowSize / 2) {
        // TODO: Currently this runs BA twice (since the window is only advanced by
        //       half of its size) since the surfel radii are only updated at the
        //       end (and they have to be updated with the new
        //       sparse_surfel_cell_size). Make a separate function to update
        //       this?
        bad_slam->RunBundleAdjustment(
            /*frame_index*/ rgbd_video.frame_count() - 1,
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
      }
      
      // Then, run it normally for the given number of iterations
      for (int iteration = 0; iteration < final_ba_iterations; ++ iteration) {
        bad_slam->RunBundleAdjustment(
            /*frame_index*/ rgbd_video.frame_count() - 1,
            bad_slam_config.optimize_intrinsics,
            bad_slam_config.optimize_intrinsics,
            /*optimize_poses*/ true,
            /*optimize_geometry*/ true,
            /*min_iterations*/ 2,
            /*max_iterations*/ 10,
            0,
            bad_slam->direct_ba().keyframes().size() - 1,
            /*increase_ba_iteration_count*/ true,
            nullptr, nullptr, 0, nullptr);
        bad_slam->UpdateOdometryVisualization(
            /*frame_index*/ rgbd_video.frame_count() - 1,
            /*show_current_frame_cloud*/ false);
      }
    }
    
    // Save the resulting point cloud?
    if (!export_point_cloud_path.empty()) {
      SavePointCloudAsPLY(/*stream*/ 0, bad_slam->direct_ba(), export_point_cloud_path);
    }
    
    // Save the resulting poses?
    if (!export_poses_path.empty()) {
      SavePoses(rgbd_video, bad_slam_config.use_geometric_residuals,
                bad_slam_config.start_frame, export_poses_path);
    }
    
    // Save the resulting calibration?
    if (!export_calibration_path.empty()) {
      SaveCalibration(/*stream*/ 0, bad_slam->direct_ba(), export_calibration_path);
    }
    
    // Save the final timings?
    if (!export_final_timings_path.empty()) {
      std::ofstream timings_file(export_final_timings_path, std::ios::out);
      timings_file << Timing::print(kSortByTotal);
      timings_file.close();
    }
    
    // Create and save a reconstruction?
    if (!export_reconstruction_path.empty()) {
      // TODO: Set the minimum observation count to 1 for the reconstruction?
      // TODO: Set do_surfel_updates to true?
      
      // Upscale the cfactor buffer to full resolution (since it uses the same
      // sparsification as the surfels).
      // TODO: Exclude pixels with zero observations from the interpolation
      CUDABufferPtr<float> cfactor_buffer = bad_slam->direct_ba().cfactor_buffer();
      CUDABufferPtr<float> scaled_cfactor_buffer(new CUDABuffer<float>(
          (bad_slam->direct_ba().depth_camera().height() - 1) / reconstruction_sparse_surfel_cell_size + 1,
          (bad_slam->direct_ba().depth_camera().width() - 1) / reconstruction_sparse_surfel_cell_size + 1));
      UpscaleBufferBilinearly(/*stream*/ 0, *cfactor_buffer, scaled_cfactor_buffer.get());
      bad_slam->direct_ba().SetCFactorBuffer(scaled_cfactor_buffer);
      
      // Run geometry-only BA without sparsification and without the descriptor residuals.
      // Use a sliding window for activating the keyframes to avoid allocating
      // a large number of surfels as an intermediate step.
      bool old_use_photometric_residuals = bad_slam_config.use_photometric_residuals;
      bad_slam->direct_ba().SetUseDescriptorResiduals(false);
      
      int old_sparse_surfel_cell_size = bad_slam->direct_ba().sparse_surfel_cell_size();
      bad_slam->direct_ba().SetSparsificationSideFactor(reconstruction_sparse_surfel_cell_size);
      
      constexpr int kWindowSize = 16;
      for (u32 window_start_index = 0;
          window_start_index < bad_slam->direct_ba().keyframes().size();
          window_start_index += kWindowSize / 2) {
        // TODO: Currently this runs BA twice (since the window is only advanced by
        //       half of its size) since the surfel radii are only updated at the
        //       end (and they have to be updated with the new
        //       sparse_surfel_cell_size). Make a separate function to update
        //       this?
        bad_slam->RunBundleAdjustment(
            /*frame_index*/ rgbd_video.frame_count() - 1,
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
      }
      
      bad_slam->direct_ba().AssignColors(/*stream*/ 0);
      
      bad_slam->direct_ba().SetSparsificationSideFactor(old_sparse_surfel_cell_size);
      bad_slam->direct_ba().SetUseDescriptorResiduals(old_use_photometric_residuals);
      
      // Save the result.
      SavePointCloudAsPLY(
          /*stream*/ 0,
          bad_slam->direct_ba(),
          export_reconstruction_path);
    }
  }
  
  pre_load_thread.RequestExitAndWaitForIt();
  
  bad_slam.reset();
  
  if (CUDAAutoTuner::Instance().tuning_active()) {
    ostringstream tuning_file_path;
    tuning_file_path << "auto_tuning_iteration_" << auto_tuning_iteration << ".txt";
    if (!CUDAAutoTuner::Instance().SaveTuningFile(tuning_file_path.str().c_str())) {
      LOG(ERROR) << "Could not save auto-tuning result file: "
                 << tuning_file_path.str();
    }
  }
  
  return EXIT_SUCCESS;
}
