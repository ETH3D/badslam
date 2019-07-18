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

#include <libvis/camera.h>
#include <libvis/libvis.h>

namespace vis {

/// Bundles BAD SLAM's config options together. The main purpose is to avoid
/// having very long function parameter lists when these options are passed
/// along.
struct BadSlamConfig {
  // --- Dataset playback parameters ---
  
  static constexpr const char* raw_to_float_depth_help =
      "Factor that converts from raw depth values (as read from the dataset, e.g.,"
      " u16-valued PNG files) to depth in meters, as follows:\n"
      "depth_in_meters = raw_to_float_depth * raw_depth";
  float raw_to_float_depth = 1. / 5000.;
  
  static constexpr const char* start_frame_help =
      "First frame of the video to process.";
  int start_frame = 0;
  
  static constexpr const char* end_frame_help =
      "Last frame of the video to process. This setting can be safely set to"
      " be larger than the last frame index in the dataset to process all of its"
      " frames.";
  int end_frame = numeric_limits<int>::max();
  
  static constexpr const char* target_frame_rate_help =
      "Specifies a frame rate which the program tries to keep by skipping BA"
      " iterations. If set to 0, the program runs in offline mode and never skips"
      " BA iterations.\n"
      "TODO: Somewhat unclear; applies to non-realtime mode only";
  float target_frame_rate = 0;
  
  static constexpr const char* fps_restriction_help =
      "Restrict the frames per second to at most the given number:"
      " If a frame is processed in less time, sleeps until the frame time has"
      " passed. This is useful to simulate live input when playing back a dataset."
      " If set to 0, no FPS restriction is used.";
  int fps_restriction = 30;
  
  static constexpr const char* pyramid_level_for_depth_help =
      "Specify the scale-space pyramid level to use for depth images. 0 uses the"
      " original sized images, 1 uses half the original resolution, etc.\n"
      "TODO: Does this still work?";
  int pyramid_level_for_depth = 0;
  
  static constexpr const char* pyramid_level_for_color_help =
      "Specify the scale-space pyramid level to use for color images. 0 uses the"
      " original sized images, 1 uses half the original resolution, etc.\n"
      "TODO: Setting this parameter to 1 is not supported in pairwise frame"
      " tracking if also using pyramid level 0 for pairwise tracking.\n"
      "TODO: Does this still work?";
  int pyramid_level_for_color = 0;
  
  
  // --- Depth preprocessing parameters ---
  
  static constexpr const char* max_depth_help =
      "Maximum input depth in meters. Larger depth values will be ignored. The"
      " default is appropriate for Asus Xtion / first generation Kinect, for"
      " example, but this should be increased for depth cameras which can"
      " accurately estimate larger depths as well.";
  float max_depth = 3.0f;
  
  static constexpr const char* baseline_fx_help =
      "The baseline (in meters) times the focal length (in pixels) of the"
      " stereo system which was used to estimate the input depth images. Used"
      " to estimate the depth uncertainty.";
  float baseline_fx = 40.f;
  
  static constexpr const char* median_filter_and_densify_iterations_help =
      "Number of iterations of a variant of median filtering performed for"
      " denoising and densification. Only required for noisy, incomplete input."
      " Should usually be left at its default of zero. The implementation of this"
      " is currently running on the CPU, thus it is probably slow.";
  int median_filter_and_densify_iterations = 0;
  
  static constexpr const char* bilateral_filter_sigma_xy_help =
      "Value of the sigma_xy parameter for depth bilateral filtering, in pixels.";
  float bilateral_filter_sigma_xy = 1.5f;
  
  static constexpr const char* bilateral_filter_radius_factor_help =
      "Factor on bilateral_filter_sigma_xy to define the kernel radius for depth"
      " bilateral filtering.";
  float bilateral_filter_radius_factor = 2.0f;
  
  static constexpr const char* bilateral_filter_sigma_inv_depth_help =
      "Sigma on the inverse depth for depth bilateral filtering.";
  float bilateral_filter_sigma_inv_depth = 0.005f;
  
  
  // --- Surfel reconstruction parameters ---
  
  static constexpr const char* max_surfel_count_help =
      "Maximum number of surfels (given in advance for being able to allocate"
      " surfel buffers only once, avoiding re-allocation).";
  int max_surfel_count = 25 * 1000 * 1000;  // 25 million.
  
  static constexpr const char* sparse_surfel_cell_size_help =
      "Surfel sparsification grid cell size. A cell size of 1 leads to fully"
      " dense surfel creation, 2 creates surfels for one quarter of the pixels"
      " only, etc.";
  int sparse_surfel_cell_size = 4;
  
  static constexpr const char* surfel_merge_dist_factor_help =
      "Factor on the minimum surfel radius of a pair of surfels to obtain the"
      " distance threshold for merging them.";
  float surfel_merge_dist_factor = 0.8f;
  
  static constexpr const char* min_observation_count_while_bootstrapping_1_help =
      "Minimum number of observations required to create / keep a surfel."
      " This value applies during early bootstrapping (with less than 5 keyframes).";
  int min_observation_count_while_bootstrapping_1 = 1;
  
  static constexpr const char* min_observation_count_while_bootstrapping_2_help =
      "Minimum number of observations required to create / keep a surfel."
      " This value applies during late bootstrapping (with less than 10 keyframes).";
  int min_observation_count_while_bootstrapping_2 = 2;
  
  static constexpr const char* min_observation_count_help =
      "Minimum number of observations required to create / keep a surfel."
      " This value applies after bootstrapping (with at least 10 keyframes).\n"
      "TODO: Change this to 2 for better reliability with fast motions after"
      " confirming on the benchmark that it does not hurt accuracy too much?";
  int min_observation_count = 3;
  
  
  // --- Odometry parameters  ---
  
  static constexpr const char* num_scales_help =
      "Number of pyramid levels in the multi-resolution scheme for pose"
      " estimation in odometry. The best value for this may depend on the camera"
      " resolution, both pixel resolution and angular resolution.";
  int num_scales = 5;
  
  static constexpr const char* use_motion_model_help =
      "Whether to use a constant motion model to predict the next frame's pose.";
  bool use_motion_model = true;
  
  
  // --- Bundle adjustment parameters ---
  
  static constexpr const char* keyframe_interval_help =
      "Determines the interval in frames for creating keyframes. By setting this"
      " to 1, a keyframe is created for every frame.";
  int keyframe_interval = 10;
  
  static constexpr const char* max_num_ba_iterations_per_keyframe_help =
      "The maximum number of bundle adjustment iterations performed after"
      " creating a keyframe.\n"
      "TODO: For real-time, it might be better to just do as many iterations as possible.";
  int max_num_ba_iterations_per_keyframe = 10;
  
  static constexpr const char* disable_deactivation_help =
      "Disables deactivation of surfels and keyframes during bundle adjustment."
      " This was a conecpt for more local bundle adjustment that was tested during"
      " development but discarded since it did not seem to help on the benchmark"
      " datasets.\n"
      "TODO: Probably remove this entirely and consider other (standard) forms"
      " of local bundle adjustment to improve performance on large datasets.";
  bool disable_deactivation = true;
  
  static constexpr const char* use_geometric_residuals_help =
      "Whether to use residuals betwwen depth measurements and surfel positions.";
  bool use_geometric_residuals = true;
  
  static constexpr const char* use_photometric_residuals_help =
      "Whether to use photometric residuals (based on descriptors computed from"
      " visible-light images).";
  bool use_photometric_residuals = true;
  
  static constexpr const char* optimize_intrinsics_help =
      "Enables or disables intrinsics optimization.";
  bool optimize_intrinsics = false;
  
  static constexpr const char* intrinsics_optimization_interval_help =
      "Specifies the frequency of intrinsics and depth deformation optimization"
      " (\"every Xth time that bundle adjustment is run\"). Set this to 1 to"
      " optimize intrinsics and depth deformation during every bundle adjustment"
      " run.";
  int intrinsics_optimization_interval = 10;
  
  static constexpr const char* do_surfel_updates_help =
      "If false, disables surfel updates (creation, merging) during BA. Only new"
      " keyframes will generate new surfels. Outlier surfels are still deleted.";
  bool do_surfel_updates = true;
  
  static constexpr const char* parallel_ba_help =
      "Whether to run bundle adjustment in parallel to odometry. If set to false,"
      " these components run alternatingly.";
  bool parallel_ba = true;
  
  static constexpr const char* use_pcg_help =
      "Whether to use a preconditioned conjugate gradient (PCG) based solver for"
      " the Gauss-Newton update equation for bundlle adjustment. If set to false,"
      " the default alternating optimization scheme is used instead.";
  bool use_pcg = false;
  
  static constexpr const char* estimate_poses_help =
      "If set to false, the given frame poses will be used instead of estimating"
      " their poses. This disables odometry and bundle adjustment. This is intended"
      " for tests where one wants to load the ground truth (or other) poses and"
      " then, for example, run bundle adjustment or 3D reconstruction.";
  bool estimate_poses = true;
  
  
  // --- Memory parameters ---
  
  static constexpr const char* min_free_gpu_memory_mb_help =
      "Minimum GPU memory amount in megabytes that shall remain free. Selected"
      " keyframes will be deleted if too much memory gets allocated.";
  int min_free_gpu_memory_mb = 250;
  
  
  // --- Loop detection parameters ---
  
  static constexpr const char* enable_loop_detection_help =
      "Enable or disable loop detection. If enable_loop_detection == false, the"
      " remaining loop detection options do not need to be set.";
  bool enable_loop_detection = true;
  
  static constexpr const char* parallel_loop_detection_help =
      "Whether to run loop detection (not loop closure, though) in parallel to"
      " the other components.";
  bool parallel_loop_detection = true;
  
  static constexpr const char* loop_detection_vocabulary_path_help =
      "Path to the .voc file for loop detection.";
  string loop_detection_vocabulary_path = "";
  
  static constexpr const char* loop_detection_pattern_path_help =
      "Path to the pattern .yml file for loop detection.";
  string loop_detection_pattern_path = "";
  
  static constexpr const char* loop_detection_image_frequency_help =
      "Frequency of images used for loop detection. If set to zero, this is"
      " computed as: fps_restriction / (1.f * keyframe_interval). However, this"
      " assumes that fps_restriction equals the FPS that the dataset was actually"
      " recorded with. If that is not the case, the image frequency should be set"
      " manually here.";
  float loop_detection_image_frequency = 0;
  
  static constexpr const char* loop_detection_images_width_help =
      "Width of the images used for loop detection (i.e., the color images).";
  int loop_detection_images_width = -1;
  
  static constexpr const char* loop_detection_images_height_help =
      "Height of the images used for loop detection (i.e., the color images).";
  int loop_detection_images_height = -1;
  
  
  // --- Structure parameters ---
  
  static constexpr const char* structure_depth_range_help =
      "Depth range setting. Supported values:"
      " VeryShort (0.35m to 0.92m),"
      " Short (0.41m to 1.36m),"
      " Medium (0.52m to 5.23m),"
      " Long (0.58m to 8.0m),"
      " VeryLong (0.58m to 10.0m),"
      " Hybrid (0.35m to 10.0m),"
      " BodyScanning,"
      " Default.";
  string structure_depth_range = "Default";
  
  static constexpr const char* structure_depth_only_help =
      "If this is set to true, only depth, but no visible images will be streamed."
      " The depth images will not be reprojected, only undistorted."
      " If this is set to false, both depth and visible images will be streamed,"
      " and the depth images will be reprojected to the visible image viewpoints.";
  bool structure_depth_only = false;
  
  static constexpr const char* structure_depth_resolution_help =
      "Resolution of the depth images. Attention, if both depth and visible images"
      " are streamed, then depth images will be reprojected to the visible images,"
      " such that high initial depth resolution will be lost. Use --structure_depth_only"
      " to disable depth reprojection (and use only depth images). Supported values:"
      " 320x240, 640x480, 1280x960.";
  string structure_depth_resolution = "640x480";
  
  static constexpr const char* structure_expensive_correction_help =
      "Whether to use the 'expensive correction' for depth images from the"
      "Structure SDK. Note that BAD SLAM also applies some of its own depth filtering.";
  bool structure_expensive_correction = false;
  
  static constexpr const char* structure_one_shot_dynamic_calibration_help =
      "Whether to use the one-shot dynamic calibration from the Structure SDK.";
  bool structure_one_shot_dynamic_calibration = false;
  
  static constexpr const char* structure_depth_diff_threshold_help =
      "Maximum difference between neighboring depth pixels to be considered on"
      " the same surface for depth reprojection.";
  float structure_depth_diff_threshold = 0.05;
  
  static constexpr const char* structure_infrared_auto_exposure_help =
      "Whether to activate auto-exposure for infrared.";
  bool structure_infrared_auto_exposure = true;
  
  static constexpr const char* structure_visible_exposure_time_help =
      "Fixed exposure time for the visible camera (in seconds).";
  float structure_visible_exposure_time = 0.016f;
  
  
  // --- K4A parameters ---
  
  static constexpr const char* k4a_mode_help = 
      "Operating modes for Azure Kinect: nfov,nfov2x2,wfov,wfov2x2 "
      "default: (narrow field of view) with resolution 640x576, fov 75°x65° and range 0.5m - 3.86m "
      "nfov2x2 was resolution 320x288, fov 75°x65° and range 0.5m - 5.46 "
      "wfov (wide field of view) has resolution 1024x1024,fov 120°x120° and range 0.25 - 2.21m "
      "wfov2x2  has resolution 512x512,fov 120°x120° and range 0.25 - 2.88m "
      "full specs at http://aka.ms/kinectdocs";
  string k4a_mode = "nfov";

  static constexpr const char* k4a_fps_help =
      "Azure Kinect frame per seconds: 30, 15, 5"
      " default: 30, not every combination of mode and resolution supports every fps";
  int k4a_fps = 30;

  static constexpr const char* k4a_resolution_help = "Azure kinect resolution: default 720, can be 1080,1440,2160,3082,1536";
  int k4a_resolution = 720;

  static constexpr const char* k4a_factor_help = "Downscaling factor for Azure kinect images";
  int k4a_factor = 1;
  
  static constexpr const char* k4a_use_depth_help = "When using this mode only the depth image plus reflectivity is used, without rgb";
  int k4a_use_depth = 0;
  
  static constexpr const char* k4a_exposure_help = "Exposure for the rgb camera of Azure Kinect, default is 8000ms when 0 it's auto";
  int k4a_exposure = 8000;
  
  
  inline float GetLoopDetectionImageFrequency() const {
    return (loop_detection_image_frequency != 0) ?
               loop_detection_image_frequency :
               (fps_restriction / (1.f * keyframe_interval));
  }
  
  bool Save(FILE* file) const;
  bool Load(FILE* file);
};

}
