## Command line arguments ##

The program can be started without command line arguments to show a settings
window that allows to select the dataset or live input.
Alternatively, the path to the dataset can be given as follows:

```
badslam <dataset_path>
```

This will run the program without visualization. To enforce visualization, pass
the `--gui` flag (to start showing the settings window) or the `--gui_run` flag
(to start running immediately).

The dataset path may be prefixed by `file://`. This makes it easier to copy/paste paths
in this format. To use RealSense live input, specify `live://realsense` as
dataset path. This is easiest with the corresponding button in the settings window
of the GUI (which is only visible if the program has been compiled with RealSense support), which will also
suggest suitable options:
`--no_photometric_residuals --bilateral_filter_sigma_inv_depth 0.01 --max_depth 4 --restrict_fps_to 0`.
To use K4A live input, specify `live://k4a` as dataset path.

A complete list of optional program arguments follows, grouped by category:

#### Dataset playback ####

* `--depth_scaling` (default 5000): Specifies the scaling of depth values in the input (e.g., if reading 16-bit PNG files): input_depth = depth_scaling * depth_in_meters. The default of 5000 applies to the TUM RGB-D and ETH3D SLAM benchmark datasets.
* `--target_frame_rate` (default 0): If --sequential_ba is used, specifies a frame rate which the program tries to keep by skipping BA iterations. If --sequential_ba is used and this is set to 0, the program runs in offline mode and never skips BA iterations.
* `--restrict_fps_to` (default 30): Restrict the frames per second to at most the given number: If a frame is processed in less time, sleeps until the frame time has passed. This
  is useful to simulate live input when playing back a dataset. If set to 0, no FPS restriction is used. For live input, set this to 0 to avoid unnecessary waiting.
* `--start_frame` (default 0): First frame of the dataset to process.
* `--end_frame` (default numeric_limits&lt;int&gt;::max()): Makes dataset playback stop after end_frame.
* `--pyramid_level_for_depth` (default 0): Specify the scale-space pyramid level to use for depth images. 0 uses the original sized images, 1 uses half the original resolution, etc. 
  TODO: This setting is likely broken, test and fix if necessary.
* `--pyramid_level_for_color` (default 0): Specify the scale-space pyramid level to use for color images. 0 uses the original sized images, 1 uses half the original resolution, etc.
  TODO: This setting is likely broken, test and fix if necessary.

#### Odometry ####

* `--num_scales` (default 5): Number of pyramid levels in the multi-resolution scheme for frame-to-frame pose estimation.
* `--no_motion_model`: Disables the constant motion model that is used to predict the next frame's pose.

#### Bundle adjustment ####

* `--keyframe_interval` (default 10): Determines the interval in frames for creating keyframes. By setting this to 1, a keyframe is created for every frame.
* `--max_num_ba_iterations_per_keyframe` (default 10): The maximum number of bundle adjustment iterations performed after creating a keyframe. Set this to 0 to disable bundle adjustment.
* `--use_deactivation`: Enables deactivation of surfels and keyframes during bundle adjustment. This attempts to deactivate nearly-converged keyframes, and surfels which are only
  observed by such keyframes. However, no benefit on the result quality in a real-time setting has been proven. Does not work together with the PCG-based solver, and may likely be broken since it has not been tested recently.
* `--no_geometric_residuals`: Disables the use of geometric residuals (comparing depth images and surfel positions).
* `--no_photometric_residuals`: Disables the use of photometric residuals (comparing visible-light images and surfel descriptors).
* `--optimize_intrinsics`: Perform self-calibration of camera intrinsics and depth deformation during operation.
* `--intrinsics_optimization_interval` (default 10): Specifies the frequency of intrinsics optimization ("every Xth time that bundle adjustment is run").
  Set this to 1 to optimize intrinsics (and depth deformation) during every bundle adjustment run.
* `--final_ba_iterations` (default 0): Specifies a number of BA iterations to perform after dataset playback finishes (applies to command line mode only, not to the GUI).
* `--no_surfel_updates`: Disables surfel updates (creation, merging) during BA. Only new keyframes will generate new surfels. Outlier surfels are still deleted.
* `--sequential_ba`: Performs bundle adjustment sequentially instead of in parallel to odometry.
* `--use_pcg`: Use a PCG (preconditioned conjugate gradient) solver on the Gauss-Newton update equation, instead of the default alternating optimization.

#### Memory ####

* `--min_free_gpu_memory_mb` (default 250): Minimum GPU memory amount in megabytes that shall remain free. Selected keyframes will be deleted if too much memory gets allocated.

#### Surfel reconstruction ####

* `--max_surfel_count` (default 25 million): Maximum number of surfels. Affects the GPU memory requirements. If you feel that too much GPU memory is used on program startup already (as can be seen in the bottom-right of the GUI), consider lowering this value. With default settings, 25 million surfels should hardly ever be created with normal use. Only if using the 'densify surfels' functionality from the GUI, if using --export_reconstruction, or if setting the --sparsification parameter to 1, the surfel count is likely to approach the default maximum value.
* `--sparsification` (default 4): Surfel sparsification grid cell size. A cell size of 1 leads to fully dense surfel creation, 2 creates surfels for one quarter of the pixels only, etc.
* `--surfel_merge_dist_factor` (default 0.8): Factor on the minimum surfel radius of a pair of surfels to obtain the distance threshold for merging them.
* `--min_observation_count_while_bootstrapping_1` (default 1): Minimum number of observations required to create / keep a surfel. This value applies during early bootstrapping (with less than 5 keyframes).
* `--min_observation_count_while_bootstrapping_2` (default 2): Minimum number of observations required to create / keep a surfel. This value applies during early bootstrapping (with less than 10 keyframes).
* `--min_observation_count` (default 3): Minimum number of observations required to create / keep a surfel. This value applies after bootstrapping (with at least 10 keyframes).
* `--reconstruction_sparsification` (default 1): Sparse surfel cell size for the final reconstruction that is done for --export_reconstruction. See --sparsification.

#### Loop closure ####

* `--no_loop_detection`: Disables loop closure search.
* `--sequential_loop_detection`: Runs loop detection sequentially instead of in parallel.
* `--loop_detection_image_frequency`: Frequency of images used for loop detection. This should be set to the frequency of keyframes in the video, considering the original frame rate of the dataset (not the frame rate it is played back with). If set to zero, this is computed as: fps_restriction / (1.f * keyframe_interval). However, this assumes that fps_restriction equals the FPS that the dataset was actually recorded with. If that is not the case, the image frequency should be set manually with this command-line parameter.

#### Depth preprocessing ####

* `--max_depth` (default 3): Maximum input depth in meters. Larger depth values will be discarded.
* `--baseline_fx` (default 40): The baseline (in meters) times the focal length (in pixels) of the stereo system which was used to estimate the input depth images. Used to estimate the depth uncertainty.
* `--median_filter_and_densify_iterations` (default 0): Number of iterations of a variant of median filtering performed for denoising and densification. This will be slow, since it is currently implemented on the CPU.
* `--bilateral_filter_sigma_xy` (default 1.5): sigma_xy for depth bilateral filtering, in pixels.
* `--bilateral_filter_radius_factor` (default 2): Factor on bilateral_filter_sigma_xy to define the kernel radius for depth bilateral filtering.
* `--bilateral_filter_sigma_inv_depth` (default 0.005): sigma on the inverse depth for depth bilateral filtering.

#### Visualization ####

* `--gui`: Show the GUI (starting with the settings window).
* `--gui_run`: Show the GUI (and start running immediately).
* `--show_input_images`: Displays the input images.
* `--splat_half_extent_in_pixels`: Half splat quad extent in pixels.
* `--window_default_width`: Default width of the 3D visualization window.
* `--window_default_height`: Default height of the 3D visualization window.
* `--show_current_frame_cloud`: Visualize the point cloud of the current frame.

#### Auto-tuning ####

* `--auto_tuning_iteration` (default -1): Used by the auto-tuning script to signal that a tuning iteration is in progress. There is no need to use this option manually.

#### Output paths ####

* `--export_point_cloud` (default ""): Save the final surfel point cloud to the given path (as a PLY file). Applies to the command line mode only, not to the GUI.
* `--export_reconstruction` (default ""): Creates a reconstruction at the end (without, or with less sparsification) and saves it as a point cloud to the given path (as a PLY file). See the --reconstruction_sparsification option. Applies to the command line mode only, not to the GUI.
* `--export_calibration` (default ""): Save the final calibration to the given base path (as three files, with extensions .depth_intrinsics.txt, .color_intrinsics.txt, and .deformation.txt). Applies to the command line mode only, not to the GUI.
* `--export_final_timings` (default ""): Save the final aggregated timing statistics to the given text file. Applies to the command line mode only, not to the GUI.
* `--save_timings` (default ""): Save the detailed BA timings (for every time BA is run) to the given file. Applies to the command line mode only, not to the GUI.
* `--export_poses` (default ""): Save the final poses to the given text file in TUM RGB-D format. Applies to the command line mode only, not to the GUI.

#### Input paths ####

* `--import_calibration` (default ""): Load the calibration from the given base path (as three files, with extensions .depth_intrinsics.txt, .color_intrinsics.txt, and .deformation.txt).

#### Structure Core live input parameters ####

* `--structure_depth_range` (default "Default"): Depth range setting. Supported values:
  * "VeryShort" (0.35m to 0.92m)
  * "Short" (0.41m to 1.36m)
  * "Medium" (0.52m to 5.23m)
  * "Long" (0.58m to 8.0m)
  * "VeryLong" (0.58m to 10.0m)
  * "Hybrid" (0.35m to 10.0m)
  * "BodyScanning"
  * "Default"
* `--structure_depth_only`: If this flag is set, only depth, but no visible images will be streamed. The depth images will not be reprojected, only undistorted. If this flag is not set, both depth and visible images will be streamed, and the depth images will be reprojected to the visible image viewpoints.
* `--structure_depth_resolution` (default 640x480): Resolution of the depth images. Attention, if both depth and visible images are streamed, then depth images will be reprojected to the visible images, such that high initial depth resolution will be lost. Use --structure_depth_only to disable depth reprojection (and use only depth images). Supported values:
  * 320x240
  * 640x480
  * 1280x960
* `--structure_expensive_correction`: Whether to use the 'expensive correction' for depth images from the Structure SDK. Note that BAD SLAM also applies some of its own depth filtering.
* `--structure_one_shot_dynamic_calibration`: Whether to use the one-shot dynamic calibration from the Structure SDK that can compensate for static deformations of the sensor. This will delay the startup by 10 seconds.
* `--structure_depth_diff_threshold` (default 0.05): Maximum difference between neighboring depth pixels (in meters) to be considered on the same surface for depth reprojection.
* `--no_structure_infrared_auto_exposure`: Disables the auto-exposure for the infrared cameras.
* `--structure_visible_exposure_time` (default 0.016): Sets the fixed exposure time for the visible camera (in seconds).

#### Sequential parameters ####

* The path to the dataset is given as first sequential parameter.
* The filename of the ground truth trajectory in TUM RGB-D format may optionally be given as second sequential parameter. It is used to set the first frame's pose only, such that it aligns with the ground truth.
  Please notice that only the filename must be given here, not the full path, and the file is assumed to be within the dataset directory.
