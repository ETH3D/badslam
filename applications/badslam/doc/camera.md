## Camera requirements ##

Consider the following three properties:

* **Camera shutter**: Camera shutters are either global (all pixels of an image are
  recorded at the same time) or rolling (pixels are recored row by row or column
  by column). BAD SLAM assumes a global shutter camera. This applies both to the
  color and the depth camera.
* **Camera synchronization**: BAD SLAM assumes that the RGB and depth images are
  recorded at exactly the same points in time.
* **Camera calibration**: The camera calibration should be accurate. A good way to
  achieve this is to use a dense non-parametric calibration approach, and making
  sure to use lots of data for calibration.
  Alternatively, doing a parametric calibration with a model that fits
  reasonably well might be acceptable, but is likely inferior since the model is
  unlikely to fit exactly. Using a default intrinsic calibration for a camera
  type, as it is sometimes being done, should be avoided.

If you cannot get suitable high-quality RGB-D data, consider using another SLAM system instead,
for example [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). If you
choose to use BAD SLAM, consider using its intrinsics optimization step
to perform depth deformation calibration (as described below in the section on
using different depth cameras).

An additional assumption to be mentioned here for completeness is that
the color and depth images are taken from the same viewpoint and have the same
dimensions. This is usually easy to achieve by reprojecting the depth image
into the color image.


### Camera compatibility ###

This table shows which of the listed properties are fulfilled by
consumer-level RGB-D cameras.

| Camera name   | Global shutter (depth) | Global shutter (color) | Synchronized | Live support in BAD SLAM |
| ------------- | ---------------------- | ---------------------- | ------------ | ------------------------ |
| Asus Xtion Live Pro |  |  |  |  |
| Intel RealSense D415 |  |  | yes after firmware update | yes (1) |
| Intel RealSense D435 | yes |  |  | yes |
| Microsoft Kinect v1 |  |  |  |  |
| Microsoft Kinect v2 | ? |  | ? |  |
| Microsoft Kinect for Azure | yes |  | yes | yes |
| Occipital Structure Core | yes | yes | yes | yes (2) |

(1): Untested.

(2): Tested with the color version of the Structure Core only. For the
monochrome large-FOV version, it might be helpful to slightly adjust the
image cropping.

The [ETH3D SLAM benchmark](https://www.eth3d.net/slam_overview) is a
suitable dataset if you are fine with working with pre-recorded datasets.


## Using a different depth camera ##

If you use the program on different datasets than the ETH3D SLAM benchmark datasets,
please keep in mind that some parameters may need to be adjusted to account for using
a different depth camera. Here are some parameters to consider (see also the [complete list of program arguments](https://github.com/ETH3D/badslam/blob/master/applications/badslam/doc/command_line.md)).

* `--max_depth`: The maximum depth value used, in meters. Defaults to 3 meters.
  Larger depth values will simply be discarded. This may be used to discard large
  depth values which contain too much noise or biases.
* `--num_scales`: Specifies the number of pyramid levels which are used in the multi-resolution scheme for frame-to-frame pose estimation.
  This must be large enough to yield a large convergence region. If it is too large, it may however increase the risk of convergence to a wrong optimum,
  since the optimization may be mislead on pyramid levels for which the images are very small. In each pyramid level, the image size is halved.
  The default parameter value is 5, which is suitable for approximately 640x480-sized images with a relatively small field of view.
  For example, if you use images with twice this resolution and the same field of view, it is likely useful to try to increase this parameter by one.
* `--bilateral_filter_sigma_inv_depth`: This controls the strength of the smoothing applied to the input depth maps.
  It may be required to increase this for noisy depth cameras to get good normal vector estimates. Good normal vectors are important, since
  the normals determine the optimization direction for surfel positions.
  To qualitatively check whether the normals are fine, click some keyframes in the GUI and inspect the visualizations of their estimated normals.
  If there is significant noise in surfaces that should be flat, increase `bilateral_filter_sigma_inv_depth`.
  Note that the visualization may also show artefacts coming from the depth sensors however.
  For example, normal maps for images taken with the first-generation Kinect or Asus Xtion will contain some vertical stripes since the camera introduces some depth discontinuities there.
  These kinds of problems should not be addressed with higher smoothing (since this would lead to oversmoothing), but instead by calibrating for the depth distortion,
  for example with the self-calibration that is included in BAD SLAM -- see below.
  Please note that this calibration will currently not affect the visualization however.
* `--baseline_fx`: This specifies the baseline of the stereo system used to create the depth maps (in meters), multiplied by the depth camera's focal length (in pixels).
  This is used to estimate the depth uncertainty from stereo matching, as stated in the paper.
  It is used together with the expected stereo matching uncertainty of 0.1 pixels, which is
  currently hard-coded as `kDepthUncertaintyEmpiricalFactor` in cost_function.cuh.
  You may need to adjust these parameters to your depth camera, or it may even be appropriate to implement a completely different
  estimate for depth uncertainty, for example if using time-of-flight depth cameras.
* `--median_filter_and_densify_iterations`: This parameter may be used to apply some median filtering and densification iterations to the input depth images.
  It defaults to 0. It may be good to activate this if using depth cameras with speckle noise. Please be aware that this is slow, since there is currently no GPU implementation for it.

Furthermore, you may want to calibrate for depth distortion (and camera intrinsics) using `--optimize_intrinsics`.
If possible, this should be done once as a calibration step and left fixed afterwards.
This is because this self-calibration impacts the runtime performance and reduces robustness (due to possible self-calibration failures). The calibration can be
saved using `--export_calibration` in command-line mode, or with the "Export intrinsic calibration ..." menu item if running the GUI.
The saved calibration can be loaded using `--import_calibration` in successive runs.
In the calibration dataset, try to make the camera observe surfaces within its whole
sensing volume. If only a part of the volume is covered with observations, then the calibration
may be inappropriate when extrapolated to the remaining parts.
