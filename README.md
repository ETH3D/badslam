# BAD SLAM #

## Overview ##

BAD SLAM is a real-time approach for Simultaneous Localization and Mapping (SLAM) for RGB-D cameras.
Supported platforms are Linux and Windows. The software requires an NVidia graphics card with CUDA compute capability 5.3 or later
(however, it would be easy to lower this requirement).

This repository contains the
[BAD SLAM application](https://github.com/ETH3D/badslam/tree/master/applications/badslam)
and the library it is based on,
[libvis](https://github.com/ETH3D/badslam/tree/master/libvis).
The library is work-in-progress and it is not recommended to use it for other projects at this point.

The application and library code is licensed under the BSD license, but please
also notice the licenses of the included or externally used third-party components.

If you use the provided code for research, please cite the paper describing the approach:

[Thomas Schöps, Torsten Sattler, Marc Pollefeys, "BAD SLAM: Bundle Adjusted Direct RGB-D SLAM", CVPR 2019.](http://openaccess.thecvf.com/content_CVPR_2019/html/Schops_BAD_SLAM_Bundle_Adjusted_Direct_RGB-D_SLAM_CVPR_2019_paper.html)

The Windows port and Kinect-for-Azure (K4A) integration has been contributed by Silvano Galliani (Microsoft AI & Vision Zurich).


## Screenshots & Videos ##

| Main window   | Surfel normals display | Keyframe inspection |
| ------------- | ---------------------- | ------------------- | 
| ![Main Window](applications/badslam/doc/main_window.png?raw=true) | ![Settings](applications/badslam/doc/surfel_normals.png?raw=true) | ![Keyframe dialog](applications/badslam/doc/keyframe_dialog.png?raw=true) |

* [Oral presentation of the conference paper by Torsten Sattler at CVPR 2019](https://www.youtube.com/watch?v=0lLnHe0xbZE)
* [Short demonstration of some of the SLAM GUI features](https://youtu.be/g3yD9qmDW4M)


## Camera requirements ##

Please keep in mind that BAD SLAM has been designed for
high-quality RGB-D videos and is likely to perform badly (no pun intended) on
lower-quality RGB-D videos. For more details, see the [documentation on camera compatibility](https://github.com/ETH3D/badslam/blob/master/applications/badslam/doc/camera.md).


## Pre-built binaries ##

Binaries are available for download as [GitHub releases](https://github.com/ETH3D/badslam/releases).

#### Windows ####

For Windows, an executable compiled with Visual Studio 2019 is provided.
It is also required to download the loop closure resource files as described below in this ReadMe, or loop closures will be disabled. In addition,
performing CUDA block-size autotuning as also described below is recommended.

If the executable fails to start due to missing DLLs, try installing the latest [Visual C++ redistributable files for Visual Studio 2019](https://aka.ms/vs/16/release/vc_redist.x64.exe).

#### Linux ####

For Linux, an [AppImage](https://appimage.org/) is provided. Please note that it is also required to download
the loop closure resource files as described below in this ReadMe, or loop closures will be disabled. In addition,
performing CUDA block-size autotuning as also described below is recommended.

In case you encounter an error like
```
./badslam: relocation error: [...]/libQt5DBus.so.5: symbol dbus_message_get_allow_interactive_authorization, version LIBDBUS_1_3 not defined in file libdbus-1.so.3 with link time reference
```
then your dbus library is too old. This can be fixed by downloading a recent version and setting `LD_LIBRARY_PATH` to the directory containing these files before starting the AppImage.


## Building ##

Building has been tested on Ubuntu 14.04 and Ubuntu 18.04 (with gcc), and on Windows (with Visual Studio 2019 and 2017).

The following external dependencies are required.

| Dependency   | Version(s) known to work |
| ------------ | ------------------------ |
| [Boost](https://www.boost.org/) | 1.54.0 |
| [CUDA](https://developer.nvidia.com/cuda-downloads) | 8, 9.1, 10.1 |
| [DLib](https://github.com/dorian3d/DLib) | commit b6c28fb |
| [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) | 3.3.7 |
| [g2o](https://github.com/RainerKuemmerle/g2o) |  |
| [GLEW](http://glew.sourceforge.net/build.html) |  |
| [GTest](https://github.com/google/googletest) |  |
| [OpenCV](https://opencv.org/) | 3.1.0, 3.2.0, 3.4.5, 3.4.6; 4.x does NOT work without changes |
| [OpenGV](https://github.com/laurentkneip/opengv) | in Visual Studio 2017 it compiles only in debug mode |
| [Qt](https://www.qt.io/) | 5.12.0; minimum version: 5.8 |
| [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) |  |
| [zlib](https://zlib.net/) |  |

Notice that OpenCV is only required as a dependency for loop detection by DLib.
Also notice that there are (at least) two different libraries with the name DLib,
so be sure to install the correct one.

The following external dependencies are optional.

| Dependency   | Purpose |
| ------------ | ------- | 
| [librealsense2](https://github.com/IntelRealSense/librealsense) | Live input from RealSense D400 series depth cameras (tested with the D435 only). |
| [Structure SDK](https://structure.io/developers) | Live input from Structure Core cameras (tested with the color version only). To use this, set the SCSDK_ROOT CMake variable to the SDK path. |
| [k4a & k4arecord](https://github.com/Microsoft/Azure-Kinect-Sensor-SDK) | Live input from Azure Kinect cameras. |


#### Build instructions for Linux ####

Since OpenGV (at the time of writing) always uses the `-march=native` flag,
both BAD SLAM and g2o must use this as well. (For g2o, check that the
`BUILD_WITH_MARCH_NATIVE` CMake option is set to ON.) If there are inconsistencies, the program
will likely crash when OpenGV or g2o functionality is used (i.e., at loop closures).

After obtaining all dependencies, the application can be built with CMake, for example as follows:

```bash
mkdir build_RelWithDebInfo
cd build_RelWithDebInfo
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="-arch=sm_61" ..
make -j badslam  # Reduce the number of threads if running out of memory, e.g., -j3
```

Make sure to specify suitable CUDA architecture(s) in CMAKE_CUDA_FLAGS.
Common settings would either be the CUDA architecture of your graphics card only (in case
you only intend to run the compiled application on the system it was compiled on), or a range of virtual
architectures (in case the compiled application is intended for distribution).
See the [corresponding CUDA documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture).

Optionally, after building, the unit tests can be run, which test some of the
bundle adjustment functionality. To do so, build and run the following executable:

```
make -j badslam_test
./build_RelWithDebInfo/applications/badslam/badslam_test
```

All tests should pass. Troubleshooting:

* If you get a CUDA error like "too many resources requested for launch", probably
  a default CUDA kernel block size does not work for your GPU. See below for
  block-size tuning. The application has been tested on GTX 1080 and GTX 1070 GPUs.
* If the `Optimization.PoseGraphOptimizer` test crashes, for example with an error
  message like "Cholesky failure", then please verify that your build of g2o has
  the `BUILD_WITH_MARCH_NATIVE` CMake option set to ON, and that BAD SLAM actually
  uses this install of g2o.


#### Build instructions for Windows ####

The application can be built by creating a Visual Studio 2019 solution for it with CMake,
then compiling the "badslam" project in this solution.

It seemed that a workaround was required to prevent some unresolved external symbols in g2o_csparse_extension
(for example, duplicating the problematic functions into g2o_solver_csparse).

Make sure to specify suitable CUDA architecture(s) in CMAKE_CUDA_FLAGS.
Common settings would either be the CUDA architecture of your graphics card only (in case
you only intend to run the compiled application on the system it was compiled on), or a range of virtual
architectures (in case the compiled application is intended for distribution).
See the [corresponding CUDA documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation-gpu-architecture).


## Dataset format ##

For CUDA block-size tuning (see below), at least one dataset should be obtained,
even if one intends to run the program with live input.

The program supports datasets in the format of the [ETH3D SLAM Benchmark](https://www.eth3d.net/slam_overview) for RGB-D videos.
This is an extension of the format introduced by the
[TUM RGB-D benchmark](https://vision.in.tum.de/data/datasets/rgbd-dataset),
containing two small additions:

* The original format does not specify the intrinsic camera calibration.
  BAD SLAM thus additionally expects a file `calibration.txt` in the
  dataset directory, consisting of a single line of text structured as follows:
  ```
  fx fy cx cy
  ```
  These values specify the parameters for the pinhole projection (fx * x + cx, fy * y + cy).
  The coordinate system convention for cx and cy is that the origin (0, 0) of pixel coordinates is at
  the **center** of the top-left pixel in the image.
* The [associate.py](https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py)
  tool from the benchmark must be run as follows to associate
  the color and depth images:
  ```
  python associate.py rgb.txt depth.txt > associated.txt
  ```


## Initial setup ##

After building the executable and obtaining a dataset, there are two more steps to be done before running the program.

First, the resource files for loop closure handling should be set up
(unless the parameter `--no_loop_detection` is used to disable loop detection).
Download the [resource files](https://drive.google.com/file/d/1MpZwPjXDAUxKfSTpeCjG0PAUpaeWuo7D) of the [DLoopDetector](https://github.com/dorian3d/DLoopDetector) demo.
The two relevant files from this archive, `brief_k10L6.voc` and `brief_pattern.yml`, must be extracted into
a directory named "resources" in the application executable's directory (or an
analogous symlink must be created), for example:

```
- build_RelWithDebInfo
  - applications
    - badslam
      - badslam (executable file)
      - resources
        - brief_k10L6.voc (notice that this is compressed in the archive and needs to be extracted separately)
        - brief_pattern.yml
```

Second, the CUDA kernel block size auto-tuning should be run. This is not strictly
required in case the default sizes work for your GPU, but strongly recommended.
This step serves two purposes:

* Sometimes, CUDA kernels won't launch with a given thread block size since this
  would require too many resources. Block size auto-tuning determines and avoids
  those problematic configurations.
* The best block sizes to call CUDA kernels may vary between different graphics
  cards, and the best way to figure them out is to benchmark it, which the
  tuning does.

To test your GPU, run the badslam executable with the provided tuning
script on any dataset in sequential mode:

```
python scripts/auto_tune_parameters.py <path_to_badslam_executable> <path_to_dataset> --sequential_ba --sequential_loop_detection
```

The script will run the program multiple times using different parameters and
measure the runtime, i.e., do not run another computing task at the same time to
not influence the measurements. It should output a file `auto_tuning_result.txt`
and intermediate files `auto_tuning_iteration_X.txt`. Move the result file into
the `resources` directory used by BAD SLAM (where the loop detector resources
are also stored in). The file will be loaded automatically if it exists in this
directory. The intermediate files can be deleted.

Since the program runs multiple times, you may want to limit the number of
frames it runs on to speed it up with `--end_frame`. Also, please notice that
tuning data will only be gathered for CUDA kernels that run during the tuning.
If later other kernels run during the actual program invocation, they will still
use the default block size. So, if for example you want to tune the PCG-related
kernels instead of those for alternating optimization, then you need to pass the
corresponding parameter `--use_pcg` in the tuning call. Since the tuning result
files are simple plain text files, the results of multiple tuning runs with
different parameters could be easily merged to create a tuning file that covers
all kernels. Doing this automatically would be a possible future addition to the
tuning script.


## Running ##

The simplest way to start the program is without any command-line arguments:

```
./build_RelWithDebInfo/applications/badslam/badslam
```

It will show a settings window then that allows to select a dataset or live input, and
allows to adjust a variety of parameters.

Alternatively, the program can be run without visualization by specifying
all parameters on the command line. If parameters are given on the command line,
the visualization can be used with the `--gui` flag (to start showing the settings window)
or the `--gui_run` flag (to start running immediately).

For example, to immediately start running SLAM on a dataset in the GUI, use:

```
./build_RelWithDebInfo/applications/badslam/badslam <dataset_path> --gui_run
```

See the [documentation on command line parameters](https://github.com/ETH3D/badslam/blob/master/applications/badslam/doc/command_line.md) for more details.

The first time the program runs on a dataset, the performance might be
limited by the time it takes to read the image files from the hard disk
(unless the dataset is on an SSD, or is already cached because the files were written recently).
Subsequent runs should be faster as long as the files remain cached.

Please also notice that the real-time mode with parallel odometry and bundle adjustment, despite being the default,
was added late in the development process and should be considered potentially unstable (in particular
when optimizing the depth camera's deformation, which lacks synchronization for the access to a GPU buffer).
Thus, to possibly increase robustness, use the `--sequential_ba` parameter.
Live operation may still be simulated by also specifying `--target_frame_rate <desired_fps>`.

### Docker ###
To build the image, do:

```
$ docker build  --build-arg CUDA_ARCH="DESIRED_ARCH" -t eth3d/badslam .
```

where `DESIRED_ARCH` corresponds to the CUDA architecture you wish to build
with.

To run the image using an example dataset, download & unzip invoke:

```
$ docker run  --gpus all -it -e DISPLAY  -v /tmp/.X11-unix:/tmp/.X11-unix:ro  --mount type=bind,source=ABSOLUTE_PATH_TO_DATASET,target=/datasets eth3d/badslam  /bin/bash
```

Using the "einstein" dataset as an example, you could run

```
$ ./applications/badslam/badslam /datasets/einstein_1 --export_reconstruction einstein.ply && meshlab einstein.ply
```

Note: if you observe something like:

```
qt.qpa.xcb: could not connect to display :0.0  
```

Make sure to:

```
$ xhost + 
```

in your terminal before running the container.

## Extending BAD SLAM ##

Contributions to this open source project are very welcome. Please try to
follow the existing coding style (which is loosely inspired by the Google C++
coding style, but somewhat relaxed in some aspects).

If you are interested in using the direct bundle adjustment component
without SLAM, then the intrinsics optimization unit test might be a good
starting point, showing how to set up keyframes and perform optimization.
It is at `applications/badslam/src/badslam/test/test_intrinsics_optimization_[photometric/geometric]_residual.cc`.

If you plan to change the cost function used for bundle adjustment, you may
want to have a look at `scripts/jacobians_derivation.py`. This script
automatically computes the Jacobians required for optimization from a
specification of the residuals in Python. It also outputs somewhat optimized
C++ functions to compute the residual, the Jacobian, and both the residual and
Jacobian at the same time. The script requires sympy to run. Its main limitation
is that it operates on a symbolic representation of the residual (instead of on
the algorithm for residual computation, as an autodiff tool would do), which means
that its internal residual term may become huge. This may cause excessive
runtimes of the script for more complex residuals. You can try removing the simplify() calls
in `jacobian_functions.py` to speed it up, while applying less simplification to
the resulting expressions.


## Differences to the paper ##

The open source version of the code has undergone strong refactoring compared to
the version used to produce the results in the paper, many new features have
been added, and many fixes were done. The photometric residual used for global optimization is slightly
different: Instead of using the gradient magnitude as the photometric descriptor,
the two components of the gradient are used separately. For these reasons,
it should not be expected that the code reproduces the results in the paper
exactly, however the results should be similar.
