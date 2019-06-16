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

#include <cuda_runtime.h>
#include <libvis/eigen.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>

#include "badslam/direct_ba.h"

namespace vis {

// Saves the complete SLAM state to a binary file.
// TODO: Use a well-defined endianness.
bool SaveState(
    const BadSlam& slam,
    const std::string& path);

// Loads the complete SLAM state from a binary file (that was saved with
// SaveState()).
// TODO: Use a well-defined endianness.
bool LoadState(
    BadSlam* slam,
    const std::string& path,
    std::function<bool (int, int)> progress_function = nullptr);

// Saves the poses in TUM-RGBD format. Transforms the frame poses such that the
// start frame is at identity.
bool SavePoses(
    const RGBDVideo<Vec3u8, u16>& rgbd_video,
    bool use_depth_timestamps,
    int start_frame,
    const std::string& export_poses_path);

// Saves the depth and color intrinsics, as well as the depth deformation, as
// separate files with the given base path. The filenames are determined by
// appending the following suffixes to the base path: .depth_intrinsics.txt,
// .color_intrinsics.txt, .deformation.txt.
bool SaveCalibration(
    cudaStream_t stream,
    const DirectBA& dense_ba,
    const string& export_base_path);

// Loads a calibration that was saved with SaveCalibration().
bool LoadCalibration(
    DirectBA* dense_ba,
    const string& import_base_path);

// Saves the surfel point cloud in PLY format. Includes colors and normals.
bool SavePointCloudAsPLY(
    cudaStream_t stream,
    const DirectBA& dense_ba,
    const string& export_path);

}
