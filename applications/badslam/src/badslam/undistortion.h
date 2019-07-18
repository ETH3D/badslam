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

#include <libvis/camera.h>
#include <libvis/image.h>
#include <libvis/libvis.h>

namespace vis {

// Given an arbitrary camera model, determines a pinhole camera that observes
// as much of the image as possible (without rotation) while avoiding to include
// any unobserved regions if avoid_invalid_pixels is set.
void DecideForUndistortedCamera(
    const Camera* camera,
    PinholeCamera4f* undistorted_camera,
    bool avoid_invalid_pixels);

/// Undistortion function that does not use an undistortion map (thus is likely slow).
void UndistortImage(
    const Image<Vec3u8>& image,
    Image<Vec3u8>* undistorted_image,
    const CameraPtr& camera,
    const PinholeCamera4f& undistorted_camera);

void CreateUndistortionMap(
    const Camera* camera,
    const PinholeCamera4f& undistorted_camera,
    Image<Vec2f>* undistortion_map);

/// Undistortion function that uses an undistortion map.
void UndistortImage(
    const Image<Vec3u8>& image,
    Image<Vec3u8>* undistorted_image,
    const Image<Vec2f>& undistortion_map);

}
