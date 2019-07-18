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

#include "badslam/undistortion.h"

namespace vis {

void DecideForUndistortedCamera(
    const Camera* camera,
    PinholeCamera4f* undistorted_camera,
    bool avoid_invalid_pixels) {
  float undistorted_initial_intrinsics[4];
  if (camera->type() == Camera::Type::kRadtanCamera9d) {
    const RadtanCamera8d* specific_camera = reinterpret_cast<const RadtanCamera8d*>(camera);
    undistorted_initial_intrinsics[0] = specific_camera->parameters()[5];
    undistorted_initial_intrinsics[1] = specific_camera->parameters()[6];
    undistorted_initial_intrinsics[2] = specific_camera->parameters()[7];
    undistorted_initial_intrinsics[3] = specific_camera->parameters()[8];
  } else {
    CHECK(false);
  }
  *undistorted_camera = PinholeCamera4f(
      camera->width(),
      camera->height(),
      undistorted_initial_intrinsics);
  
  Vec2f min_pixel;
  Vec2f max_pixel;
  if (avoid_invalid_pixels) {
    min_pixel = Vec2f(-numeric_limits<float>::infinity(), -numeric_limits<float>::infinity());
    max_pixel = Vec2f(numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
  } else {
    min_pixel = Vec2f(numeric_limits<float>::infinity(), numeric_limits<float>::infinity());
    max_pixel = Vec2f(-numeric_limits<float>::infinity(), -numeric_limits<float>::infinity());
  }
  
  for (u32 x = 0; x < camera->width(); ++ x) {
    // top point (x, 0)
    Vec3f dir = camera->UnprojectFromPixelCenterConv(Vec2d(x, 0)).cast<float>();
    Vec2f pixel = undistorted_camera->ProjectToPixelCenterConv(dir);
    min_pixel.y() = avoid_invalid_pixels ? std::max(min_pixel.y(), pixel.y()) : std::min(min_pixel.y(), pixel.y());
    
    // bottom point (x, height - 1)
    dir = camera->UnprojectFromPixelCenterConv(Vec2d(x, camera->height() - 1)).cast<float>();
    pixel = undistorted_camera->ProjectToPixelCenterConv(dir);
    max_pixel.y() = avoid_invalid_pixels ? std::min(max_pixel.y(), pixel.y()) : std::max(max_pixel.y(), pixel.y());
  }
  
  for (u32 y = 0; y < camera->height(); ++ y) {
    // left point (0, y)
    Vec3f dir = camera->UnprojectFromPixelCenterConv(Vec2d(0, y)).cast<float>();
    Vec2f pixel = undistorted_camera->ProjectToPixelCenterConv(dir);
    min_pixel.x() = avoid_invalid_pixels ? std::max(min_pixel.x(), pixel.x()) : std::min(min_pixel.x(), pixel.x());
    
    // right point (width - 1, y)
    dir = camera->UnprojectFromPixelCenterConv(Vec2d(camera->width() - 1, y)).cast<float>();
    pixel = undistorted_camera->ProjectToPixelCenterConv(dir);
    max_pixel.x() = avoid_invalid_pixels ? std::min(max_pixel.x(), pixel.x()) : std::max(max_pixel.x(), pixel.x());
  }
  
  // The first pixel center will be placed on the coordinates of min_pixel, the last not farther away than max_pixel.
  int target_undistorted_width = static_cast<int>(max_pixel.x() - min_pixel.x());
  int target_undistorted_height = static_cast<int>(max_pixel.y() - min_pixel.y());
  float undistorted_final_intrinsics[4];
  undistorted_final_intrinsics[0] = undistorted_initial_intrinsics[0];
  undistorted_final_intrinsics[1] = undistorted_initial_intrinsics[1];
  undistorted_final_intrinsics[2] = undistorted_initial_intrinsics[2] - min_pixel.x();
  undistorted_final_intrinsics[3] = undistorted_initial_intrinsics[3] - min_pixel.y();
  *undistorted_camera = PinholeCamera4f(
      target_undistorted_width,
      target_undistorted_height,
      undistorted_final_intrinsics);
}

void UndistortImage(
    const Image<Vec3u8>& image,
    Image<Vec3u8>* undistorted_image,
    const CameraPtr& camera,
    const PinholeCamera4f& undistorted_camera) {
  undistorted_image->SetSize(undistorted_camera.width(), undistorted_camera.height());
  for (u32 y = 0; y < undistorted_image->height(); ++ y) {
    for (u32 x = 0; x < undistorted_image->width(); ++ x) {
      Vec3f dir = undistorted_camera.UnprojectFromPixelCenterConv(Vec2f(x, y));
      Vec2f pixel = camera->ProjectToPixelCenterConv(dir.cast<double>()).cast<float>();
      
      pixel.x() = std::max(0.f, pixel.x());
      pixel.y() = std::max(0.f, pixel.y());
      pixel.x() = std::min(image.width() - 1 - numeric_limits<float>::epsilon(), pixel.x());
      pixel.y() = std::min(image.height() - 1 - numeric_limits<float>::epsilon(), pixel.y());
      
      Vec3f interpolated = image.InterpolateBilinear(pixel);
      (*undistorted_image)(x, y) = (interpolated + Vec3f::Constant(0.5f)).cast<u8>();
    }
  }
}

void CreateUndistortionMap(
    const Camera* camera,
    const PinholeCamera4f& undistorted_camera,
    Image<Vec2f>* undistortion_map) {
  undistortion_map->SetSize(undistorted_camera.width(), undistorted_camera.height());
  for (u32 y = 0; y < undistorted_camera.height(); ++ y) {
    for (u32 x = 0; x < undistorted_camera.width(); ++ x) {
      Vec3f dir = undistorted_camera.UnprojectFromPixelCenterConv(Vec2f(x, y));
      Vec2f pixel = camera->ProjectToPixelCenterConv(dir.cast<double>()).cast<float>();
      
      pixel.x() = std::max(0.f, pixel.x());
      pixel.y() = std::max(0.f, pixel.y());
      pixel.x() = std::min(camera->width() - 1 - numeric_limits<float>::epsilon(), pixel.x());
      pixel.y() = std::min(camera->height() - 1 - numeric_limits<float>::epsilon(), pixel.y());
      
      undistortion_map->at(x, y) = pixel;
    }
  }
}

void UndistortImage(
    const Image<Vec3u8>& image,
    Image<Vec3u8>* undistorted_image,
    const Image<Vec2f>& undistortion_map) {
  undistorted_image->SetSize(undistortion_map.width(), undistortion_map.height());
  for (u32 y = 0; y < undistorted_image->height(); ++ y) {
    for (u32 x = 0; x < undistorted_image->width(); ++ x) {
      Vec2f pixel = undistortion_map(x, y);
      // TODO: We should also cache the interpolation factors for the bilinear
      //       interpolation for even higher speed.
      Vec3f interpolated = image.InterpolateBilinear(pixel);
      (*undistorted_image)(x, y) = (interpolated + Vec3f::Constant(0.5f)).cast<u8>();
    }
  }
}

}
