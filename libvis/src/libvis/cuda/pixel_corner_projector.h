// Copyright 2017, 2019 ETH Zürich, Thomas Schöps
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

#include "libvis/camera.h"
#include "libvis/cuda/cuda_buffer.h"
#include "libvis/cuda/pixel_corner_projector.cuh"
#include "libvis/eigen.h"

namespace vis {

struct PixelCornerProjector {
  PixelCornerProjector(const Camera& camera)
      : grid(reinterpret_cast<const NonParametricBicubicProjectionCamerad&>(camera).parameters()[1],
             reinterpret_cast<const NonParametricBicubicProjectionCamerad&>(camera).parameters()[0]) {
    d.type = static_cast<int>(camera.type());
    d.width = camera.width();
    d.height = camera.height();
    
    // NOTE: Commented out (also further below) for shorter compile times. Find a better solution ...
  //     if (camera.type() == Camera::Type::kPinholeCamera4f) {
  //       const PinholeCamera4f& pinhole_camera = reinterpret_cast<const PinholeCamera4f&>(camera);
  //       
  //       fx = pinhole_camera.parameters()[0];
  //       fy = pinhole_camera.parameters()[1];
  //       cx = pinhole_camera.parameters()[2];
  //       cy = pinhole_camera.parameters()[3];
  //     } else if (camera.type() == Camera::Type::kRadtanCamera8d) {
  //       const RadtanCamera8d& radtan_camera = reinterpret_cast<const RadtanCamera8d&>(camera);
  //       
  //       k1 = radtan_camera.parameters()[0];
  //       k2 = radtan_camera.parameters()[1];
  //       p1 = radtan_camera.parameters()[2];
  //       p2 = radtan_camera.parameters()[3];
  //       
  //       fx = radtan_camera.parameters()[4];
  //       fy = radtan_camera.parameters()[5];
  //       cx = radtan_camera.parameters()[6];
  //       cy = radtan_camera.parameters()[7];
  //     } else if (camera.type() == Camera::Type::kThinPrismFisheyeCamera12d) {
  //       const ThinPrismFisheyeCamera12d& thinprism_camera = reinterpret_cast<const ThinPrismFisheyeCamera12d&>(camera);
  //       
  //       k1 = thinprism_camera.parameters()[0];
  //       k2 = thinprism_camera.parameters()[1];
  //       k3 = thinprism_camera.parameters()[2];
  //       k4 = thinprism_camera.parameters()[3];
  //       p1 = thinprism_camera.parameters()[4];
  //       p2 = thinprism_camera.parameters()[5];
  //       sx1 = thinprism_camera.parameters()[6];
  //       sy1 = thinprism_camera.parameters()[7];
  //       
  //       fx = thinprism_camera.parameters()[8];
  //       fy = thinprism_camera.parameters()[9];
  //       cx = thinprism_camera.parameters()[10];
  //       cy = thinprism_camera.parameters()[11];
  //     }
    
    if (camera.type() == Camera::Type::kNonParametricBicubicProjectionCamerad) {
      const NonParametricBicubicProjectionCamerad& np_camera = reinterpret_cast<const NonParametricBicubicProjectionCamerad&>(camera);
      
      d.resolution_x = np_camera.parameters()[0];
      d.resolution_y = np_camera.parameters()[1];
      d.min_nx = np_camera.parameters()[2];
      d.min_ny = np_camera.parameters()[3];
      d.max_nx = np_camera.parameters()[4];
      d.max_ny = np_camera.parameters()[5];
      
      vector<float2> grid_data(d.resolution_x * d.resolution_y);
      for (int i = 0; i < d.resolution_x * d.resolution_y; ++ i) {
        grid_data[i] = make_float2(
            np_camera.parameters()[6 + 2 * i + 0],
            np_camera.parameters()[6 + 2 * i + 1]);
      }
      grid.UploadAsync(/*stream*/ 0, grid_data.data());
    } else {
      LOG(FATAL) << "Constructor called with unsupported camera type";
    }
  }
  
  PixelCornerProjector(const PixelCornerProjector& other) = delete;
  
  __forceinline__ PixelCornerProjector_ ToCUDA() {
    return d;
  }
  
 private:
  CUDABuffer<float2> grid;
  PixelCornerProjector_ d;
};

}
