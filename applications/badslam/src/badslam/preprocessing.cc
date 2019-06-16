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

#include "badslam/preprocessing.h"

#include <iomanip>

#include "badslam/cuda_depth_processing.cuh"
#include "badslam/cuda_image_processing.cuh"
#include "badslam/util.cuh"

namespace vis {

// Runs a median filter on the depth map to perform denoising and fill-in.
void MedianFilterAndDensifyDepthMap(const Image<u16>& input, Image<u16>* output) {
  vector<u16> values;
  
  constexpr int kRadius = 1;
  constexpr int kMinNeighbors = 2;
  
  for (int y = 0; y < static_cast<int>(input.height()); ++ y) {
    for (int x = 0; x < static_cast<int>(input.width()); ++ x) {
      values.clear();
      
      int dy_end = std::min<int>(input.height() - 1, y + kRadius);
      for (int dy = std::max<int>(0, static_cast<int>(y) - kRadius);
           dy <= dy_end;
           ++ dy) {
        int dx_end = std::min<int>(input.width() - 1, x + kRadius);
        for (int dx = std::max<int>(0, static_cast<int>(x) - kRadius);
             dx <= dx_end;
             ++ dx) {
          if (input(dx, dy) != 0) {
            values.push_back(input(dx, dy));
          }
        }
      }
      
      if (values.size() >= kMinNeighbors) {
        std::sort(values.begin(), values.end());  // NOTE: slow, need to get center element only
        if (values.size() % 2 == 0) {
          // Take the element which is closer to the average.
          float sum = 0;
          for (u16 value : values) {
            sum += value;
          }
          float average = sum / values.size();
          
          float prev_diff = std::fabs(values[values.size() / 2 - 1] - average);
          float next_diff = std::fabs(values[values.size() / 2] - average);
          (*output)(x, y) = (prev_diff < next_diff) ? values[values.size() / 2 - 1] : values[values.size() / 2];
        } else {
          (*output)(x, y) = values[values.size() / 2];
        }
      } else {
        (*output)(x, y) = input(x, y);
      }
    }
  }
}

}
