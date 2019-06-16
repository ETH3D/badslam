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

#include <libvis/libvis.h>
#include <libvis/sophus.h>

namespace vis {

// Attempts to find good parameters to determine convergence of direct pose
// estimation.
void RunConvergenceAnalysis(const std::string& convergence_file_path);

// Attempts to determine whether direct pose estimation is converged based on
// the pose update x.
inline bool IsScale1PoseEstimationConverged(const Eigen::Matrix<float, 6, 1>& x) {
  constexpr float translation_threshold = 1e-06;
  constexpr float rotation_threshold = 1e-07;
  // Scale the rotation part to the translation scale and then use the translation criterion on the whole vector norm.
  Eigen::Matrix<float, 6, 1> scaled_x = x;
  scaled_x.bottomRows<3>() *= translation_threshold / rotation_threshold;
  return (scaled_x.squaredNorm() < translation_threshold);
}

// Attempts to determine whether direct pose estimation is converged based on
// the pose update x.
inline bool IsScaleNPoseEstimationConverged(const Eigen::Matrix<float, 6, 1>& x, float scaling_factor) {
  constexpr float translation_threshold = 1e-08;
  constexpr float rotation_threshold = 1e-08;
  // Scale the rotation part to the translation scale and then use the translation criterion on the whole vector norm.
  Eigen::Matrix<float, 6, 1> scaled_x = x;
  scaled_x.bottomRows<3>() *= translation_threshold / rotation_threshold;
  return (scaled_x.squaredNorm() < scaling_factor * scaling_factor * translation_threshold);
}

}
