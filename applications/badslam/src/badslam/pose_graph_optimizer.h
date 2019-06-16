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

#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <libvis/libvis.h>
#include <libvis/rgbd_video.h>
#include <libvis/sophus.h>

#include "badslam/kernels.h"
#include "badslam/direct_ba.h"
#include "badslam/render_window.h"
#include "badslam/keyframe.h"

namespace vis {

// Performs pose graph optimization (using a set of pairwise constraints between
// keyframes) to change keyframe poses. Information matrices are all set to
// identity (i.e., all constraints get the same weight).
class PoseGraphOptimizer {
 public:
  // Constructor. If add_current_state_odometry_constraints is true, constraints
  // will be added between keyframes with subsequent indices using their current
  // relative poses as a "measurement".
  PoseGraphOptimizer(DirectBA* dense_ba, bool add_current_state_odometry_constraints);
  
  // Destructor.
  ~PoseGraphOptimizer();
  
  // Adds a pairwise pose constraint between keyframes with indices A and B.
  void AddEdge(u32 id_A, u32 id_B, const SE3f& A_tr_B);
  
  // Runs the optimization (after adding all desired constraints).
  void Optimize();
  
  // Retrieves the resulting absolute pose for the keyframe with the given id.
  inline SE3f GetGlobalTFrame(u32 keyframe_id) const {
    return SE3f(reinterpret_cast<const g2o::VertexSE3*>(optimizer.vertex(keyframe_id))->estimate().matrix().cast<float>());
  }
  
 private:
  g2o::SparseOptimizer optimizer;
};

}
