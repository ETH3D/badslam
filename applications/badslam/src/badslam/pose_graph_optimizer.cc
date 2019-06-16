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

#include "badslam/pose_graph_optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/slam3d/edge_se3.h>

using namespace g2o;

namespace vis {

typedef BlockSolver< BlockSolverTraits<VertexSE3::Dimension, -1> > SlamBlockSolver;
typedef LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

// It seems that g2o at some point changed the constructor argument of
// OptimizationAlgorithmGaussNewton and SlamBlockSolver. Use some illegible C++
// hack to find out which version we have to use. Apparently C++ does not
// consider a template function overload if a parameter from it resolves to an
// invalid type, which should happen if the enable_if gets false as argument.
// This allows us to figure out which constructor exists.
// Variant 1:
template <typename T>
SlamBlockSolver* AllocateSlamBlockSolver(
    typename std::enable_if<std::is_constructible<T, SlamLinearSolver*>::value>::type* = 0) {
  auto linearSolver = new SlamLinearSolver();
  linearSolver->setBlockOrdering(false);
  return new T(linearSolver);
}

// Variant 2:
template <typename T>
unique_ptr<SlamBlockSolver> AllocateSlamBlockSolver(
    typename std::enable_if<!std::is_constructible<T, SlamLinearSolver*>::value>::type* = 0) {
  auto linearSolver = unique_ptr<SlamLinearSolver>(new SlamLinearSolver());
  linearSolver->setBlockOrdering(false);
  return unique_ptr<SlamBlockSolver>(new T(std::move(linearSolver)));
}

PoseGraphOptimizer::PoseGraphOptimizer(DirectBA* dense_ba, bool add_current_state_odometry_constraints) {
  // How the problem is mapped to g2o:
  // The nodes get the global_T_frame transformation.
  // The edges get A as "from" (vertices()[0]),
  //               B as "to" (vertices()[1]), and
  //               A_tr_B as measurement.
  
  optimizer.setAlgorithm(
      new OptimizationAlgorithmGaussNewton(
          AllocateSlamBlockSolver<SlamBlockSolver>()));
  
  // Add keyframe nodes and, if specified, odometry constraints that use the current state
  Keyframe* prev_keyframe = nullptr;
  for (usize i = 0; i < dense_ba->keyframes().size(); ++ i) {
    Keyframe* keyframe = dense_ba->keyframes()[i].get();
    if (!keyframe) {
      continue;
    }
    
    VertexSE3* keyframe_node = new VertexSE3();
    keyframe_node->setId(keyframe->id());
    keyframe_node->setEstimate(Isometry3d(keyframe->global_T_frame().matrix().cast<double>()));
    optimizer.addVertex(keyframe_node);
    
    if (add_current_state_odometry_constraints && prev_keyframe) {
      EdgeSE3* odometry = new EdgeSE3();
      odometry->vertices()[0] = optimizer.vertex(prev_keyframe->id());
      odometry->vertices()[1] = optimizer.vertex(keyframe->id());
      SE3f transformation = prev_keyframe->frame_T_global() * keyframe->global_T_frame();
      odometry->setMeasurement(Isometry3d(transformation.matrix().cast<double>()));
      odometry->setInformation(Matrix<double, 6, 6>::Identity());
      optimizer.addEdge(odometry);
    }
    
    prev_keyframe = keyframe;
  }
  
  // Fix the first pose to account for gauge freedom.
  optimizer.vertex(0)->setFixed(true);
}

PoseGraphOptimizer::~PoseGraphOptimizer() {
  // Free the g2o graph memory.
  optimizer.clear();
}

void PoseGraphOptimizer::AddEdge(u32 id_A, u32 id_B, const SE3f& A_tr_B) {
  EdgeSE3* edge = new EdgeSE3();
  edge->vertices()[0] = optimizer.vertex(id_A);
  edge->vertices()[1] = optimizer.vertex(id_B);
  edge->setMeasurement(Isometry3d(A_tr_B.matrix().cast<double>()));
  edge->setInformation(Matrix<double, 6, 6>::Identity());
  optimizer.addEdge(edge);
}

void PoseGraphOptimizer::Optimize() {
  // optimizer.setVerbose(true);
  
  LOG(INFO) << "- Performing pose graph optimization ...";
  optimizer.initializeOptimization();
  constexpr int kMaxIterations = 20;  // TODO: Tune this number?
  optimizer.optimize(kMaxIterations);
  LOG(INFO) << "- Pose graph optimization done";
  
//   // DEBUG: Show optimized frame poses
//   constexpr bool kDebugOptimizedFramePoses = false;
//   if (kDebugOptimizedFramePoses) {
//     vector<Mat4f> keyframe_poses;
//     vector<Keyframe*> keyframe_ptrs;
//     keyframe_poses.reserve(dense_ba->keyframes().size());
//     for (usize i = 0; i < dense_ba->keyframes().size(); ++ i) {
//       Keyframe* keyframe = dense_ba->keyframes()[i].get();
//       if (!keyframe) {
//         continue;
//       }
//       keyframe_poses.push_back(
//           //keyframe->global_T_frame().matrix()
//           reinterpret_cast<VertexSE3*>(optimizer.vertex(keyframe->id()))->estimate().matrix().cast<float>()
//           );
//       keyframe_ptrs.push_back(keyframe);
//     }
//     
//     unique_lock<mutex> render_mutex_lock(render_window->render_mutex());
//     render_window->SetKeyframePosesNoLock(std::move(keyframe_poses), std::move(keyframe_ptrs));
//     render_mutex_lock.unlock();
//     
//     render_window->RenderFrame();
//     
//     std::getchar();
//   }
}

}
