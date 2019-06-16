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

#include "badslam/kernel_opt_intrinsics.h"

#include "badslam/cuda_util.cuh"
#include "badslam/kernels.h"
#include "badslam/keyframe.h"
#include "badslam/surfel_projection.cuh"
#include "badslam/surfel_projection.h"

namespace vis {

void OptimizeIntrinsicsCUDA(
    cudaStream_t stream,
    bool optimize_depth_intrinsics,
    bool optimize_color_intrinsics,
    const vector<shared_ptr<Keyframe>>& keyframes,
    const PinholeCamera4f& color_camera,
    const PinholeCamera4f& depth_camera,
    const DepthParameters& depth_params,
    u32 surfels_size,
    const CUDABuffer<float>& surfels,
    PinholeCamera4f* out_color_camera,
    PinholeCamera4f* out_depth_camera,
    float* a,
    CUDABufferPtr<float>* cfactor_buffer,
    IntrinsicsOptimizationHelperBuffers* buffers) {
  CHECK(optimize_depth_intrinsics || optimize_color_intrinsics);
  CUDA_CHECK();
  if (surfels_size == 0) {
    return;
  }
  
  constexpr int kARows = 4 + 1;
  
  PixelCenterUnprojector depth_center_unprojector(CreatePixelCenterUnprojector(depth_camera));
  
  const int sparse_pixel_count = ((depth_camera.width() - 1) / depth_params.sparse_surfel_cell_size + 1) *
                                 ((depth_camera.height() - 1) / depth_params.sparse_surfel_cell_size + 1);
  
  // Reset accumulation fields to zero.
  // TODO: Clear them all in one kernel call?
  if (optimize_depth_intrinsics) {
    buffers->observation_count.Clear(0, stream);
    buffers->depth_A.Clear(0, stream);
    buffers->depth_B.Clear(0, stream);
    buffers->depth_D.Clear(0, stream);
    buffers->depth_b1.Clear(0, stream);
    buffers->depth_b2.Clear(0, stream);
  }
  if (optimize_color_intrinsics) {
    buffers->color_H.Clear(0, stream);
    buffers->color_b.Clear(0, stream);
  }
  CUDA_CHECK();
  
  // Accumulate H and b.
  for (usize keyframe_id = 0; keyframe_id < keyframes.size(); ++ keyframe_id) {
    const shared_ptr<Keyframe>& keyframe = keyframes[keyframe_id];
    if (!keyframe) {
      continue;
    }
    
    CallAccumulateIntrinsicsCoefficientsCUDAKernel(
        stream,
        optimize_color_intrinsics,
        optimize_depth_intrinsics,
        CreateSurfelProjectionParameters(depth_camera, depth_params, surfels_size, surfels, keyframe.get()),
        CreateDepthToColorPixelCorner(depth_camera, color_camera),
        CreatePixelCornerProjector(color_camera),
        depth_center_unprojector,
        color_camera.parameters()[0], color_camera.parameters()[1],
        keyframe->color_texture(),
        buffers->observation_count.ToCUDA(),
        buffers->depth_A.ToCUDA(),
        buffers->depth_B.ToCUDA(),
        buffers->depth_D.ToCUDA(),
        buffers->depth_b1.ToCUDA(),
        buffers->depth_b2.ToCUDA(),
        buffers->color_H.ToCUDA(),
        buffers->color_b.ToCUDA());
  }
  
//   // DEBUG
//   float depth_b1_cpu[kARows];
//   buffers->depth_b1.DownloadAsync(stream, depth_b1_cpu);
//   for (int i = 0; i < kARows; ++ i) {
//     LOG(ERROR) << "depth_b1_cpu[" << i << "]: " << depth_b1_cpu[i];
//   }
  
  if (optimize_depth_intrinsics) {
    // Solve for the update using the Schur complement trick.
    // Step 1: Compute intermediate matrices
    CallComputeIntrinsicsIntermediateMatricesCUDAKernel(
        stream,
        sparse_pixel_count,
        buffers->depth_A.ToCUDA(),
        buffers->depth_B.ToCUDA(),
        buffers->depth_D.ToCUDA(),
        buffers->depth_b1.ToCUDA(),
        buffers->depth_b2.ToCUDA());
    
    // Step 2: Solve small system on CPU, transfer result back to GPU
    float cpu_matrix_buffer[kARows * (kARows + 1) / 2];
    Eigen::Matrix<float, kARows, kARows> cpu_matrix;
    Eigen::Matrix<float, kARows, 1> cpu_rhs;
    
    buffers->depth_A.DownloadAsync(stream, cpu_matrix_buffer);
    buffers->depth_b1.DownloadAsync(stream, cpu_rhs.data());
    cudaStreamSynchronize(stream);
    int index = 0;
    for (int row = 0; row < kARows; ++ row) {
      for (int col = row; col < kARows; ++ col) {
        cpu_matrix(row, col) = cpu_matrix_buffer[index];
        ++ index;
      }
    }
    
    // Add a weak prior on the a parameter, pulling it towards zero. This intends
    // to avoid issues when the cfactors are close to zero, which makes a unconstrained.
    // Without the prior, a can become very large then and cause numerical issues.
    // The prior residual is: kAPriorWeight * a
    // Its Jacobian wrt. a is: kAPriorWeight
    // TODO: Should this weight depend on something (e.g., residual count) instead
    //       of being constant? A constant, as it is now, can likely be overruled
    //       by enough other residuals.
    constexpr float kAPriorWeight = 10;
    cpu_matrix(4, 4) += kAPriorWeight * kAPriorWeight;
    cpu_rhs(4) += kAPriorWeight * kAPriorWeight * (*a);
    
//     // DEBUG
//     for (int row = 0; row < kARows; ++ row) {
//       for (int col = row + 1; col < kARows; ++ col) {
//         cpu_matrix(col, row) = cpu_matrix(row, col);
//       }
//     }
//     LOG(ERROR) << "DEBUG cpu_matrix: " << std::endl << cpu_matrix;
//     LOG(ERROR) << "DEBUG cpu_rhs: " << std::endl << cpu_rhs;
    
    // NOTE: It is important to use double here, otherwise a_0, a_1 don't change
    //       (alternatively, the parameters in the optimization would have to be
    //        scaled to be roughly equal in scale)
    // NOTE: Do not apply damping here, otherwise the computation of x2 will be
    //       wrong.
    Eigen::Matrix<float, kARows, 1> cpu_x1 = cpu_matrix.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(cpu_rhs.cast<double>()).cast<float>();
    
    //LOG(ERROR) << "DEBUG cpu_x1: " << cpu_x1.transpose();
    
//     // DEBUG
//     for (int i = 0; i < 6; ++ i) {
//       if (fabs(cpu_x1(i) - cpu_x1_test(i)) > fabs(0.02 * cpu_x1_test(i))) {
//         LOG(ERROR) << "Global parameter update differs: schur: " << cpu_x1(i) << " vs non-schur: " << cpu_x1_test(i) << " vs schur(cpu): " << x1(i);
//       }
//     }
//     // END DEBUG
    
    float new_depth_fx = 1.0f / (depth_center_unprojector.fx_inv - cpu_x1(0));
    float new_depth_fy = 1.0f / (depth_center_unprojector.fy_inv - cpu_x1(1));
    float new_depth_cx = -(new_depth_fx * (depth_center_unprojector.cx_inv - cpu_x1(2))) + 0.5f;
    float new_depth_cy = -(new_depth_fy * (depth_center_unprojector.cy_inv - cpu_x1(3))) + 0.5f;
    float new_depth_camera_parameters[4] = {
        new_depth_fx,
        new_depth_fy,
        new_depth_cx,
        new_depth_cy};
    *out_depth_camera = PinholeCamera4f(depth_camera.width(), depth_camera.height(), new_depth_camera_parameters);
    
    *a -= cpu_x1(4);
    
    buffers->depth_b1.UploadAsync(stream, cpu_x1.data());  // re-use b1 for x1
    CUDA_CHECK();
    
    // Step 3: Solve the rest on the GPU with matrix multiplication
//     // DEBUG
//     Image<float> cfactor_buffer_old_cpu((*cfactor_buffer)->width(), (*cfactor_buffer)->height());
//     (*cfactor_buffer)->DownloadAsync(stream, &cfactor_buffer_old_cpu);
//     // END DEBUG
    
    CallSolveForPixelIntrinsicsUpdateCUDAKernel(
        stream,
        sparse_pixel_count,
        buffers->observation_count.ToCUDA(),
        buffers->depth_B.ToCUDA(),
        buffers->depth_D.ToCUDA(),
        buffers->depth_b1.ToCUDA(),  // x1
        (*cfactor_buffer)->ToCUDA());
    
//     // DEBUG
//     Image<float> cfactor_buffer_new_cpu((*cfactor_buffer)->width(), (*cfactor_buffer)->height());
//     (*cfactor_buffer)->DownloadAsync(stream, &cfactor_buffer_new_cpu);
//     
//     int num_cfactor_warnings = 0;
//     for (int y = 0; y < cfactor_buffer_new_cpu.height(); ++ y) {
//       for (int x = 0; x < cfactor_buffer_new_cpu.width(); ++ x) {
//         float cfactor_change = cfactor_buffer_new_cpu(x, y) - cfactor_buffer_old_cpu(x, y);
//         float cfactor_non_schur_update = - cpu_x1_test(kARows + x + y * cfactor_buffer_new_cpu.width());
//         if (fabs(cfactor_change - cfactor_non_schur_update) > 1e-5f) {
//           LOG(ERROR) << "cfactor change differs: schur: " << cfactor_change << " vs non-schur: " << cfactor_non_schur_update << " vs schur(cpu): " << x2(x + y * cfactor_buffer_new_cpu.width());
//           ++ num_cfactor_warnings;
//           if (num_cfactor_warnings > 10) {
//             LOG(ERROR) << "More than 10 cfactor warnings, aborting check.";
//             break;
//           }
//         }
//       }
//       if (num_cfactor_warnings > 10) {
//         break;
//       }
//     }
//     // END DEBUG
    
//     
//     // DEBUG: Show observation count.
//     vector<u32> observation_count_cpu(sparse_pixel_count);
//     buffers->observation_count.DownloadAsync(stream, observation_count_cpu.data());
//     cudaStreamSynchronize(stream);
//     
//     Image<u8> observation_count_image((*cfactor_buffer)->width(), (*cfactor_buffer)->height());
//     for (u32 i = 0; i < sparse_pixel_count; ++ i) {
//       u32 y = i / (*cfactor_buffer)->width();
//       u32 x = i % (*cfactor_buffer)->width();
//       
//       observation_count_image(x, y) = observation_count_cpu[i];
//     }
//     
//     static shared_ptr<ImageDisplay> observation_count_debug_display(new ImageDisplay());
//     observation_count_debug_display->Update(observation_count_image, "intrinsics observation count debug");
  }
  
  if (optimize_color_intrinsics) {
    float cpu_matrix_buffer[4 * (4 + 1) / 2];
    Eigen::Matrix<float, 4, 4> cpu_matrix;
    Eigen::Matrix<float, 4, 1> cpu_rhs;
    
    buffers->color_H.DownloadAsync(stream, cpu_matrix_buffer);
    buffers->color_b.DownloadAsync(stream, cpu_rhs.data());
    cudaStreamSynchronize(stream);
    int index = 0;
    for (int row = 0; row < 4; ++ row) {
      for (int col = row; col < 4; ++ col) {
        cpu_matrix(row, col) = cpu_matrix_buffer[index];
        ++ index;
      }
    }
    
    Eigen::Matrix<float, 4, 1> x = cpu_matrix.cast<double>().selfadjointView<Eigen::Upper>().ldlt().solve(cpu_rhs.cast<double>()).cast<float>();
    
    float new_color_camera_parameters[4] = {
        color_camera.parameters()[0] - x(0),
        color_camera.parameters()[1] - x(1),
        color_camera.parameters()[2] - x(2),
        color_camera.parameters()[3] - x(3)};
    *out_color_camera = PinholeCamera4f(color_camera.width(), color_camera.height(), new_color_camera_parameters);
  }
}

}
