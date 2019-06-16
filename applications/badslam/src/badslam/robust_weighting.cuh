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

namespace vis {

  // Calculates the value of the robust weight function for a residual, to be
  // used in residual calculation.
__forceinline__ __device__ float TukeyResidual(
    float raw_residual,
    float tukey_parameter) {
  if (fabs(raw_residual) < tukey_parameter) {
    const float quot = raw_residual / tukey_parameter;
    const float term = 1.f - quot * quot;
    return (1 / 6.f) * tukey_parameter * tukey_parameter * (1 - term * term * term);
  } else {
    return (1 / 6.f) * tukey_parameter * tukey_parameter;
  }
}

// Computes the weight used in the weighted least squares update equation.
// Is equal to (1 / residual) * (d TukeyResidual(residual)) / (d residual) .
__forceinline__ __device__ float TukeyWeight(
    float raw_residual,
    float tukey_parameter) {
  if (fabs(raw_residual) < tukey_parameter) {
    const float quot = raw_residual / tukey_parameter;
    const float term = 1.f - quot * quot;
    return term * term;
  } else {
    return 0.f;
  }
}


  // Calculates the value of the robust weight function for a residual, to be
  // used in residual calculation.
__forceinline__ __device__ float HuberResidual(
    float raw_residual,
    float huber_parameter) {
  const float abs_residual = fabs(raw_residual);
  if (abs_residual < huber_parameter) {
    return 0.5f * raw_residual * raw_residual;
  } else {
    return huber_parameter * (abs_residual - 0.5f * huber_parameter);
  }
}

// Computes the weight used in the weighted least squares update equation.
// Is equal to (1 / residual) * (d HuberResidual(residual)) / (d residual) .
__forceinline__ __device__ float HuberWeight(
    float raw_residual,
    float huber_parameter) {
  const float abs_residual = fabs(raw_residual);
  return (abs_residual < huber_parameter) ? 1.f : (huber_parameter / abs_residual);
}

}
